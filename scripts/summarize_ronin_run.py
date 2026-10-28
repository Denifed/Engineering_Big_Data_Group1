"""Verify saved predictions and generate a Chinese report and local gallery."""
import argparse
import csv
import hashlib
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    out = args.run.resolve()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "ronin-master" / "source"))
    from metric import compute_ate_rte
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    with (out / "sequence_metrics.csv").open(encoding="utf-8") as h:
        rows = list(csv.DictReader(h))
    verification = {}
    for split in ("seen_test", "unseen_test"):
        squared = absolute = 0.
        count = 0
        sequence_checks = []
        for row in [r for r in rows if r["split"] == split]:
            path = out / split / (row["sequence"] + "_predictions.npz")
            with np.load(path, allow_pickle=False) as d:
                pred, truth = d["predicted_velocity"], d["target_velocity"]
                assert pred.shape == truth.shape == (int(row["windows"]), 2)
                assert all(np.isfinite(d[k]).all() for k in d.files)
                error = pred - truth
                np.testing.assert_allclose(np.mean(error**2), float(row["MSE"]), rtol=1e-12)
                np.testing.assert_allclose(np.sqrt(np.mean(error**2)), float(row["RMSE"]), rtol=1e-12)
                np.testing.assert_allclose(np.mean(abs(error)), float(row["MAE"]), rtol=1e-12)
                ate, rte = compute_ate_rte(d["predicted_positions"], d["ground_truth_positions"])
                np.testing.assert_allclose(ate, float(row["ATE_official_m"]), rtol=1e-12)
                np.testing.assert_allclose(rte, float(row["RTE_60s_official_m"]), rtol=1e-12)
                squared += np.sum(error**2); absolute += np.sum(abs(error)); count += len(pred)
                sequence_checks.append(row["sequence"])
        saved = next(r for r in summary["test_metrics"] if r["split"] == split and r["model"] == "RoNIN ResNet18")
        np.testing.assert_allclose(squared/(count*2), saved["MSE"], rtol=1e-12)
        np.testing.assert_allclose(absolute/(count*2), saved["MAE"], rtol=1e-12)
        selected_rows = [r for r in rows if r["split"] == split]
        for metric in ("ATE_official_m", "RTE_60s_official_m"):
            np.testing.assert_allclose(np.mean([float(r[metric]) for r in selected_rows]), saved[metric], rtol=1e-12)
        assert count == saved["windows"]
        verification[split] = {"sequences_checked": len(sequence_checks), "windows": count,
                               "saved_prediction_metrics_match": True}
    for source in ("model_resnet1d.py", "metric.py", "data_glob_speed.py", "ronin_resnet.py"):
        p = root / "ronin-master" / "source" / source
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        assert actual == config["source_sha256"][str(p.relative_to(root))]
    verification["official_sources_unchanged"] = True
    shutil.copy2(__file__, out / "source_snapshot" / Path(__file__).name)
    verification["postprocessing_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "result_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    selected = [r for r in summary["test_metrics"] if r["model"] == "RoNIN ResNet18"]
    table = "\n".join(f"| {r['split']} | {r['windows']:,} | {r['MSE']:.6f} | {r['RMSE']:.6f} | {r['MAE']:.6f} | {r['ATE_official_m']:.3f} | {r['RTE_60s_official_m']:.3f} |" for r in selected)
    readme = f"""# RoNIN ResNet18 基线实验结果

本次使用交付的 NPZ 数据，调用 `ronin-master/source/model_resnet1d.py` 中未修改的官方 ResNet18 网络训练，并在全部 35 条 seen_test、32 条 unseen_test 序列评估。数据预处理正确性作为本次实验前提。

## 训练结果

- 完成轮数：{summary['epochs_completed']} / {config['max_epochs']}；最佳轮数：{summary['best_epoch']}。
- 最佳验证 MSE：{summary['best_val_mse']:.8f}。
- 训练轮次累计耗时：{summary['training_seconds']/60:.2f} 分钟；训练与最终评估总耗时：{summary['total_seconds']/60:.2f} 分钟（不含最初数据加载）。
- CUDA GPU：{config['gpu']}；Python：{config['python_executable']}；PyTorch：{config['torch']}。
- 训练 72 条、验证 13 条；窗口 200，步长 10；训练 batch {config['batch_size']}；Adam 1e-4；MSE；随机种子 42。
- 每轮完整遍历训练窗口；训练速度异常筛选 >10 m/s；验证和测试不筛选。
- 水平随机旋转、起点随机偏移；float32；TF32 关闭。
- 验证集连续 {config['early_stopping_patience']} 轮无改善时早停，测试集未用于选模。

## 测试结果

| 测试集 | 窗口数 | MSE | RMSE (m/s) | MAE (m/s) | 官方 ATE (m) | 官方 60s RTE (m) |
|---|---:|---:|---:|---:|---:|---:|
{table}

速度 MSE/RMSE/MAE 对所有窗口和两个分量加权计算。ATE/RTE 先按序列计算，再对序列等权平均。`sequence_metrics.csv` 另存欧氏距离 ATE 和终点误差。

## 重要协议说明

1. 标签是整个窗口的平均速度 `(p[start+200]-p[start])/(t[start+200]-t[start])`，对应官方 ResNet 的 StridedSequenceDataset。旧 CNN/CNN-GRU 实验使用 `targets[end]` 单步速度，因此不能直接横向比较旧 CSV 数字；公平比较须重训并使用相同标签协议。
2. 使用交付的 72/13/35/32 划分，并非源码自带名单；所有 NPZ 姿态来源为 game_rv；训练增加了项目约定的 >10 m/s 异常筛选。这是官方网络在当前项目数据协议上的基线，不能直接宣称达到论文原条件复现。
3. 网络和官方 metric.py 未修改。NPZ 适配、GPU 常驻窗口批处理、边界安全的随机偏移及实验管理由独立入口实现。导出 NPZ 缺失原始末帧位置，序列末端窗口数与直接 HDF5 路径可能有极小差异。
4. 轨迹按官方 `recon_traj_with_preds` 公式重建：预测速度乘平均输出间隔后累计，以真实首位置初始化，端点延拓后线性插值到原始时间。未做额外平移、旋转或 SE(2) 最优拟合。此公式已与本地官方函数逐元素核对。
5. 官方 ATE/RTE 使用每个坐标的平方误差均值开方；二维欧氏 RMSE 是其 sqrt(2) 倍。RTE 用 12,000 帧，即约 60 秒；不足 60 秒沿用官方缩放。不能混用不同定义。
6. 每个预测代表约 1 秒平均速度，间隔约 0.05 秒。重建沿用官方约定，不是把重叠窗口各积分一整秒。
7. cuDNN benchmark 开启，未强制逐位确定性；种子、数据指纹、代码快照和完整配置已保存。

## 文件入口

- `index.html`：本地可视化总览，可切换全部测试序列。
- `training_curves.png`：训练和验证损失、学习率。
- `test_summary.png`、`ate_distribution.png`：测试汇总和轨迹误差分布。
- `seen_test/*.png`、`unseen_test/*.png`：每条序列的轨迹、速度、误差图。
- `seen_test/*_predictions.npz`、`unseen_test/*_predictions.npz`：窗口索引、时刻、预测/真实速度及完整轨迹。
- `test_metrics.csv`、`sequence_metrics.csv`：汇总和逐序列指标。
- `checkpoint_best.pt`、`checkpoint_last.pt`：最佳与最后一轮权重、优化器及配置。
- `config.json`、`training_log.csv`、`run.log`、`summary.json`：配置与完整运行记录。
- `source_snapshot/`、`data_fingerprints.csv`：代码、名单快照和数据 SHA256。
- `acceptance_checks.json`、`result_verification.json`：加载、重建和结果复核证据。

## 重跑命令

在项目根目录运行，务必使用一个新的输出目录：

```powershell
& 'E:\\conda_envs\\torch_env\\python.exe' -B -u scripts/run_ronin_npz_baseline.py --out results/ronin_resnet18_npz_new_run --epochs 30 --patience 6 --batch-size 256
```

原始交付数据、旧实验结果与官方源码均保留。
"""
    (out / "README_实验结果.md").write_text(readme, encoding="utf-8")
    html_rows = "".join(f"<tr><td>{html.escape(r['split'])}</td><td>{r['windows']:,}</td><td>{r['RMSE']:.4f}</td><td>{r['MAE']:.4f}</td><td>{r['ATE_official_m']:.3f}</td><td>{r['RTE_60s_official_m']:.3f}</td></tr>" for r in selected)
    gallery_data = [{"split": r["split"], "sequence": r["sequence"],
                     "ate": float(r["ATE_official_m"]), "rte": float(r["RTE_60s_official_m"])} for r in rows]
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RoNIN ResNet18 实验结果</title><style>
body{font:16px/1.65 system-ui,sans-serif;background:#f4f6f8;color:#152331;max-width:1180px;margin:32px auto;padding:0 22px}
h1,h2{line-height:1.3}section{background:white;padding:24px;margin:20px 0;border-radius:10px}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:10px;text-align:right;border-bottom:1px solid #dde3e8}td:first-child,th:first-child{text-align:left}select,button{font:inherit;padding:7px 12px;margin:4px}a{color:#1b63a0}.muted{color:#516271} .figures{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:700px){.figures{display:block}section{padding:12px}table{font-size:12px}}
</style><h1>RoNIN ResNet18 实验结果</h1><p class="muted">官方网络 · 交付 NPZ 数据 · 窗口平均水平速度 · 独立测试集评估</p>
<section><h2>训练与测试</h2><p>__TRAINING__</p><table><tr><th>测试集</th><th>窗口数</th><th>速度 RMSE</th><th>速度 MAE</th><th>官方 ATE</th><th>官方 RTE</th></tr>__ROWS__</table><p class="muted">速度单位 m/s；轨迹单位 m。ATE/RTE 为官方坐标平均定义、按序列等权汇总。采用项目划分和 game_rv 特征；与旧单步速度标签实验不可直接比较。</p><p><a href="README_实验结果.md">详细实验说明</a> · <a href="config.json">配置</a> · <a href="test_metrics.csv">汇总 CSV</a> · <a href="sequence_metrics.csv">逐序列 CSV</a> · <a href="run.log">日志</a></p></section>
<section><h2>训练曲线</h2><img src="training_curves.png" alt="训练验证损失与学习率"></section>
<section><h2>测试概览</h2><div class="figures"><img src="test_summary.png" alt="测试指标"><img src="ate_distribution.png" alt="ATE 分布"></div></section>
<section><h2>全部测试轨迹</h2><label>测试集 <select id="split"><option>seen_test</option><option>unseen_test</option></select></label><label>序列 <select id="sequence"></select></label><button id="prev">上一条</button><button id="next">下一条</button><p id="caption"></p><img id="trajectory" alt="轨迹、预测速度和位置误差"><p><a id="prediction">下载该序列预测 NPZ</a></p></section>
<script>const data=__DATA__;const s=document.getElementById('split'),q=document.getElementById('sequence');function render(){const r=data.find(r=>r.split===s.value&&r.sequence===q.value);document.getElementById('trajectory').src=r.split+'/'+r.sequence+'.png';document.getElementById('caption').textContent=r.split+'/'+r.sequence+' — ATE '+r.ate.toFixed(3)+' m，RTE '+r.rte.toFixed(3)+' m';document.getElementById('prediction').href=r.split+'/'+r.sequence+'_predictions.npz';}function update(){q.replaceChildren(...data.filter(r=>r.split===s.value).map(r=>{const o=document.createElement('option');o.value=o.textContent=r.sequence;return o;}));render();}s.onchange=update;q.onchange=render;for(const [id,d] of [['prev',-1],['next',1]])document.getElementById(id).onclick=()=>{q.selectedIndex=(q.selectedIndex+d+q.length)%q.length;render();};update();</script></html>"""
    page = page.replace("__ROWS__", html_rows).replace("__DATA__", json.dumps(gallery_data))
    page = page.replace("__TRAINING__", f"完成 {summary['epochs_completed']} 轮，最佳第 {summary['best_epoch']} 轮；验证 MSE {summary['best_val_mse']:.6f}。全部 67 条测试序列的预测文件已复核。")
    (out / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"verification": verification, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
