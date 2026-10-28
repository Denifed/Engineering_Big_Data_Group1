"""Create an honest paired analysis and gallery for the completed C4 experiment."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def read_csv(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(); out = args.run.resolve()
    assert read_json(out / "final_audit.json")["status"] == "passed"
    summary = read_json(out / "summary.json"); metrics = summary["metrics"]
    source = read_csv(out / "comparison/sequence_metrics.csv")
    paired = read_json(out / "comparison/paired_sequence_analysis.json")
    benchmark = read_json(out / "inference_benchmark.json")
    conditions = ("original_single", "original_c4", "mixed_single", "mixed_c4", "consecutive_single", "consecutive_c4")
    labels = {"original_single": "原始 BN / 单次", "original_c4": "原始 BN / C4",
              "mixed_single": "混合校准 BN / 单次", "mixed_c4": "混合校准 BN / C4",
              "consecutive_single": "片段校准 BN / 单次", "consecutive_c4": "片段校准 BN / C4"}
    lookup = {(r["split"], r["variant"]): r for r in metrics}
    deltas = []; examples = []
    keys = ("RMSE", "ATE_official_m", "RTE_60s_official_m", "endpoint_error_m", "mean_velocity_bias_norm_mps")
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for j, split in enumerate(("seen_test", "unseen_test")):
        base = {r["sequence"]: r for r in source if r["split"] == split and r["variant"] == "original_single"}
        improved = {r["sequence"]: r for r in source if r["split"] == split and r["variant"] == "original_c4"}
        for stem, a in base.items():
            b = improved[stem]
            deltas.append(dict(split=split, sequence=stem, ATE_single=float(a["ATE_official_m"]),
                ATE_c4=float(b["ATE_official_m"]), ATE_reduction_m=float(a["ATE_official_m"])-float(b["ATE_official_m"]),
                endpoint_single=float(a["endpoint_error_m"]), endpoint_c4=float(b["endpoint_error_m"])))
        selection = sorted([r for r in deltas if r["split"] == split], key=lambda r: r["ATE_reduction_m"])
        for label, r in (("largest_ATE_deterioration", selection[0]), ("largest_ATE_reduction", selection[-1])):
            examples.append(dict(example_type=label, **r))
        for i, key in enumerate(("ATE_official_m", "endpoint_error_m")):
            ax = axes[i, j]
            a = np.array([float(base[n][key]) for n in base]); b = np.array([float(improved[n][key]) for n in base])
            limit = max(a.max(), b.max())*1.06
            ax.scatter(a, b, c=np.where(b<a, "#218a68", "#cb633b"), alpha=.8, s=32)
            ax.plot([0, limit], [0, limit], "--", c="gray", lw=1)
            ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="Original single [m]", ylabel="Original C4 [m]",
                   title=f"{split}: {'ATE' if i == 0 else 'Endpoint'} | improved {int((b<a).sum())}/{len(a)}")
            ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.2)
    fig.suptitle("All test sequences | below diagonal = lower error | no sequences removed")
    fig.tight_layout(); fig.savefig(out / "comparison/paired_sequence_scatter.png", dpi=160); plt.close(fig)
    (out / "comparison/primary_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")
    with (out / "comparison/primary_sequence_deltas.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(deltas[0])); w.writeheader(); w.writerows(deltas)
    primary_table = []; effects = []
    for split in ("val", "seen_test", "unseen_test"):
        a, b = lookup[split, "original_single"], lookup[split, "original_c4"]
        for r in (a, b):
            primary_table.append(f"| {split} | {labels[r['variant']]} | {r['RMSE']:.6f} | {r['ATE_official_m']:.4f} | {r['RTE_60s_official_m']:.4f} | {r['endpoint_error_m']:.4f} | {r['mean_velocity_bias_norm_mps']:.6f} |")
        change = {k: 100*(a[k]-b[k])/a[k] for k in keys}
        effects.append(f"| {split} | {change['RMSE']:+.2f}% | {change['ATE_official_m']:+.2f}% | {change['RTE_60s_official_m']:+.2f}% | {change['endpoint_error_m']:+.2f}% | {change['mean_velocity_bias_norm_mps']:+.2f}% |")
    factorial = []
    for split in ("seen_test", "unseen_test"):
        for variant in conditions:
            r = lookup[split, variant]
            factorial.append(f"| {split} | {labels[variant]} | {r['RMSE']:.6f} | {r['ATE_official_m']:.4f} | {r['RTE_60s_official_m']:.4f} | {r['endpoint_error_m']:.4f} | {r['mean_velocity_bias_norm_mps']:.6f} |")
    distributions = []
    for r in paired:
        if r["bn_state"] != "original":
            continue
        for key in ("ATE_official_m", "endpoint_error_m", "mean_velocity_bias_norm_mps"):
            m = r["metrics"][key]
            distributions.append(f"| {r['split']} | {key} | {m['improved']}/{r['sequences']} | {m['median_single']:.4f} → {m['median_c4']:.4f} | {m['p90_single']:.4f} → {m['p90_c4']:.4f} |")
    example_table = "\n".join(f"| {r['split']} | {r['sequence']} | {r['example_type']} | {r['ATE_single']:.4f} → {r['ATE_c4']:.4f} |" for r in examples)
    criterion_table = "\n".join(f"| {k} | {'满足' if v else '未满足'} |" for k, v in summary["primary_criteria"].items())
    verdict = ("主要对照满足预先写明的整体改善条件，但仅限本次复用测试集的探索性结果，不能宣称普遍有效或统计显著。"
               if summary["primary_overall_improvement"] else
               "主要对照未满足预先写明的整体改善条件。应分别报告具体指标收益和退化，不能宣称整体性能提升。")
    report = f"""# CNN-GRU 水平旋转一致性集成实验

## 结论

{verdict}

主要比较在相同旧 CNN-GRU 第 8 轮 checkpoint、原始 BN 状态下，仅改变推理方法；BN 校准属于另行列出的交叉消融，不把两种变化的收益混在一起。所有六种条件、80 条完整序列都已保存，无筛选轨迹。

## 方法与执行计划

先固定 `PLAN_执行计划.md` 和 `protocol.json`，验证旋转/逆旋转、C4 群复合、Z 保持及集成等变性，再完整评估 val、seen_test、unseen_test。没有按照验证或测试结果调整角度、权重、BN 状态或 checkpoint。

对输入 x，同步旋转角速度、加速度的 XY 分量，Z 和时间不变；用同一个冻结模型 f 计算四次输出，逆旋转到原坐标后等权平均：

`v_C4(x) = (1/4) * sum_k R(-k*90°) f(R(k*90°) x), k=0,1,2,3`。

四个网络前向为 FP32，逆旋转用精确符号与分量交换，均值以 float64 计算。输入窗口仍 200 帧、步长 10 帧，输出仍是该窗口平均全局 XY 速度。各窗口独立，不增加历史状态、传感器、标签输入或在线参数更新。

BN 三种状态直接使用之前保存的旧模型权重（原始/训练集混合校准/训练集片段校准）；六组全部学习参数完全一致，均 eval 推理，BN 不再更新。原始测试预测逐元素复现了之前 67 条测试轨迹。

旋转等变性是惯性里程计已有研究方向，参考 RIO：https://arxiv.org/abs/2111.11676 。本实验是固定 C4 推理群平均，不是完整 RIO 复现，也没有实施旋转一致性训练损失或测试时训练，不能将群平均称为首创算法。

## 主要结果：原始 BN，单次 vs C4

速度 RMSE/偏置单位 m/s，ATE/RTE/终点误差单位 m。轨迹指标为序列等权平均，速度指标按窗口及 XY 分量计算。与旧实验共用官方式轨迹重建，不进行旋转拟合；仅以首个真值位置设置坐标原点。

| 数据集 | 方法 | RMSE | ATE | 60s RTE | 终点误差 | 速度偏置 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(primary_table)}

正数表示 C4 相比相同 BN 的单次预测误差下降：

| 数据集 | RMSE 降低 | ATE 降低 | RTE 降低 | 终点误差降低 | 速度偏置降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(effects)}

## 预先固定的整体改善条件

| 条件 | 判断 |
| --- | --- |
{criterion_table}

## BN × 推理方式完整对照

| 数据集 | 条件 | RMSE | ATE | 60s RTE | 终点误差 | 速度偏置 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(factorial)}

BN 自身的变化看同推理方法下不同 BN 状态；C4 自身作用看同 BN 下单次和 C4。不得只从测试集选最优组合并把它当成事先确定的方法。

本次三种 BN 状态下，C4 均降低了两个测试集的平均 ATE。原始 BN 的 C4 与混合校准 BN 的 C4 相比，seen ATE 为 5.2162 vs 5.2068 m，unseen ATE 为 7.1837 vs 7.1779 m，差异均不足 0.2%。因此本次主要收益来自 C4 集成，BN 校准没有显示出明显的额外 ATE 收益；保留事先指定的“原始 BN＋C4”作为主要候选即可。

## 逐序列分布与失败例子

| 数据集 | 指标 | 改善序列数 | 中位数：单次 → C4 | P90：单次 → C4 |
| --- | --- | --- | --- | --- |
{chr(10).join(distributions)}

下面按绝对 ATE 变化选出每个测试集改善最多、退化最多的轨迹，用于解释，不代替完整统计：

| 数据集 | 序列 | 例子 | ATE：单次 → C4 |
| --- | --- | --- | --- |
{example_table}

全部图见 `index.html`，散点图显示所有测试序列，不剔除异常值。

## 计算开销

同一 GPU、相同验证输入、batch=256，10 次预热后各同步测量 30 次：单次预测中位 {benchmark['single']['median_batch_ms']:.4f} ms/batch，C4 中位 {benchmark['c4']['median_batch_ms']:.4f} ms/batch，耗时比 {benchmark['c4_to_single_median_ratio']:.2f} 倍。这是 GPU 内前向及聚合计时，不含输出传回 CPU、磁盘、画图及轨迹重建；正式保存预测的均值在 CPU 用 float64 计算，而计时采用 GPU float64 均值。它不能直接当成端到端部署时延。

网络仍 78,370 个参数，C4 每窗口四次前向。若有效，收益应称为增加推理计算获得的物理一致性集成收益，不能与单次预测方法忽略成本地比较。

## 解释边界

- C4 集成按构造满足四个离散角度的等变性；检查通过并不自动证明定位精度改善，也不保证任意角度的精确等变性。
- 如果速度残差和轨迹误差下降，可以支持当前模型的方向敏感误差具有可利用空间；不能仅凭集成收益确认唯一根因。
- 该方法不增加绝对位置观测或跨窗口状态，不保证长期误差有界；低频残差若未消除，仍可累计漂移。
- 单个 seed=42 训练的 checkpoint，旧测试集多次参与项目分析，结果属于探索性证据；不宣称独立盲测或统计显著性。
- 所有均值下降不代表全部轨迹改善；例如 unseen 的 RTE P90 从 9.4493 m 升至 9.6944 m，仍存在尾部退化。
- 本轮没有一致性损失训练、模型重训、超参数搜索、测试集 BN 更新或真值校正。

## 产物与核验

- `final_audit.json`、`result_verification.json`：全量保存预测重读、四方向均值复算、轨迹重建及指标复算，模型和原文件保持不变。
- `comparison/test_metrics.csv`、`all_metrics.csv`、`sequence_metrics.csv`：完整指标。
- `comparison/c4_effect.csv`、`paired_sequence_analysis.json`：相同 BN 下的集成收益和逐序列统计。
- `comparison/rotation_consistency.csv`：原模型对水平旋转的预测不一致程度，属机制诊断，不是定位精度。
- 六条件目录：每条序列完整预测 NPZ；C4 目录另保存四个逆旋转后的分支输出。
- `comparison/test_summary.png`、`validation_summary.png`、`paired_sequence_scatter.png` 和 80 张逐序列对比图。
- `source_snapshot/`、数据指纹、checkpoint 来源及参数哈希、环境、日志、计划、协议和计时记录。

复跑命令：`E:\\conda_envs\\torch_env\\python.exe -B cnn_baseline/evaluate_rotation_consistency.py --out results/一个不存在的新目录`。分析命令：`E:\\conda_envs\\torch_env\\python.exe -B cnn_baseline/analyze_rotation_consistency.py --run results/该实验目录`。
"""
    (out / "README_实验结果.md").write_text(report, encoding="utf-8")
    options = [dict(split=r["split"], sequence=r["sequence"], ate_single=r["ATE_single"], ate_c4=r["ATE_c4"]) for r in deltas]
    table_html = "".join(f"<tr><td>{r['split']}</td><td>{labels[r['variant']]}</td><td>{r['RMSE']:.5f}</td><td>{r['ATE_official_m']:.3f}</td><td>{r['RTE_60s_official_m']:.3f}</td><td>{r['endpoint_error_m']:.3f}</td><td>{r['mean_velocity_bias_norm_mps']:.5f}</td></tr>" for r in metrics if r["split"] != "val")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CNN-GRU 水平旋转一致性</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1500px;margin:28px auto;padding:0 20px;color:#1a3041;background:#f4f6f8}section{background:white;padding:22px;margin:20px 0;border-radius:10px}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,td:nth-child(2){text-align:left}select,button{font:inherit;padding:7px;margin:5px}a{color:#16689f}</style>
<h1>CNN-GRU：水平旋转一致性集成</h1><p>__VERDICT__</p><p>同一第 8 轮权重，原始/两种校准 BN × 单次/C4。固定四方向等权平均，无重训或测试适应。</p>
<section><h2>完整测试结果</h2><table><tr><th>数据集</th><th>条件</th><th>RMSE</th><th>ATE</th><th>RTE</th><th>终点误差</th><th>速度偏置</th></tr>__TABLE__</table><p><a href="README_实验结果.md">实验报告</a> · <a href="PLAN_执行计划.md">固定计划</a> · <a href="comparison/test_metrics.csv">指标 CSV</a> · <a href="final_audit.json">核验</a></p></section>
<section><h2>指标与全部序列分布</h2><img src="comparison/test_summary.png"><img src="comparison/paired_sequence_scatter.png"></section>
<section><h2>逐序列对比，包含成功和失败</h2><select id="split"><option>seen_test</option><option>unseen_test</option></select><select id="seq"></select><button id="prev">上一条</button><button id="next">下一条</button><p id="caption"></p><img id="chart"><p><a id="baseline">原始预测 NPZ</a> · <a id="ensemble">C4 预测 NPZ</a> · <a id="branches">四方向输出</a></p></section>
<script>const data=__DATA__,s=document.getElementById('split'),q=document.getElementById('seq');function render(){const r=data.find(r=>r.split===s.value&&r.sequence===q.value);document.getElementById('chart').src='comparison/'+s.value+'/'+q.value+'.png';document.getElementById('caption').textContent='原始 BN：单次 ATE '+r.ate_single.toFixed(3)+' m → C4 '+r.ate_c4.toFixed(3)+' m';document.getElementById('baseline').href='original_single/'+s.value+'/'+q.value+'_predictions.npz';document.getElementById('ensemble').href='original_c4/'+s.value+'/'+q.value+'_predictions.npz';document.getElementById('branches').href='original_c4/'+s.value+'/'+q.value+'_aligned_branches.npz';}function update(){q.replaceChildren(...data.filter(r=>r.split===s.value).map(r=>{const o=document.createElement('option');o.value=o.textContent=r.sequence;return o;}));render();}s.onchange=update;q.onchange=render;for(const [id,d] of [['prev',-1],['next',1]])document.getElementById(id).onclick=()=>{q.selectedIndex=(q.selectedIndex+d+q.length)%q.length;render();};update();</script></html>"""
    (out / "index.html").write_text(page.replace("__VERDICT__", verdict).replace("__TABLE__", table_html).replace("__DATA__", json.dumps(options)), encoding="utf-8")
    shutil.copy2(__file__, out / "source_snapshot" / Path(__file__).name)
    print(json.dumps(dict(status="analysis_completed", overall_improvement=summary["primary_overall_improvement"],
                         primary_test_metrics=[r for r in metrics if r["split"] != "val" and r["variant"].startswith("original")],
                         benchmark=benchmark, primary_examples=examples), indent=2))


if __name__ == "__main__":
    main()
