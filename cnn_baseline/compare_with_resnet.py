"""Evaluate a mean-velocity CNN run and compare saved ResNet trajectories."""
import argparse
import csv
import hashlib
import html
import json
import logging
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn_baseline.common import MODELS, select_device
from cnn_baseline.dataset import LABEL_PROTOCOL, validate_checkpoint
from cnn_baseline.evaluate import velocity_metrics
from scripts.run_ronin_npz_baseline import load_sequence, reconstruct, compute_ate_rte
import matplotlib.pyplot as plt


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def trajectory_metrics(pred, target, positions, gt):
    result = velocity_metrics(pred, target)
    result["windows"] = result.pop("samples")
    ate, rte = compute_ate_rte(positions, gt, pred_per_min=12000)
    result.update(ATE_official_m=float(ate), RTE_60s_official_m=float(rte),
                  ATE_euclidean_m=float(np.sqrt(np.mean(np.sum((positions - gt) ** 2, axis=1)))),
                  endpoint_error_m=float(np.linalg.norm(positions[-1] - gt[-1])))
    return result


def predict_windows(model, features, starts, device, batch_size):
    ft = torch.from_numpy(features).to(device)
    offsets = torch.arange(200, device=device)
    chunks = []
    with torch.inference_mode():
        for begin in range(0, len(starts), batch_size):
            ids = torch.from_numpy(starts[begin:begin + batch_size]).to(device)
            x = ft[ids[:, None] + offsets].transpose(1, 2).contiguous()
            chunks.append(model(x).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def make_gallery(out, rows, summary_rows, training):
    options = [{"split": r["split"], "sequence": r["sequence"],
                "ate": r["ATE_official_m"], "rte": r["RTE_60s_official_m"]}
               for r in rows if r["model"] == "CNN"]
    table = "".join(
        f"<tr><td>{html.escape(r['split'])}</td><td>{html.escape(r['model'])}</td>"
        f"<td>{r['windows']:,}</td><td>{r['RMSE']:.4f}</td><td>{r['MAE']:.4f}</td>"
        f"<td>{r['ATE_official_m']:.3f}</td><td>{r['RTE_60s_official_m']:.3f}</td></tr>"
        for r in summary_rows if r["model"] != "Zero velocity")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CNN 与 RoNIN ResNet18 实验对比</title>
<style>body{font:16px/1.65 system-ui,sans-serif;background:#f4f6f8;color:#152331;max-width:1200px;margin:30px auto;padding:0 20px}
section{background:white;padding:22px;margin:20px 0;border-radius:10px}img{width:100%;height:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:9px;text-align:right;border-bottom:1px solid #dde3e8}
td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){text-align:left}
select,button{font:inherit;padding:7px;margin:5px}a{color:#1b63a0}.muted{color:#516271}
@media(max-width:700px){section{padding:10px}table{font-size:11px}}</style>
<h1>CNN 与 RoNIN ResNet18 实验对比</h1>
<p class="muted">同一窗口平均速度标签、同一测试序列、同一轨迹重建与指标定义。</p>
<section><h2>结果</h2><p>__TRAINING__</p><table><tr><th>测试集</th><th>模型</th><th>窗口数</th><th>速度 RMSE</th><th>速度 MAE</th><th>官方 ATE</th><th>官方 RTE</th></tr>__TABLE__</table>
<p class="muted">速度单位 m/s，轨迹单位 m。CNN 无随机增强，ResNet 有随机旋转及时间偏移；训练轮次和网络容量也不同，此表不是严格控制变量的结构消融。</p>
<p><a href="README_实验结果.md">实验说明</a> · <a href="test_metrics.csv">汇总指标</a> · <a href="comparison_sequence_metrics.csv">逐序列对比</a> · <a href="training_log.csv">训练日志</a> · <a href="result_verification.json">复核记录</a></p></section>
<section><h2>训练曲线</h2><img src="training_curves.png" alt="CNN 训练与验证损失"></section>
<section><h2>汇总对比</h2><img src="cnn_vs_resnet_summary.png" alt="CNN与ResNet指标对比"><img src="ate_distribution.png" alt="轨迹误差分布"></section>
<section><h2>逐序列轨迹、速度与误差</h2>
<select id="split"><option>seen_test</option><option>unseen_test</option></select>
<select id="seq"></select><button id="prev">上一条</button><button id="next">下一条</button>
<p id="caption"></p><img id="image" alt="CNN、ResNet和参考轨迹对比"><p><a id="prediction">CNN 预测数据 NPZ</a></p></section>
<script>const data=__DATA__,s=document.getElementById('split'),q=document.getElementById('seq');
function render(){const r=data.find(r=>r.split===s.value&&r.sequence===q.value);
document.getElementById('image').src=r.split+'/'+r.sequence+'.png';
document.getElementById('prediction').href=r.split+'/'+r.sequence+'_predictions.npz';
document.getElementById('caption').textContent=r.split+'/'+r.sequence+'：CNN ATE '+r.ate.toFixed(3)+' m，RTE '+r.rte.toFixed(3)+' m';}
function update(){q.replaceChildren(...data.filter(r=>r.split===s.value).map(r=>{const o=document.createElement('option');o.value=o.textContent=r.sequence;return o;}));render();}
s.onchange=update;q.onchange=render;for(const [id,d] of [['prev',-1],['next',1]])document.getElementById(id).onclick=()=>{q.selectedIndex=(q.selectedIndex+d+q.length)%q.length;render();};update();</script></html>"""
    page = page.replace("__TABLE__", table).replace("__DATA__", json.dumps(options))
    page = page.replace("__TRAINING__", f"CNN 完成 {training['epochs_completed']} 轮；验证集选出的最佳轮次为 {training['best_epoch']}，验证 MSE {training['best_val_mse']:.6f}。")
    (out / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-run", type=Path, required=True)
    parser.add_argument("--resnet-run", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    out = args.cnn_run.resolve(); reference = args.resnet_run.resolve()
    if out == reference:
        raise ValueError("CNN and ResNet run directories must differ")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if any((out / name).exists() for name in ("seen_test", "unseen_test", "evaluation_summary.json")):
        raise FileExistsError("Evaluation outputs already exist; refusing overwrite")
    training = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    ref_summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    if training["status"] != "completed" or training["epochs_completed"] != 30:
        raise ValueError("This requested experiment must complete all 30 CNN epochs")
    checkpoint = torch.load(out / "checkpoint_best.pt", map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint, "cnn", 200, 10)
    if checkpoint["epoch"] != training["best_epoch"]:
        raise ValueError("Checkpoint does not match selected best epoch")
    device = select_device(args.device); torch.set_num_threads(4)
    model = MODELS["cnn"]().to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True); model.eval()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "evaluation.log", encoding="utf-8")])
    shutil.copy2(__file__, out / "source_snapshot" / Path(__file__).name)
    provenance = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "cnn_checkpoint_sha256": hashlib.sha256((out / "checkpoint_best.pt").read_bytes()).hexdigest(),
                  "resnet_run": str(reference), "label_protocol": LABEL_PROTOCOL,
                  "trajectory_function": "scripts/run_ronin_npz_baseline.py:reconstruct",
                  "metric_function": "ronin-master/source/metric.py:compute_ate_rte",
                  "test_filter_outliers": False}
    for p in (ROOT / "scripts/run_ronin_npz_baseline.py", ROOT / "ronin-master/source/metric.py"):
        provenance[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        shutil.copy2(p, out / "source_snapshot" / p.name)
    (out / "evaluation_config.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    rows = []; summary_rows = []; verification = {}
    for split in ("seen_test", "unseen_test"):
        names = [s.strip() for s in (Path(config["split_root"]) / f"{split}.txt").read_text().splitlines() if s.strip()]
        folder = out / split; folder.mkdir()
        totals = {name: [0., 0., 0] for name in ("CNN", "RoNIN ResNet18", "Zero velocity")}
        for no, name in enumerate(names, 1):
            f, p, t, _, y = load_sequence(Path(config["data_root"]) / name)
            starts = np.arange(0, len(y), 10, dtype=np.int64)
            truth = y[starts].astype(np.float64); gt = p[:, :2]
            pred = predict_windows(model, f, starts, device, args.batch_size)
            trajectory = reconstruct(t, p, starts, pred)
            stem = Path(name).stem
            with np.load(reference / split / f"{stem}_predictions.npz", allow_pickle=False) as ref:
                np.testing.assert_array_equal(ref["starts"], starts)
                np.testing.assert_array_equal(ref["timestamps"], t)
                np.testing.assert_array_equal(ref["target_velocity"], truth)
                np.testing.assert_array_equal(ref["ground_truth_positions"], gt)
                res_pred = ref["predicted_velocity"].copy()
                res_traj = ref["predicted_positions"].copy()
            for model_name, velocity, positions in (("CNN", pred, trajectory), ("RoNIN ResNet18", res_pred, res_traj)):
                row = dict(split=split, sequence=stem, model=model_name,
                           duration_s=float(t[-1] - t[0]), **trajectory_metrics(velocity, truth, positions, gt))
                rows.append(row)
                total = totals[model_name]; error = velocity - truth
                total[0] += float(np.sum(error ** 2)); total[1] += float(np.sum(abs(error))); total[2] += len(truth)
            total = totals["Zero velocity"]
            total[0] += float(np.sum(truth ** 2)); total[1] += float(np.sum(abs(truth))); total[2] += len(truth)
            np.savez_compressed(folder / f"{stem}_predictions.npz",
                                starts=starts, prediction_timestamps=t[starts],
                                window_end_timestamps=t[starts + 200], predicted_velocity=pred,
                                target_velocity=truth, timestamps=t,
                                predicted_positions=trajectory, ground_truth_positions=gt)
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            ax = axes[0, 0]
            ax.plot(gt[:, 0], gt[:, 1], label="Ground truth", color="black", lw=1.3)
            ax.plot(res_traj[:, 0], res_traj[:, 1], label="RoNIN ResNet18", color="#df8b32", lw=1)
            ax.plot(trajectory[:, 0], trajectory[:, 1], label="CNN (30 epochs)", color="#3478b8", lw=1.1)
            ax.scatter(*gt[0], s=25, c="green", label="Start")
            ax.set(xlabel="X [m]", ylabel="Y [m]", title=f"{split}/{stem}")
            ax.axis("equal"); ax.legend()
            for component, ax in ((0, axes[0, 1]), (1, axes[1, 0])):
                times = t[starts] - t[0]
                ax.plot(times, truth[:, component], color="black", label="Window-mean target", lw=.7, alpha=.7)
                ax.plot(times, res_pred[:, component], color="#df8b32", label="ResNet18", lw=.7, alpha=.8)
                ax.plot(times, pred[:, component], color="#3478b8", label="CNN", lw=.7, alpha=.8)
                ax.set(xlabel="Time [s]", ylabel=("Vx" if component == 0 else "Vy") + " [m/s]")
                ax.legend()
            ax = axes[1, 1]
            ax.plot(t - t[0], np.linalg.norm(res_traj - gt, axis=1), label="ResNet18", color="#df8b32")
            ax.plot(t - t[0], np.linalg.norm(trajectory - gt, axis=1), label="CNN", color="#3478b8")
            cnn_row = rows[-2]; res_row = rows[-1]
            ax.set(xlabel="Time [s]", ylabel="Euclidean position error [m]",
                   title=f"Official ATE: CNN {cnn_row['ATE_official_m']:.2f}m / ResNet {res_row['ATE_official_m']:.2f}m")
            ax.legend()
            for ax in axes.flat:
                ax.grid(alpha=.25)
            fig.tight_layout(); fig.savefig(folder / f"{stem}.png", dpi=140); plt.close(fig)
            logging.info("%s %d/%d %s CNN MSE=%.6f ATE=%.3f RTE=%.3f",
                         split, no, len(names), stem, cnn_row["MSE"], cnn_row["ATE_official_m"], cnn_row["RTE_60s_official_m"])
        for model_name, (sq, absolute, n) in totals.items():
            selected = [r for r in rows if r["split"] == split and r["model"] == model_name]
            row = dict(split=split, model=model_name, sequences=len(names), windows=n,
                       MSE=sq / (2 * n), RMSE=math.sqrt(sq / (2 * n)), MAE=absolute / (2 * n),
                       ATE_official_m=float(np.mean([r["ATE_official_m"] for r in selected])) if selected else None,
                       RTE_60s_official_m=float(np.mean([r["RTE_60s_official_m"] for r in selected])) if selected else None)
            summary_rows.append(row)
            if model_name == "RoNIN ResNet18":
                original = next(r for r in ref_summary["test_metrics"] if r["split"] == split and r["model"] == model_name)
                for key in ("windows", "MSE", "RMSE", "MAE", "ATE_official_m", "RTE_60s_official_m"):
                    np.testing.assert_allclose(row[key], original[key], rtol=1e-12)
        verification[split] = dict(sequences=len(names), windows=totals["CNN"][2],
                                   labels_indices_timestamps_ground_truth_identical=True,
                                   resnet_saved_metrics_recomputed_match=True)
    cnn_rows = [r for r in rows if r["model"] == "CNN"]
    write_csv(out / "sequence_metrics.csv", cnn_rows)
    write_csv(out / "comparison_sequence_metrics.csv", rows)
    write_csv(out / "test_metrics.csv", summary_rows)
    # Recompute CNN metrics from persisted files, independent of inference accumulators.
    for split in ("seen_test", "unseen_test"):
        sq = absolute = 0.; n = 0; ates = []; rtes = []
        for row in [r for r in cnn_rows if r["split"] == split]:
            with np.load(out / split / f"{row['sequence']}_predictions.npz", allow_pickle=False) as d:
                assert all(np.isfinite(d[key]).all() for key in d.files)
                metrics = trajectory_metrics(d["predicted_velocity"], d["target_velocity"],
                                             d["predicted_positions"], d["ground_truth_positions"])
                for key, value in metrics.items():
                    np.testing.assert_allclose(value, row[key], rtol=1e-12)
                error = d["predicted_velocity"] - d["target_velocity"]
                sq += np.sum(error ** 2); absolute += np.sum(abs(error)); n += len(error)
                ates.append(metrics["ATE_official_m"]); rtes.append(metrics["RTE_60s_official_m"])
        saved = next(r for r in summary_rows if r["split"] == split and r["model"] == "CNN")
        for key, value in dict(MSE=sq/(2*n), MAE=absolute/(2*n), ATE_official_m=np.mean(ates),
                               RTE_60s_official_m=np.mean(rtes)).items():
            np.testing.assert_allclose(value, saved[key], rtol=1e-12)
        verification[split]["cnn_saved_prediction_metrics_recomputed_match"] = True
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ("RMSE", "ATE_official_m", "RTE_60s_official_m")):
        x = np.arange(2)
        for i, model_name in enumerate(("CNN", "RoNIN ResNet18")):
            values = [next(r[metric] for r in summary_rows if r["model"] == model_name and r["split"] == split)
                      for split in ("seen_test", "unseen_test")]
            bars = ax.bar(x + (i - .5) * .35, values, .35, label=model_name)
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_xticks(x, ("seen_test", "unseen_test"))
        ax.set(title=metric, ylabel="m/s" if metric == "RMSE" else "m"); ax.grid(axis="y", alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out / "cnn_vs_resnet_summary.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for model_name in ("CNN", "RoNIN ResNet18"):
        for split in ("seen_test", "unseen_test"):
            values = sorted(r["ATE_official_m"] for r in rows if r["model"] == model_name and r["split"] == split)
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=f"{model_name} / {split}")
    ax.set(xlabel="Official ATE [m]", ylabel="Fraction of sequences")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / "ate_distribution.png", dpi=160); plt.close(fig)
    evaluation = dict(status="completed", model="CNN", label_protocol=LABEL_PROTOCOL,
                      epochs_completed=training["epochs_completed"], best_epoch=training["best_epoch"],
                      best_val_mse=training["best_val_mse"], test_metrics=summary_rows)
    with (out / "training_log.csv").open(encoding="utf-8") as handle:
        epoch_records = list(csv.DictReader(handle))
    evaluation["training_wall_seconds"] = sum(float(r["seconds"]) for r in epoch_records)
    evaluation["epochs_with_elapsed_over_300s"] = [int(r["epoch"]) for r in epoch_records if float(r["seconds"]) > 300]
    (out / "evaluation_summary.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    (out / "result_verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    make_gallery(out, rows, summary_rows, training)
    table = "\n".join(f"| {r['split']} | {r['model']} | {r['RMSE']:.6f} | {r['MAE']:.6f} | {r['ATE_official_m']:.3f} | {r['RTE_60s_official_m']:.3f} |"
                      for r in summary_rows if r["model"] != "Zero velocity")
    report = f"""# CNN 30 轮训练与 ResNet 轨迹对比

CNN 完整训练 30 轮，关闭早停。使用验证集选出的第 {training['best_epoch']} 轮权重评估，验证 MSE 为 {training['best_val_mse']:.8f}；末轮权重另存 checkpoint_last.pt。

| 测试集 | 模型 | 速度 RMSE (m/s) | MAE (m/s) | 官方 ATE (m) | 官方 60s RTE (m) |
|---|---|---:|---:|---:|---:|
{table}

输入 [B,6,200]，标签为窗口两端 XY 位移除以真实时间差，步长 10。训练 72 条、验证 13 条、seen 35 条、unseen 32 条。训练窗口过滤极端速度区间，测试不筛选。

轨迹重建复用 ResNet 的 reconstruct 函数；官方 ATE/RTE 为坐标平方误差平均后开方，先计算每条序列再等权平均，不额外旋转对齐。速度指标按窗口及两个分量平均。

本次 CNN 参数 45,218，Adam lr=1e-4、batch=256、seed=42，无增强；ResNet 参数 4,634,882，包含水平旋转/时间偏移增强，实际训练 {ref_summary['epochs_completed']} 轮、最佳第 {ref_summary['best_epoch']} 轮。共同标签和评估已核对，但训练方案及网络容量不同，不是严格的结构消融。

训练日志中耗时超过 300 秒的轮次：{evaluation['epochs_with_elapsed_over_300s']}。运行期间出现长时间日志间隔，恢复后继续训练；记录的墙钟耗时包含停顿，不宜据此比较模型训练效率。

seen_test/ 和 unseen_test/ 每条序列保存四面板图：参考/CNN/ResNet 轨迹、vx、vy、位置误差，以及 CNN 预测 NPZ。index.html 提供全部序列浏览；cnn_vs_resnet_summary.png 和 ate_distribution.png 为汇总图。

test_metrics.csv 为汇总，sequence_metrics.csv 为 CNN 逐序列指标，comparison_sequence_metrics.csv 包含两模型逐序列指标。result_verification.json 记录全部预测落盘后的指标复算及与 ResNet 的标签、索引、时间、参考轨迹一致性检查。

训练命令：使用 torch_env 的 Python 运行 cnn_baseline/train.py --model cnn --epochs 30 --no-early-stopping --device cuda --batch-size 256 --out 本目录。
评估命令：使用同一 Python 运行 cnn_baseline/compare_with_resnet.py --cnn-run 本目录 --resnet-run 上级ResNet目录 --device cuda。
"""
    (out / "README_实验结果.md").write_text(report, encoding="utf-8")
    logging.info("COMPLETED %s", json.dumps(evaluation))


if __name__ == "__main__":
    main()

