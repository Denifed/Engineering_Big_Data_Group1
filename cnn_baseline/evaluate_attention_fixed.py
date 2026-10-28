"""Evaluate the fixed Attention experiment and compare frozen baseline predictions.

This never retrains or modifies either reference model. Every reference label,
timestamp, trajectory metric and data fingerprint is checked before comparison.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
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
from cnn_baseline.train_attention_fixed import load_reference
from cnn_baseline.run_fixed_comparison import dump, sha
from cnn_baseline.dataset import validate_checkpoint
from cnn_baseline.model_cnn_gru_attention import CNNGRUAttention
from cnn_baseline.compare_with_resnet import trajectory_metrics, write_csv
from scripts.run_ronin_npz_baseline import load_sequence, reconstruct
import matplotlib.pyplot as plt

MODEL = "cnn_gru_attention"
NAMES = {"resnet18": "RoNIN ResNet18", "cnn_gru": "CNN-GRU", MODEL: "CNN-GRU-Attention"}
COLORS = {"resnet18": "#dc8a26", "cnn_gru": "#3579b5", MODEL: "#a14ca0"}
SPLITS = ("seen_test", "unseen_test")
METRICS = ("MSE", "RMSE", "MAE", "ATE_official_m", "RTE_60s_official_m")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def predict_attention(model, features, starts, device, batch_size):
    ft = torch.from_numpy(features).to(device)
    offsets = torch.arange(200, device=device)
    predictions, weights = [], []
    with torch.inference_mode():
        for begin in range(0, len(starts), batch_size):
            ids = torch.from_numpy(starts[begin:begin + batch_size]).to(device)
            x = ft[ids[:, None] + offsets].transpose(1, 2).contiguous()
            pred, alpha = model.forward_with_attention(x)
            if begin == 0:
                torch.testing.assert_close(pred, model(x), rtol=0, atol=0)
            predictions.append(pred.cpu().numpy())
            weights.append(alpha.cpu().numpy())
    pred = np.concatenate(predictions).astype(np.float64)
    alpha = np.concatenate(weights)
    if alpha.shape != (len(starts), 50) or not np.isfinite(alpha).all() or (alpha < 0).any():
        raise ValueError("Invalid attention weights")
    np.testing.assert_allclose(alpha.sum(axis=1), 1, atol=2e-6, rtol=0)
    return pred, alpha


def plot_sequence(path, split, stem, t, starts, gt, truth, predictions, trajectories):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(gt[:, 0], gt[:, 1], color="black", lw=1.2, label="Ground truth")
    for name, traj in trajectories.items():
        ax.plot(traj[:, 0], traj[:, 1], color=COLORS[name], lw=1, label=NAMES[name])
    ax.scatter(*gt[0], c="green", s=20, label="Start")
    ax.set(title=f"{split}/{stem}", xlabel="X [m]", ylabel="Y [m]")
    ax.axis("equal"); ax.legend(fontsize=8)
    for c, ax in ((0, axes[0, 1]), (1, axes[1, 0])):
        ax.plot(t[starts]-t[0], truth[:, c], color="black", alpha=.65, lw=.7, label="Window-mean target")
        for name, pred in predictions.items():
            ax.plot(t[starts]-t[0], pred[:, c], color=COLORS[name], alpha=.8, lw=.7, label=NAMES[name])
        ax.set(xlabel="Time [s]", ylabel=("Vx" if c == 0 else "Vy") + " [m/s]")
        ax.legend(fontsize=8)
    for name, traj in trajectories.items():
        axes[1, 1].plot(t-t[0], np.linalg.norm(traj-gt, axis=1), color=COLORS[name], label=NAMES[name])
    axes[1, 1].set(xlabel="Time [s]", ylabel="Euclidean position error [m]")
    axes[1, 1].legend(fontsize=8)
    for ax in axes.flat:
        ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_attention(path, split, stem, t, starts, weights):
    selected = len(starts)//2
    subset = np.unique(np.linspace(0, len(starts)-1, min(250, len(starts)), dtype=int))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    image = axes[0].imshow(weights[subset], aspect="auto", origin="lower", interpolation="nearest",
                           extent=(-.5, 49.5, t[starts[subset[0]]]-t[0], t[starts[subset[-1]]]-t[0]), cmap="viridis")
    axes[0].set(xlabel="Downsampled GRU timestep within window", ylabel="Window start time [s]",
                title=f"{split}/{stem}: sampled windows")
    fig.colorbar(image, ax=axes[0], label="Attention weight")
    axes[1].bar(np.arange(50), weights[selected], color=COLORS[MODEL], width=.9)
    axes[1].axhline(1/50, ls="--", color="gray", label="Uniform weight (1/50)")
    axes[1].set(xlabel="Downsampled GRU timestep within window", ylabel="Attention weight",
                title=f"Middle window: start {t[starts[selected]]-t[0]:.2f} s")
    axes[1].legend(); axes[1].grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_summaries(out, reference, summaries):
    comparison = out / "comparison"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, title in zip(axes, ("RMSE", "ATE_official_m", "RTE_60s_official_m"),
                               ("Velocity RMSE [m/s]", "Official ATE [m]", "Official 60s RTE [m]")):
        for i, name in enumerate(NAMES):
            values = [next(r[metric] for r in summaries if r["split"] == split and r["model"] == name) for split in SPLITS]
            bars = ax.bar(np.arange(2)+(i-1)*.25, values, .25, label=NAMES[name], color=COLORS[name])
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_xticks(np.arange(2), SPLITS); ax.set_title(title); ax.grid(axis="y", alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(comparison / "test_summary.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in NAMES:
        log = read_csv((out if name == MODEL else reference) / name / "training_log.csv")
        for ax, metric in zip(axes, ("train_mse", "val_mse")):
            ax.plot([int(r["epoch"]) for r in log], [float(r[metric]) for r in log], label=NAMES[name], color=COLORS[name])
            ax.set(xlabel="Epoch", ylabel=metric + " [(m/s)^2]"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(comparison / "training_comparison.png", dpi=160); plt.close(fig)


def write_report(out, reference, training, summaries, rows):
    baseline = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    training_table = []
    for name in NAMES:
        r = training if name == MODEL else baseline["training"][name]
        training_table.append(f"| {NAMES[name]} | {r['parameter_count']:,} | {r['epochs_completed']} | {r['best_epoch']} | {r['best_val_mse']:.8f} |")
    metrics_table = [f"| {r['split']} | {NAMES[r['model']]} | {r['RMSE']:.6f} | {r['MAE']:.6f} | {r['ATE_official_m']:.3f} | {r['RTE_60s_official_m']:.3f} |" for r in summaries]
    changes = []
    for split in SPLITS:
        control = next(r for r in summaries if r["split"] == split and r["model"] == "cnn_gru")
        new = next(r for r in summaries if r["split"] == split and r["model"] == MODEL)
        changes.append(dict(split=split, **{key: 100*(control[key]-new[key])/control[key] for key in METRICS}))
    dump(out / "comparison/improvement_vs_cnn_gru_percent.json", dict(
        definition="100 * (CNN-GRU metric - Attention metric) / CNN-GRU metric; positive means error reduction", rows=changes))
    text = """# CNN-GRU-Attention：固定协议实验

Attention 模型从头完成 30 轮训练，以验证集最低 MSE 选择权重。主要参考为 RoNIN ResNet18，直接消融对照为 CNN-GRU；两者引用此前固定协议实验的已保存预测，本次没有重新训练或改写基线结果。

共同训练设置：窗口 200，步长 10，batch 256，Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，MSE，seed=42；无增强、标准化、学习率调度、早停或梯度裁剪。CUDA float32，关闭 TF32，开启确定性算法。数据文件 SHA256、划分、窗口索引、标签及 30 轮样本顺序均与参考实验核对一致。

监督目标是输入窗口的平均 XY 速度；输入只有六通道 IMU，不包含位置或标签。CNN-GRU-Attention 与 CNN-GRU 共用 CNN、GRU、回归头定义，将末状态汇总改成所有 50 个 GRU 时刻的加性时间注意力汇总，新增 2,112 个参数。

| 模型 | 参数量 | 完成轮数 | 最佳轮次 | 最佳验证 MSE |
|---|---:|---:|---:|---:|
""" + "\n".join(training_table) + """

| 测试集 | 模型 | 速度 RMSE (m/s) | MAE (m/s) | 官方 ATE (m) | 官方 60s RTE (m) |
|---|---|---:|---:|---:|---:|
""" + "\n".join(metrics_table) + f"""

训练/验证/seen/unseen 划分为 72/13/35/32 条交付序列；测试不筛选异常速度。速度指标按所有窗口的两个分量平均；ATE/RTE 使用官方坐标误差定义、先逐序列计算再等权平均，60s 对应 12,000 帧。轨迹共用 reconstruct()，仅用首个真值位置初始化，不进行轨迹旋转拟合。测试集不参与选模。

基线来源：`{reference}`。保存的 Attention 预测已经重新读取并独立复算指标；基线预测与当前数据的索引、标签、时间、真值完全一致，基线汇总指标也已经复核。

目录入口：index.html；comparison/ 为三模型共同轨迹、训练曲线及指标；cnn_gru_attention/seen_test 和 unseen_test 为 Attention 的 67 条轨迹、速度、误差图与逐窗口预测 NPZ；每个 NPZ 额外包含 [N,50] attention_weights，对应图为 *_attention.png。权重对应 CNN 降采样后的 GRU 时刻，并非原始 200 帧的逐帧解释，也不能单独证明因果关系或性能提升。

本实验只有 seed=42。正负变化均如实报告；单次结果不能证明统计显著性。improvement_vs_cnn_gru_percent.json 中正数表示误差下降，负数表示变差。
"""
    (out / "README_实验结果.md").write_text(text, encoding="utf-8")
    table = "".join(f"<tr><td>{r['split']}</td><td>{NAMES[r['model']]}</td><td>{r['RMSE']:.4f}</td><td>{r['MAE']:.4f}</td><td>{r['ATE_official_m']:.3f}</td><td>{r['RTE_60s_official_m']:.3f}</td></tr>" for r in summaries)
    sequences = [dict(split=r["split"], sequence=r["sequence"]) for r in rows if r["model"] == MODEL]
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Attention 固定协议实验</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f4f6f8;color:#182735;max-width:1250px;margin:30px auto;padding:0 20px}section{background:white;border-radius:10px;padding:22px;margin:18px 0}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,td:nth-child(2){text-align:left}select,button{font:inherit;padding:7px;margin:4px}a{color:#17639a}</style>
<h1>CNN-GRU-Attention：固定协议对照实验</h1><p>三模型均训练 30 轮，按验证 MSE 选模；seed=42。Attention 已完成训练和 67 条测试序列评估，基线引用此前同协议结果。</p>
<section><h2>测试结果</h2><table><tr><th>测试集</th><th>模型</th><th>速度 RMSE</th><th>速度 MAE</th><th>官方 ATE</th><th>官方 60s RTE</th></tr>__TABLE__</table>
<p>速度单位 m/s，轨迹单位 m；单种子结果。<a href="README_实验结果.md">实验说明</a> · <a href="comparison/test_metrics.csv">指标 CSV</a> · <a href="protocol.json">训练协议</a> · <a href="result_verification.json">复核记录</a></p></section>
<section><h2>训练与测试对比</h2><img src="comparison/training_comparison.png"><img src="comparison/test_summary.png"></section>
<section><h2>逐序列轨迹、速度和误差</h2><select id="split"><option>seen_test</option><option>unseen_test</option></select><select id="seq"></select><button id="prev">上一条</button><button id="next">下一条</button><img id="comparison"><p><a id="prediction">Attention 预测数据及全部权重 NPZ</a></p></section>
<section><h2>时间注意力</h2><p>左图为序列中均匀抽取的窗口，右图为中间窗口；横轴为 CNN 降采样后的 50 个 GRU 时刻。权重图用于观察模型行为，不代表因果解释。</p><img id="attention"></section>
<script>const data=__DATA__,s=document.getElementById('split'),q=document.getElementById('seq');function render(){document.getElementById('comparison').src='comparison/'+s.value+'/'+q.value+'.png';document.getElementById('attention').src='cnn_gru_attention/'+s.value+'/'+q.value+'_attention.png';document.getElementById('prediction').href='cnn_gru_attention/'+s.value+'/'+q.value+'_predictions.npz';}function update(){q.replaceChildren(...data.filter(r=>r.split===s.value).map(r=>{let o=document.createElement('option');o.value=o.textContent=r.sequence;return o;}));render();}s.onchange=update;q.onchange=render;for(const [id,d] of [['prev',-1],['next',1]])document.getElementById(id).onclick=()=>{q.selectedIndex=(q.selectedIndex+d+q.length)%q.length;render();};update();</script></html>"""
    (out / "index.html").write_text(page.replace("__TABLE__", table).replace("__DATA__", json.dumps(sequences)), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    out = args.run.resolve()
    reference = Path(json.loads((out / "reference.json").read_text(encoding="utf-8"))["reference_run"])
    protocol, digest, orders = load_reference(reference)
    if json.loads((out / "protocol.json").read_text(encoding="utf-8")) != dict(protocol=protocol, protocol_sha256=digest):
        raise ValueError("Attention protocol differs from baseline")
    training = json.loads((out / MODEL / "summary.json").read_text(encoding="utf-8"))
    if training["status"] != "training_completed" or training["epochs_completed"] != protocol["epochs"]:
        raise ValueError("Full training must finish before test evaluation")
    if not training["checkpoint_reload_passed"] or training["protocol_sha256"] != digest:
        raise ValueError("Training checkpoint acceptance failed")
    log = read_csv(out / MODEL / "training_log.csv")
    best = min(log, key=lambda r: float(r["val_mse"]))
    assert len(log) == 30 and int(best["epoch"]) == training["best_epoch"]
    manifest = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
    for i, row in enumerate(log, 1):
        assert int(row["epoch"]) == i and row["order_sha256"] == orders[str(i)]
        assert int(row["train_samples"]) == manifest["train"]["windows"]
        assert int(row["val_samples"]) == manifest["val"]["windows"]
    for path in (out / "comparison", out / MODEL / "seen_test", out / MODEL / "unseen_test", out / "evaluation_summary.json"):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation: {path}")
    baseline_hashes = json.loads((reference / "source_fingerprints.json").read_text(encoding="utf-8"))
    for name in ("scripts/run_ronin_npz_baseline.py", "ronin-master/source/metric.py", "cnn_baseline/compare_with_resnet.py"):
        if sha(ROOT / name) != baseline_hashes[str(Path(name))]:
            raise ValueError(f"Common evaluation source changed since baseline: {name}")
    data_root = ROOT / "RoNIN_Team_Delivery/dataset"
    fingerprints = read_csv(reference / "data_fingerprints.csv")
    for row in fingerprints:
        p = data_root / row["path"]
        if p.stat().st_size != int(row["bytes"]) or sha(p) != row["sha256"]:
            raise ValueError(f"Data changed: {p}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for fixed evaluation")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    checkpoint_path = out / MODEL / "checkpoint_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    validate_checkpoint(checkpoint, MODEL, 200, 10)
    assert checkpoint["protocol_sha256"] == digest and checkpoint["epoch"] == training["best_epoch"]
    model = CNNGRUAttention().to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    (out / "comparison").mkdir()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(out / "evaluation.log", encoding="utf-8")])
    for path in (Path(__file__), ROOT / "scripts/run_ronin_npz_baseline.py", ROOT / "ronin-master/source/metric.py"):
        shutil.copy2(path, out / "source_snapshot" / path.name)
    provenance = dict(protocol_sha256=digest, reference_run=str(reference),
        checkpoint_sha256=sha(checkpoint_path), checkpoint_epoch=checkpoint["epoch"],
        script_sha256=sha(Path(__file__)), common_evaluation_sources_match_baseline=True,
        data_files_match_reference=True, evaluation_batch_size=protocol["evaluation_batch_size"],
        baseline_predictions_reused=True, baseline_prediction_sha256={},
        attention_weights="[windows,50], CNN-downsampled GRU timesteps; not causal explanations")
    rows, summaries, verification = [], [], {}
    reference_sequences = {(r["model"], r["split"], r["sequence"]): r for r in read_csv(reference / "comparison/sequence_metrics.csv")}
    reference_summaries = {(r["model"], r["split"]): r for r in read_csv(reference / "comparison/test_metrics.csv")}
    for split in SPLITS:
        folder = out / MODEL / split; folder.mkdir()
        comparison = out / "comparison" / split; comparison.mkdir()
        names = [r["path"] for r in fingerprints if r["split"] == split]
        totals = {name: [0., 0., 0] for name in NAMES}
        for no, filename in enumerate(names, 1):
            f, p, t, _, y = load_sequence(data_root / filename)
            starts = np.arange(0, len(y), 10, dtype=np.int64)
            truth = y[starts].astype(np.float64); gt = p[:, :2]; stem = Path(filename).stem
            predictions, trajectories = {}, {}
            for name in ("resnet18", "cnn_gru"):
                path = reference / name / split / f"{stem}_predictions.npz"
                provenance["baseline_prediction_sha256"][str(path.relative_to(reference))] = sha(path)
                with np.load(path, allow_pickle=False) as ref:
                    for key, expected in dict(starts=starts, timestamps=t, target_velocity=truth,
                                              ground_truth_positions=gt, prediction_timestamps=t[starts],
                                              window_end_timestamps=t[starts+200]).items():
                        np.testing.assert_array_equal(ref[key], expected)
                    predictions[name] = ref["predicted_velocity"].copy()
                    trajectories[name] = ref["predicted_positions"].copy()
                np.testing.assert_array_equal(reconstruct(t, p, starts, predictions[name]), trajectories[name])
            pred, weights = predict_attention(model, f, starts, device, protocol["evaluation_batch_size"])
            traj = reconstruct(t, p, starts, pred)
            predictions[MODEL] = pred; trajectories[MODEL] = traj
            for name in NAMES:
                metrics = trajectory_metrics(predictions[name], truth, trajectories[name], gt)
                if name != MODEL:
                    old = reference_sequences[name, split, stem]
                    for key, value in metrics.items():
                        np.testing.assert_allclose(value, float(old[key]), rtol=1e-12, atol=1e-12)
                rows.append(dict(split=split, sequence=stem, model=name, **metrics))
                error = predictions[name]-truth
                totals[name][0] += float(np.sum(error**2)); totals[name][1] += float(np.sum(abs(error))); totals[name][2] += len(truth)
            np.savez_compressed(folder / f"{stem}_predictions.npz", starts=starts,
                prediction_timestamps=t[starts], window_end_timestamps=t[starts+200],
                predicted_velocity=pred, target_velocity=truth, timestamps=t,
                predicted_positions=traj, ground_truth_positions=gt, attention_weights=weights)
            plot_sequence(folder / f"{stem}.png", split, stem, t, starts, gt, truth, {MODEL: pred}, {MODEL: traj})
            plot_sequence(comparison / f"{stem}.png", split, stem, t, starts, gt, truth, predictions, trajectories)
            plot_attention(folder / f"{stem}_attention.png", split, stem, t, starts, weights)
            logging.info("TEST %s %d/%d %s RMSE=%.6f ATE=%.3f RTE=%.3f", split, no, len(names), stem,
                         metrics["RMSE"], metrics["ATE_official_m"], metrics["RTE_60s_official_m"])
            dump(out / "progress.json", dict(stage="evaluation", split=split, sequence=no, sequences=len(names)))
        for name, (sq, absolute, n) in totals.items():
            selected = [r for r in rows if r["split"] == split and r["model"] == name]
            summary = dict(split=split, model=name, sequences=len(names), windows=n, MSE=sq/(2*n),
                RMSE=math.sqrt(sq/(2*n)), MAE=absolute/(2*n),
                ATE_official_m=float(np.mean([r["ATE_official_m"] for r in selected])),
                RTE_60s_official_m=float(np.mean([r["RTE_60s_official_m"] for r in selected])))
            if name != MODEL:
                for key in METRICS:
                    np.testing.assert_allclose(summary[key], float(reference_summaries[name, split][key]), rtol=1e-12, atol=1e-12)
            summaries.append(summary)
        # Re-read saved Attention predictions independently to verify all metrics.
        sq = absolute = 0.; count = 0; ates = []; rtes = []
        for row in [r for r in rows if r["split"] == split and r["model"] == MODEL]:
            with np.load(folder / f"{row['sequence']}_predictions.npz", allow_pickle=False) as saved:
                assert all(np.isfinite(saved[k]).all() for k in saved.files)
                actual = trajectory_metrics(saved["predicted_velocity"], saved["target_velocity"],
                                            saved["predicted_positions"], saved["ground_truth_positions"])
                for key, value in actual.items():
                    np.testing.assert_allclose(value, row[key], rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(saved["attention_weights"].sum(axis=1), 1, atol=2e-6, rtol=0)
                e = saved["predicted_velocity"]-saved["target_velocity"]
                sq += float(np.sum(e**2)); absolute += float(np.sum(abs(e))); count += len(e)
                ates.append(actual["ATE_official_m"]); rtes.append(actual["RTE_60s_official_m"])
        actual = dict(MSE=sq/(2*count), RMSE=math.sqrt(sq/(2*count)), MAE=absolute/(2*count),
                      ATE_official_m=float(np.mean(ates)), RTE_60s_official_m=float(np.mean(rtes)))
        summary = next(r for r in summaries if r["split"] == split and r["model"] == MODEL)
        for key, value in actual.items():
            np.testing.assert_allclose(value, summary[key], rtol=1e-12, atol=1e-12)
        verification[split] = dict(sequences_checked=len(names), windows=count, saved_metrics_match=True,
            attention_weights_finite_nonnegative_normalized=True, baseline_metrics_recomputed=True,
            labels_times_indices_gt_identical_between_models=True, reconstructed_baseline_positions_identical=True)
    write_csv(out / "comparison/test_metrics.csv", summaries)
    write_csv(out / "comparison/sequence_metrics.csv", rows)
    write_csv(out / MODEL / "test_metrics.csv", [r for r in summaries if r["model"] == MODEL])
    write_csv(out / MODEL / "sequence_metrics.csv", [r for r in rows if r["model"] == MODEL])
    plot_summaries(out, reference, summaries)
    write_report(out, reference, training, summaries, rows)
    dump(out / "evaluation_config.json", provenance)
    dump(out / "result_verification.json", verification)
    dump(out / "evaluation_summary.json", dict(status="completed", protocol_sha256=digest, test_metrics=summaries))
    training.update(status="completed", test_metrics=[r for r in summaries if r["model"] == MODEL])
    dump(out / MODEL / "summary.json", training)
    dump(out / "summary.json", dict(status="completed", protocol_sha256=digest, training={MODEL: training}, test_metrics=summaries))
    checks = json.loads((out / "acceptance_checks.json").read_text(encoding="utf-8"))
    checks.update(evaluation_completed=True, complete_30_epochs=True, saved_metrics_recomputed=True,
                  test_sequences=67, reference_results_not_modified=True)
    dump(out / "acceptance_checks.json", checks)
    dump(out / "progress.json", dict(stage="completed"))
    logging.info("COMPLETED %s", json.dumps(summaries))


if __name__ == "__main__":
    main()
