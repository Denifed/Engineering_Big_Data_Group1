"""Frozen CNN-GRU C4 yaw ensemble, crossed with three pre-existing BN states.

This is inference-only group averaging, not RIO training or test-time training.
Original experimental artifacts are never modified.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import hashlib
import html
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn_baseline.dataset import validate_checkpoint
from cnn_baseline.model_cnn_gru import CNNGRU
from cnn_baseline.run_fixed_comparison import dump, sha
from cnn_baseline.train_drift_comparison import setup_runtime, state_sha, bias_metrics
from cnn_baseline.diagnose_bn_calibration import tensor_hash
from cnn_baseline.compare_with_resnet import trajectory_metrics, write_csv
from scripts.run_ronin_npz_baseline import load_sequence, reconstruct

OLD = ROOT / "results/fixed_resnet18_vs_cnn_gru_20261025"
BN = ROOT / "results/bn_calibration_diagnosis_20261026"
DRIFT = ROOT / "results/drift_loss_cnn_gru_20261026"
SOURCES = {
    "original": OLD / "cnn_gru/checkpoint_best.pt",
    "mixed": BN / "old_random_mse/mixed_windows/checkpoint_bn_calibrated.pt",
    "consecutive": BN / "old_random_mse/consecutive_segments/checkpoint_bn_calibrated.pt",
}
VARIANTS = tuple(f"{bn}_{mode}" for bn in SOURCES for mode in ("single", "c4"))
LABELS = {"original_single": "Original / single", "original_c4": "Original / C4",
          "mixed_single": "Mixed BN / single", "mixed_c4": "Mixed BN / C4",
          "consecutive_single": "Consecutive BN / single", "consecutive_c4": "Consecutive BN / C4"}
COLORS = dict(zip(VARIANTS, ("#777777", "#1675b8", "#a28535", "#e98d22", "#7a72a5", "#2d9b74")))
MACRO = ("ATE_official_m", "RTE_60s_official_m", "endpoint_error_m", "mean_velocity_bias_norm_mps")
COMPARE = ("RMSE", "MAE", *MACRO)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def yaw_features(x, quarter_turns):
    """Rotate both global gyro and accelerometer XY; preserve Z and time."""
    k = quarter_turns % 4
    out = x.clone()
    for a, b in ((0, 1), (3, 4)):
        if k == 1:
            out[:, a], out[:, b] = -x[:, b], x[:, a]
        elif k == 2:
            out[:, a], out[:, b] = -x[:, a], -x[:, b]
        elif k == 3:
            out[:, a], out[:, b] = x[:, b], -x[:, a]
    return out


def yaw_velocity(v, quarter_turns):
    k = quarter_turns % 4
    if k == 0:
        return v.clone()
    a, b = v[..., 0], v[..., 1]
    if k == 1:
        return torch.stack((-b, a), dim=-1)
    if k == 2:
        return -v
    return torch.stack((b, -a), dim=-1)


def aligned_branches(model, x):
    return torch.stack([yaw_velocity(model(yaw_features(x, k)), -k) for k in range(4)], dim=1)


def checks(model):
    """Check rotation algebra and the actual nonlinear model's C4 projection."""
    x = torch.randn(32, 6, 200, device="cuda")
    v = torch.randn(32, 2, device="cuda")
    with torch.inference_mode():
        for k in range(4):
            torch.testing.assert_close(yaw_features(yaw_features(x, k), -k), x, rtol=0, atol=0)
            torch.testing.assert_close(yaw_velocity(yaw_velocity(v, k), -k), v, rtol=0, atol=0)
            torch.testing.assert_close(yaw_features(x, k)[:, [2, 5]], x[:, [2, 5]], rtol=0, atol=0)
            torch.testing.assert_close(torch.linalg.vector_norm(yaw_velocity(v, k), dim=-1),
                                       torch.linalg.vector_norm(v, dim=-1), rtol=1e-6, atol=1e-6)
        for a in range(4):
            for b in range(4):
                torch.testing.assert_close(yaw_features(yaw_features(x, a), b), yaw_features(x, a+b), rtol=0, atol=0)
        projected = aligned_branches(model, x).double().mean(1)
        maximum = 0.
        for k in range(4):
            rotated = aligned_branches(model, yaw_features(x, k)).double().mean(1)
            expected = yaw_velocity(projected, k)
            maximum = max(maximum, float((rotated-expected).abs().max()))
            torch.testing.assert_close(rotated, expected, rtol=1e-6, atol=1e-7)
    # An exactly equivariant synthetic predictor must be unchanged by averaging.
    def equivariant(z):
        return z[:, :2].mean(-1) + 2*z[:, 3:5].mean(-1)
    with torch.inference_mode():
        torch.testing.assert_close(aligned_branches(equivariant, x).double().mean(1),
                                   equivariant(x).double(), rtol=1e-6, atol=1e-7)
    return dict(status="passed", quarter_turn_composition_and_inverse=True,
                gyro_and_acceleration_rotated_together=True, z_preserved=True,
                equivariant_predictor_preserved=True, model_c4_equivariance_max_abs_mps=maximum)


def infer(model, features, starts, batch_size):
    ft = torch.from_numpy(features).cuda()
    offsets = torch.arange(200, device="cuda")
    pieces = []
    with torch.inference_mode():
        for begin in range(0, len(starts), batch_size):
            ids = torch.from_numpy(starts[begin:begin+batch_size]).cuda()
            x = ft[ids[:, None] + offsets].transpose(1, 2).contiguous()
            pieces.append(aligned_branches(model, x).cpu().numpy())
    return np.concatenate(pieces)


def summarize(split, variant, records, sq, absolute, count):
    return dict(split=split, variant=variant, sequences=len(records), windows=count,
                MSE=sq/(2*count), RMSE=math.sqrt(sq/(2*count)), MAE=absolute/(2*count),
                **{key: float(np.mean([r[key] for r in records])) for key in MACRO})


def benchmark(model, features, starts):
    ft = torch.from_numpy(features).cuda()
    ids = torch.from_numpy(starts[:256]).cuda()
    x = ft[ids[:, None]+torch.arange(200, device="cuda")].transpose(1, 2).contiguous()
    measurements = {}
    with torch.inference_mode():
        for name, fn in (("single", lambda: model(x)),
                         ("c4", lambda: aligned_branches(model, x).double().mean(1))):
            for _ in range(10):
                fn()
            torch.cuda.synchronize(); values = []
            for _ in range(30):
                torch.cuda.synchronize(); tic = time.perf_counter()
                fn(); torch.cuda.synchronize()
                values.append((time.perf_counter()-tic)*1000)
            measurements[name] = dict(batch_size=256, median_batch_ms=float(np.median(values)),
                                      p90_batch_ms=float(np.percentile(values, 90)),
                                      median_ms_per_window=float(np.median(values)/256))
    measurements["c4_to_single_median_ratio"] = measurements["c4"]["median_batch_ms"]/measurements["single"]["median_batch_ms"]
    measurements["scope"] = "same validation inputs; warmed GPU; excludes disk, trajectory reconstruction and plotting"
    return measurements


def plot_sequence(path, split, stem, t, starts, truth, gt, predictions, trajectories):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    times = t[starts]-t[0]
    axes[0, 0].plot(gt[:, 0], gt[:, 1], c="black", label="Ground truth", lw=1.2)
    for name in VARIANTS:
        lw = 1.2 if name.startswith("original") else .7
        axes[0, 0].plot(trajectories[name][:, 0], trajectories[name][:, 1], c=COLORS[name], lw=lw, label=LABELS[name])
        axes[1, 0].plot(t-t[0], np.linalg.norm(trajectories[name]-gt, axis=1), c=COLORS[name], lw=lw)
    axes[0, 0].axis("equal"); axes[0, 0].set(xlabel="X [m]", ylabel="Y [m]", title=f"{split}/{stem}")
    axes[0, 0].legend(fontsize=7)
    for j in range(2):
        ax = axes[0, j+1]
        ax.plot(times, truth[:, j], c="black", lw=.6, alpha=.7, label="Target")
        for name in ("original_single", "original_c4"):
            ax.plot(times, predictions[name][:, j], c=COLORS[name], lw=.6, label=LABELS[name])
        ax.set(xlabel="Time [s]", ylabel=("Vx" if j == 0 else "Vy")+" [m/s]")
        ax.legend(fontsize=7)
    delta = t[starts+10]-t[starts]
    for name in VARIANTS:
        residual = np.cumsum((predictions[name]-truth)*delta[:, None], axis=0)
        axes[1, 1].plot(times, np.linalg.norm(residual, axis=1), c=COLORS[name], lw=.7)
        axes[1, 2].plot(times, np.linalg.norm(residual/np.cumsum(delta)[:, None], axis=1), c=COLORS[name], lw=.7)
    axes[1, 0].set(xlabel="Time [s]", ylabel="Euclidean position error [m]")
    axes[1, 1].set(xlabel="Time [s]", ylabel="Integrated velocity residual [m]")
    axes[1, 2].set(xlabel="Time [s]", ylabel="Time-weighted prefix bias [m/s]")
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_summary(path, rows, splits):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, key, title in zip(axes.flat, ("RMSE", "ATE_official_m", "RTE_60s_official_m", "MAE", "endpoint_error_m", "mean_velocity_bias_norm_mps"),
                             ("Velocity RMSE [m/s]", "Official ATE [m]", "60s RTE [m]", "Velocity MAE [m/s]", "Endpoint error [m]", "Mean velocity bias [m/s]")):
        for j, variant in enumerate(VARIANTS):
            values = [next(r[key] for r in rows if r["variant"] == variant and r["split"] == split) for split in splits]
            bars = ax.bar(np.arange(len(splits))+(j-2.5)*.13, values, width=.125, color=COLORS[variant], label=LABELS[variant])
            ax.bar_label(bars, fmt="%.3f", fontsize=6, rotation=65, padding=2)
        ax.set_xticks(range(len(splits)), splits); ax.set_title(title); ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True); ax.margins(y=.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    setup_runtime(); torch.manual_seed(42)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out / "comparison").mkdir(); (out / "source_snapshot").mkdir()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", encoding="utf-8")])
    started = time.perf_counter()
    protocol = dict(experiment="frozen_cnn_gru_c4_yaw_ensemble_v1", source="old shuffled-window CNN-GRU best epoch 8",
        angles_degrees=[0, 90, 180, 270], ensemble="inverse rotate four FP32 velocity outputs, equal average in float64",
        input="rotate both gyro XY and acceleration XY about global Z; preserve Z and time",
        bn_conditions=list(SOURCES), inference_conditions=["single", "c4"], all_variants=list(VARIANTS),
        label_protocol="window_mean_xy_velocity_v1", window_size=200, step_size=10,
        validation_batch_size=256, test_batch_size=512, data_splits=["val", "seen_test", "unseen_test"],
        primary_contrast=["original_single", "original_c4"],
        secondary_contrast="within each fixed BN state, single vs C4; report every condition",
        candidate_selection="none; fixed angles and weights, all six conditions evaluated irrespective of validation gains",
        overall_improvement_rule="both seen/unseen macro ATE lower; unseen endpoint, bias norm and velocity RMSE no worse than original_single",
        limitations="exploratory reused tests; no new training, adaptation, sensor information, SE2 fit, or drift guarantee",
        metric_definition="same official coordinate-wise ATE, 60s RTE and reconstruction; trajectory macro, velocity window weighted")
    dump(out / "protocol.json", protocol)
    dump(out / "environment.json", dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__),
        numpy=np.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    (out / "PLAN_执行计划.md").write_text("""# 水平旋转一致性：固定执行计划

1. 固定旧 CNN-GRU 第 8 轮权重、200 帧窗口、10 帧步长和所有学习参数。
2. 三种既有 BN 状态（原始、训练集随机混合校准、训练集连续片段校准）各做单次与 C4 集成，共六组；校准权重直接复用，不重新校准。
3. C4 固定采用 0/90/180/270 度，同时旋转角速度与加速度 XY；预测逆旋转回原坐标系后等权平均。Z 和时间不变。不使用任何真值决定旋转或融合权重。
4. 先验证旋转代数、等变预测器保持性及实际网络集成的四方向等变性，再评估完整验证集，随后评估两个完整测试集。所有条件全部报告，不以验证结果调角度、删条件或调权重。
5. 主要比较 original_single 与 original_c4；BN 的两组交叉条件用于解释作用。统一轨迹重建和指标；逐序列保存四方向输出、最终预测、真值、轨迹、累计速度残差。
6. 整体改善的预设判断：seen/unseen 平均 ATE 均下降，同时 unseen 终点误差、速度偏置和速度 RMSE 不劣于原始单次模型。否则仅报告对应指标收益或失败，不声称整体改进。
7. 报告推理耗时、逐序列改善比例、分布尾部和成功/失败例子。全部结果在新目录保存，核验原始文件未变。

本轮仅验证推理期群平均，不实施旋转一致性损失训练，不构成完整 RIO 复现。原有测试集已多次用于分析，结果属于探索性实验，不宣称独立盲测或统计显著性。
""", encoding="utf-8")
    assert read_json(BN / "final_audit.json")["status"] == "passed"
    watched = {p for root in (OLD, BN, DRIFT) for p in root.rglob("*") if p.is_file()}
    sources = [*Path(__file__).parent.glob("*.py"), ROOT / "scripts/run_ronin_npz_baseline.py",
               ROOT / "ronin-master/source/metric.py", *list((ROOT / "RoNIN_Team_Delivery/splits").glob("*"))]
    watched.update(sources)
    for p in sources:
        shutil.copy2(p, out / "source_snapshot" / p.name)
    dump(out / "source_fingerprints.json", {str(p.relative_to(ROOT)): sha(p) for p in sources})
    fingerprints = read_csv(OLD / "data_fingerprints.csv")
    for r in fingerprints:
        p = ROOT / "RoNIN_Team_Delivery/dataset" / r["path"]
        assert sha(p) == r["sha256"] and p.stat().st_size == int(r["bytes"])
        watched.add(p)
    before = {str(p.relative_to(ROOT)): sha(p) for p in sorted(watched)}
    dump(out / "original_artifact_hashes.json", before)
    write_csv(out / "data_fingerprints.csv", fingerprints)
    models = {}; states = {}; provenance = {}; sanity = {}
    for name, path in SOURCES.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint(checkpoint, "cnn_gru", 200, 10)
        assert checkpoint["epoch"] == 8
        model = CNNGRU().cuda().eval(); model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        for p in model.parameters():
            p.requires_grad_(False)
        param_hash = tensor_hash(model.named_parameters())
        if models:
            assert param_hash == next(iter(provenance.values()))["learned_parameters_sha256"]
        models[name] = model; states[name] = state_sha(model)
        provenance[name] = dict(checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=sha(path),
                                learned_parameters_sha256=param_hash, state_sha256=states[name], epoch=8)
        sanity[name] = checks(model)
    dump(out / "checkpoint_provenance.json", provenance)
    dump(out / "sanity_checks.json", sanity)
    validation_old = {r["variant"]: r for r in read_csv(BN / "validation_metrics.csv") if r["model"] == "old_random_mse"}
    reference_map = {"original": "original", "mixed": "mixed_windows", "consecutive": "consecutive_segments"}
    rows = []; summaries = []; consistency = []; verification = []; baseline_checks = []
    for split in ("val", "seen_test", "unseen_test"):
        files = [r["path"] for r in fingerprints if r["split"] == split]
        assert len(files) == dict(val=13, seen_test=35, unseen_test=32)[split]
        (out / "comparison" / split).mkdir()
        for variant in VARIANTS:
            (out / variant / split).mkdir(parents=True)
        totals = {v: [0., 0., 0] for v in VARIANTS}
        bn_consistency = {n: [0., 0] for n in SOURCES}
        for no, filename in enumerate(files, 1):
            f, p, t, _, y = load_sequence(ROOT / "RoNIN_Team_Delivery/dataset" / filename)
            starts = np.arange(0, len(y), 10, dtype=np.int64)
            truth = y[starts].astype(np.float64); gt = p[:, :2]; stem = Path(filename).stem
            if split == "val" and no == 1:
                dump(out / "inference_benchmark.json", benchmark(models["original"], f, starts))
            predictions = {}; trajectories = {}
            for bn_name, model in models.items():
                branches = infer(model, f, starts, 256 if split == "val" else 512)
                aligned = branches.astype(np.float64)
                diff = aligned[:, 1:]-aligned[:, :1]
                bn_consistency[bn_name][0] += float(np.sum(diff**2))
                bn_consistency[bn_name][1] += diff.size
                np.savez_compressed(out / f"{bn_name}_c4" / split / f"{stem}_aligned_branches.npz",
                                    starts=starts, aligned_velocity=branches, angles_degrees=np.array([0, 90, 180, 270]))
                for mode, pred in (("single", aligned[:, 0]), ("c4", aligned.mean(axis=1))):
                    name = f"{bn_name}_{mode}"
                    positions = reconstruct(t, p, starts, pred)
                    row = dict(split=split, sequence=stem, variant=name,
                        **trajectory_metrics(pred, truth, positions, gt), **bias_metrics(pred, truth, t, starts))
                    rows.append(row); predictions[name] = pred; trajectories[name] = positions
                    e = pred-truth; delta = t[starts+10]-t[starts]
                    totals[name][0] += float(np.sum(e**2)); totals[name][1] += float(np.sum(abs(e))); totals[name][2] += len(e)
                    np.savez_compressed(out / name / split / f"{stem}_predictions.npz", starts=starts,
                        prediction_timestamps=t[starts], window_end_timestamps=t[starts+200],
                        predicted_velocity=pred, target_velocity=truth, timestamps=t,
                        predicted_positions=positions, ground_truth_positions=gt,
                        prediction_spacing=delta, cumulative_velocity_residual_m=np.cumsum(e*delta[:, None], axis=0))
                assert state_sha(model) == states[bn_name], "Inference changed state"
            if split != "val":
                with np.load(OLD / "cnn_gru" / split / f"{stem}_predictions.npz", allow_pickle=False) as ref:
                    for key, actual in dict(starts=starts, target_velocity=truth, timestamps=t, ground_truth_positions=gt).items():
                        np.testing.assert_array_equal(ref[key], actual)
                    np.testing.assert_array_equal(ref["predicted_velocity"], predictions["original_single"])
                    np.testing.assert_array_equal(ref["predicted_positions"], trajectories["original_single"])
                baseline_checks.append(dict(split=split, sequence=stem, historical_predictions_exactly_reproduced=True))
            plot_sequence(out / "comparison" / split / f"{stem}.png", split, stem, t, starts, truth, gt, predictions, trajectories)
            dump(out / "progress.json", dict(stage="evaluation", split=split, sequences_done=no, total=len(files)))
            logging.info("SEQUENCE_DONE %s %d/%d %s baseline_ATE=%.4f C4_ATE=%.4f", split, no, len(files), stem,
                next(r["ATE_official_m"] for r in rows[-6:] if r["variant"] == "original_single"),
                next(r["ATE_official_m"] for r in rows[-6:] if r["variant"] == "original_c4"))
        for name, (sq, absolute, count) in totals.items():
            selected = [r for r in rows if r["split"] == split and r["variant"] == name]
            summaries.append(summarize(split, name, selected, sq, absolute, count))
        for bn_name, (sq, count) in bn_consistency.items():
            consistency.append(dict(split=split, bn_state=bn_name, single_rotation_consistency_RMSE_mps=math.sqrt(sq/count),
                                    definition="RMS component disagreement of inverse-rotated 90/180/270 predictions with original prediction"))
        if split == "val":
            for name in SOURCES:
                actual = next(r for r in summaries if r["split"] == split and r["variant"] == f"{name}_single")
                ref = validation_old[reference_map[name]]
                for key in COMPARE:
                    # Earlier validation batched across sequence boundaries;
                    # this run batches per sequence, changing tail CUDA shapes.
                    np.testing.assert_allclose(actual[key], float(ref[key]), rtol=2e-6, atol=1e-6)
            dump(out / "validation_summary.json", dict(status="completed", metrics=[r for r in summaries if r["split"] == "val"],
                 angles_and_conditions_unchanged_for_tests=True))
    # Re-open every saved prediction, reconstruct trajectories and recompute metrics.
    for split in ("val", "seen_test", "unseen_test"):
        for variant in VARIANTS:
            selected = [r for r in rows if r["split"] == split and r["variant"] == variant]
            sq = absolute = 0.; count = 0; recomputed = []
            for r in selected:
                path = out / variant / split / f"{r['sequence']}_predictions.npz"
                with np.load(path, allow_pickle=False) as z:
                    assert all(np.isfinite(z[k]).all() for k in z.files)
                    pred = z["predicted_velocity"]; truth = z["target_velocity"]
                    got = dict(**trajectory_metrics(pred, truth, z["predicted_positions"], z["ground_truth_positions"]),
                               **bias_metrics(pred, truth, z["timestamps"], z["starts"]))
                    for key in ("MSE", *COMPARE):
                        np.testing.assert_allclose(got[key], r[key], rtol=1e-12, atol=1e-12)
                    np.testing.assert_array_equal(reconstruct(z["timestamps"], z["ground_truth_positions"], z["starts"], pred), z["predicted_positions"])
                    e = pred-truth
                    np.testing.assert_array_equal(np.cumsum(e*z["prediction_spacing"][:, None], axis=0), z["cumulative_velocity_residual_m"])
                    with np.load(out / "original_single" / split / path.name, allow_pickle=False) as ref:
                        for key in ("starts", "target_velocity", "timestamps", "ground_truth_positions", "prediction_spacing"):
                            np.testing.assert_array_equal(z[key], ref[key])
                    with np.load(out / (variant.rsplit("_", 1)[0]+"_c4") / split / f"{r['sequence']}_aligned_branches.npz", allow_pickle=False) as branches:
                        aligned = branches["aligned_velocity"].astype(np.float64)
                        expected = aligned[:, 0] if variant.endswith("single") else aligned.mean(axis=1)
                        np.testing.assert_array_equal(pred, expected)
                    sq += float(np.sum(e**2)); absolute += float(np.sum(abs(e))); count += len(e); recomputed.append(got)
            actual = summarize(split, variant, recomputed, sq, absolute, count)
            ref = next(r for r in summaries if r["split"] == split and r["variant"] == variant)
            for key in ("MSE", *COMPARE):
                np.testing.assert_allclose(actual[key], ref[key], rtol=1e-12, atol=1e-12)
            verification.append(dict(split=split, variant=variant, sequences=len(selected), windows=count,
                                     metrics_recomputed=True, trajectories_reconstructed=True,
                                     ensemble_recomputed_from_four_branches=True, labels_and_indices_identical=True))
    write_csv(out / "comparison/all_metrics.csv", summaries)
    write_csv(out / "comparison/test_metrics.csv", [r for r in summaries if r["split"] != "val"])
    write_csv(out / "comparison/sequence_metrics.csv", rows)
    write_csv(out / "comparison/rotation_consistency.csv", consistency)
    for variant in VARIANTS:
        write_csv(out / variant / "metrics.csv", [r for r in summaries if r["variant"] == variant])
    effects = []; paired = []
    for split in ("val", "seen_test", "unseen_test"):
        for bn_name in SOURCES:
            a = next(r for r in summaries if r["split"] == split and r["variant"] == f"{bn_name}_single")
            b = next(r for r in summaries if r["split"] == split and r["variant"] == f"{bn_name}_c4")
            effects.append(dict(split=split, bn_state=bn_name,
                **{key+"_error_reduction_percent": 100*(a[key]-b[key])/a[key] for key in COMPARE}))
            ra = {r["sequence"]: r for r in rows if r["split"] == split and r["variant"] == a["variant"]}
            rb = {r["sequence"]: r for r in rows if r["split"] == split and r["variant"] == b["variant"]}
            record = dict(split=split, bn_state=bn_name, sequences=len(ra), metrics={})
            for key in MACRO:
                av = np.array([ra[n][key] for n in ra]); bv = np.array([rb[n][key] for n in ra])
                record["metrics"][key] = dict(improved=int((bv<av).sum()), worsened=int((bv>av).sum()),
                    median_single=float(np.median(av)), median_c4=float(np.median(bv)),
                    p90_single=float(np.percentile(av, 90)), p90_c4=float(np.percentile(bv, 90)))
            paired.append(record)
    write_csv(out / "comparison/c4_effect.csv", effects)
    dump(out / "comparison/paired_sequence_analysis.json", paired)
    plot_summary(out / "comparison/test_summary.png", summaries, ("seen_test", "unseen_test"))
    plot_summary(out / "comparison/validation_summary.png", summaries, ("val",))
    primary = {s: {mode: next(r for r in summaries if r["split"] == s and r["variant"] == f"original_{mode}")
                   for mode in ("single", "c4")} for s in ("seen_test", "unseen_test")}
    criteria = dict(both_test_ATE_lower=all(primary[s]["c4"]["ATE_official_m"] < primary[s]["single"]["ATE_official_m"] for s in primary),
                    unseen_endpoint_not_worse=primary["unseen_test"]["c4"]["endpoint_error_m"] <= primary["unseen_test"]["single"]["endpoint_error_m"],
                    unseen_bias_not_worse=primary["unseen_test"]["c4"]["mean_velocity_bias_norm_mps"] <= primary["unseen_test"]["single"]["mean_velocity_bias_norm_mps"],
                    unseen_RMSE_not_worse=primary["unseen_test"]["c4"]["RMSE"] <= primary["unseen_test"]["single"]["RMSE"])
    dump(out / "summary.json", dict(status="completed", metrics=summaries, primary_overall_improvement=all(criteria.values()),
                                   primary_criteria=criteria, inference_benchmark=read_json(out / "inference_benchmark.json"),
                                   wall_seconds=time.perf_counter()-started))
    dump(out / "result_verification.json", verification)
    dump(out / "historical_baseline_verification.json", baseline_checks)
    for name, model in models.items():
        assert state_sha(model) == states[name]
        assert tensor_hash(model.named_parameters()) == provenance[name]["learned_parameters_sha256"]
    changes = [rel for rel, digest in before.items() if sha(ROOT / rel) != digest]
    assert not changes
    from PIL import Image
    for p in out.rglob("*.png"):
        with Image.open(p) as img:
            img.verify()
    dump(out / "final_audit.json", dict(status="passed", frozen_model_and_BN_states_unchanged=True,
         all_original_files_unchanged=len(before), changed_original_files=changes,
         full_sequences_per_variant=dict(val=13, seen_test=35, unseen_test=32),
         full_windows_per_variant={s: next(r["windows"] for r in summaries if r["split"] == s) for s in ("val", "seen_test", "unseen_test")},
         historical_test_predictions_exactly_reproduced=67, all_480_predictions_reopened=True,
         primary_criteria_fixed_before_evaluation=True, no_training_or_test_adaptation=True,
         plot_files_verified=len(list(out.rglob("*.png")))))
    dump(out / "progress.json", dict(status="completed"))
    logging.info("COMPLETED criteria=%s wall_seconds=%.1f", json.dumps(criteria), time.perf_counter()-started)


if __name__ == "__main__":
    main()
