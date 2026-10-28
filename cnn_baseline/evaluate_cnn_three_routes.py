"""Evaluate a newly trained fixed-protocol CNN against saved GRU/Attention runs."""
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
from cnn_baseline.model_cnn1d import CNN1D
from cnn_baseline.compare_with_resnet import trajectory_metrics, predict_windows, write_csv
from scripts.run_ronin_npz_baseline import load_sequence, reconstruct
import matplotlib.pyplot as plt

NAMES = {"cnn": "CNN", "cnn_gru": "CNN-GRU", "cnn_gru_attention": "CNN-GRU-Attention"}
COLORS = {"cnn": "#d88425", "cnn_gru": "#3279b7", "cnn_gru_attention": "#a14ca0"}
SPLITS = ("seen_test", "unseen_test")
METRICS = ("MSE", "RMSE", "MAE", "ATE_official_m", "RTE_60s_official_m")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def plot_summaries(out, roots, summaries):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, title in zip(axes, ("RMSE", "ATE_official_m", "RTE_60s_official_m"),
                               ("Velocity RMSE [m/s]", "Official ATE [m]", "Official 60s RTE [m]")):
        for i, name in enumerate(NAMES):
            values = [next(r[metric] for r in summaries if r["split"] == s and r["model"] == name) for s in SPLITS]
            bars = ax.bar(np.arange(2)+(i-1)*.25, values, .25, label=NAMES[name], color=COLORS[name])
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_xticks(np.arange(2), SPLITS); ax.set_title(title)
        ax.grid(axis="y", alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "comparison/test_summary.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for name in NAMES:
        log = read_csv(roots[name] / name / "training_log.csv")
        for ax, metric in zip(axes, ("train_mse", "val_mse")):
            ax.plot([int(r["epoch"]) for r in log], [float(r[metric]) for r in log], color=COLORS[name], label=NAMES[name])
            ax.set(xlabel="Epoch", ylabel=metric+" [(m/s)^2]"); ax.grid(alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out / "comparison/training_comparison.png", dpi=160); plt.close(fig)


def write_report(out, roots, training, summaries, rows):
    changes = []
    for before, after in (("cnn", "cnn_gru"), ("cnn_gru", "cnn_gru_attention")):
        for split in SPLITS:
            a = next(r for r in summaries if r["split"] == split and r["model"] == before)
            b = next(r for r in summaries if r["split"] == split and r["model"] == after)
            changes.append(dict(split=split, before=before, after=after,
                                **{metric: 100*(a[metric]-b[metric])/a[metric] for metric in METRICS}))
    dump(out / "comparison/module_changes_percent.json", dict(
        definition="100*(before_error-after_error)/before_error; positive means error reduction", rows=changes))
    train_table = "\n".join(f"| {NAMES[name]} | {r['parameter_count']:,} | {r['epochs_completed']} | {r['best_epoch']} | {r['best_val_mse']:.8f} |" for name, r in training.items())
    metric_table = "\n".join(f"| {r['split']} | {NAMES[r['model']]} | {r['RMSE']:.6f} | {r['MAE']:.6f} | {r['ATE_official_m']:.3f} | {r['RTE_60s_official_m']:.3f} |" for r in summaries)
    change_table = "\n".join(f"| {r['split']} | {NAMES[r['before']]} → {NAMES[r['after']]} | {r['RMSE']:+.2f}% | {r['ATE_official_m']:+.2f}% | {r['RTE_60s_official_m']:+.2f}% |" for r in changes)
    report = f"""# CNN / CNN-GRU / CNN-GRU-Attention：固定协议三路线对照

本次将原有纯 CNN 结构从头训练 30 轮，CNN-GRU 和 Attention 复用已完成的同协议实验及预测。旧实验目录未改写，旧纯 CNN 结果未混入本表。

共同配置：窗口 200、步长 10、batch 256、Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0，MSE，seed=42；无增强、标准化、学习率调度、早停或梯度裁剪。CUDA FP32，关闭 TF32，开启确定性算法。三者数据文件、72/13/35/32 序列划分、训练/验证窗口索引、窗口平均 XY 速度标签和每轮排列均已核对一致。训练集 787,920 个有效窗口，验证集 136,686 个窗口；测试集不筛选异常速度。

| 模型 | 参数量 | 完成轮数 | 最佳轮次 | 最佳验证 MSE |
|---|---:|---:|---:|---:|
{train_table}

| 测试集 | 模型 | 速度 RMSE (m/s) | MAE (m/s) | 官方 ATE (m) | 官方 60s RTE (m) |
|---|---|---:|---:|---:|---:|
{metric_table}

## 逐步结构变化对应的误差变化

下表正数表示误差降低，负数表示变差；没有预设逐步改善。

| 测试集 | 比较 | RMSE 降低比例 | ATE 降低比例 | RTE 降低比例 |
|---|---|---:|---:|---:|
{change_table}

## 结构与解释边界

三者均使用三层卷积（32/64/128 通道，核 7/5/3）。CNN 采用全局平均池化和 128→64→2 回归头；CNN-GRU 使用 hidden_size=64 的单层 GRU 最终状态和 64→64→2 回归头。因此 CNN→CNN-GRU 同时改变汇总方式、参数量和回归头输入维度，是结构对照，不能把差异全部归因于孤立插入一个 GRU。CNN-GRU→Attention 保留骨干与回归头，改为对 50 个 GRU 状态进行加性时间注意力汇总，新增 2,112 个参数。相同 seed 下相同形状的共享 CNN 初始参数已逐元素验证一致；Attention 与 GRU 的共享层初始参数也一致。

三个模型都按最低验证 MSE 选权重，未用测试集选模。速度误差按全部窗口及 XY 两分量平均；ATE/RTE 使用同一官方坐标误差定义，逐序列计算后等权平均；60s 对应 12,000 帧。轨迹采用同一 reconstruct()，只以首个参考位置初始化，不进行轨迹旋转拟合。已保存预测重新读取复算指标，两个已完成模型的原指标也逐条复核。

这是单个种子下的共同预算结构比较，不能证明统计显著性、推理效率或模块必然有效。时间成本不等同于参数量；未专门进行推理延迟测试。主要参考 RoNIN ResNet18 可在之前固定协议报告中单独展示。

## 结果目录

- cnn/：本次纯 CNN 的 best/last 权重、配置、日志、训练曲线、35 条 seen 和 32 条 unseen 轨迹图及预测 NPZ。
- comparison/：三模型共同轨迹图、训练曲线、指标总览、逐序列 CSV 和结构变化百分比。
- index.html：可选择序列的交互式总览；result_verification.json、acceptance_checks.json、final_audit.json 为复核记录。
- CNN-GRU 来源：`{roots['cnn_gru']}`。
- Attention 来源：`{roots['cnn_gru_attention']}`，该目录保留所有注意力权重与权重图，本次不重复训练这两个模型。
"""
    (out / "README_实验结果.md").write_text(report, encoding="utf-8")
    table = "".join(f"<tr><td>{r['split']}</td><td>{NAMES[r['model']]}</td><td>{r['RMSE']:.5f}</td><td>{r['MAE']:.5f}</td><td>{r['ATE_official_m']:.3f}</td><td>{r['RTE_60s_official_m']:.3f}</td></tr>" for r in summaries)
    sequences = [dict(split=r["split"], sequence=r["sequence"]) for r in rows if r["model"] == "cnn"]
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CNN 三路线固定协议实验</title><style>body{font:16px/1.65 system-ui,sans-serif;background:#f4f6f8;color:#182735;max-width:1250px;margin:30px auto;padding:0 20px}section{background:white;border-radius:10px;padding:22px;margin:18px 0}img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,td:nth-child(2){text-align:left}select,button{font:inherit;padding:7px;margin:4px}a{color:#17639a}</style>
<h1>CNN → CNN-GRU → CNN-GRU-Attention</h1><p>同数据、同标签、同顺序、30 轮、seed=42，按验证 MSE 选模。三个模型的 67 条测试序列均已复核。本次补跑纯 CNN，复用前两组的固定协议结果。</p>
<section><h2>测试结果</h2><table><tr><th>测试集</th><th>模型</th><th>速度 RMSE</th><th>速度 MAE</th><th>官方 ATE</th><th>官方 60s RTE</th></tr>__TABLE__</table>
<p>速度单位 m/s，轨迹单位 m；单种子结果。<a href="README_实验结果.md">实验说明与变化比例</a> · <a href="comparison/test_metrics.csv">指标 CSV</a> · <a href="protocol.json">训练协议</a> · <a href="result_verification.json">复核记录</a></p></section>
<section><h2>训练与测试对比</h2><img src="comparison/training_comparison.png"><img src="comparison/test_summary.png"></section>
<section><h2>逐序列轨迹、速度与误差</h2><select id="split"><option>seen_test</option><option>unseen_test</option></select><select id="seq"></select><button id="prev">上一条</button><button id="next">下一条</button><img id="comparison"><p><a id="prediction">本次 CNN 预测 NPZ</a></p></section>
<script>const data=__DATA__,s=document.getElementById('split'),q=document.getElementById('seq');function render(){document.getElementById('comparison').src='comparison/'+s.value+'/'+q.value+'.png';document.getElementById('prediction').href='cnn/'+s.value+'/'+q.value+'_predictions.npz';}function update(){q.replaceChildren(...data.filter(r=>r.split===s.value).map(r=>{let o=document.createElement('option');o.value=o.textContent=r.sequence;return o;}));render();}s.onchange=update;q.onchange=render;for(const [id,d] of [['prev',-1],['next',1]])document.getElementById(id).onclick=()=>{q.selectedIndex=(q.selectedIndex+d+q.length)%q.length;render();};update();</script></html>"""
    (out / "index.html").write_text(page.replace("__TABLE__", table).replace("__DATA__", json.dumps(sequences)), encoding="utf-8")


def validate_training(root, name, digest, orders, manifest):
    training = read_json(root / name / "summary.json")
    assert training["status"] in ("training_completed", "completed") and training["epochs_completed"] == 30
    assert training["protocol_sha256"] == digest and training["checkpoint_reload_passed"] is True
    log = read_csv(root / name / "training_log.csv")
    assert len(log) == 30 and int(min(log, key=lambda r: float(r["val_mse"]))["epoch"]) == training["best_epoch"]
    for i, row in enumerate(log, 1):
        assert int(row["epoch"]) == i and row["order_sha256"] == orders[str(i)]
        assert int(row["train_samples"]) == manifest["train"]["windows"]
        assert int(row["val_samples"]) == manifest["val"]["windows"]
    return training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--attention-run", type=Path, default=ROOT / "results/cnn_gru_attention_fixed_20261026")
    args = parser.parse_args()
    out = args.run.resolve()
    reference = Path(read_json(out / "reference.json")["reference_run"])
    attention = args.attention_run.resolve()
    protocol, digest, orders = load_reference(reference)
    expected_protocol = dict(protocol=protocol, protocol_sha256=digest)
    assert read_json(out / "protocol.json") == expected_protocol
    assert read_json(attention / "protocol.json") == expected_protocol
    assert read_json(attention / "summary.json")["status"] == "completed"
    attention_checks = read_json(attention / "acceptance_checks.json")
    for key in ("same_protocol_as_reference", "data_files_match_reference", "splits_match_reference",
                "window_indices_and_labels_match_reference", "same_order_every_epoch", "saved_metrics_recomputed", "evaluation_completed"):
        assert attention_checks[key] is True, key
    roots = {"cnn": out, "cnn_gru": reference, "cnn_gru_attention": attention}
    manifest = read_json(reference / "dataset_manifest.json")
    for root in (out, attention):
        actual = read_json(root / "dataset_manifest.json")
        for split in ("train", "val"):
            for key in ("windows", "index_sha256", "target_sha256"):
                assert actual[split][key] == manifest[split][key]
    training = {name: validate_training(root, name, digest, orders, manifest) for name, root in roots.items()}
    for p in (out / "comparison", out / "cnn/seen_test", out / "cnn/unseen_test", out / "evaluation_summary.json"):
        if p.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation: {p}")
    source_hashes = read_json(reference / "source_fingerprints.json")
    for relative in ("scripts/run_ronin_npz_baseline.py", "ronin-master/source/metric.py", "cnn_baseline/compare_with_resnet.py"):
        assert sha(ROOT / relative) == source_hashes[str(Path(relative))]
    for name, relative in (("cnn", "cnn_baseline/model_cnn1d.py"), ("cnn_gru", "cnn_baseline/model_cnn_gru.py"),
                           ("cnn_gru_attention", "cnn_baseline/model_cnn_gru_attention.py")):
        hashes = read_json(roots[name] / "source_fingerprints.json")
        assert sha(ROOT / relative) == hashes[str(Path(relative))]
    data_root = ROOT / "RoNIN_Team_Delivery/dataset"
    fingerprints = read_csv(reference / "data_fingerprints.csv")
    assert read_csv(out / "data_fingerprints.csv") == fingerprints
    assert read_csv(attention / "data_fingerprints.csv") == fingerprints
    for row in fingerprints:
        path = data_root / row["path"]
        assert path.stat().st_size == int(row["bytes"]) and sha(path) == row["sha256"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for fixed evaluation")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    checkpoint_path = out / "cnn/checkpoint_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
    validate_checkpoint(checkpoint, "cnn", 200, 10)
    assert checkpoint["protocol_sha256"] == digest and checkpoint["epoch"] == training["cnn"]["best_epoch"]
    model = CNN1D().cuda().eval(); model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    (out / "comparison").mkdir()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(out / "evaluation.log", encoding="utf-8")])
    for path in (Path(__file__), ROOT / "scripts/run_ronin_npz_baseline.py", ROOT / "ronin-master/source/metric.py"):
        shutil.copy2(path, out / "source_snapshot" / path.name)
    provenance = dict(protocol_sha256=digest, reference_runs={k: str(v) for k, v in roots.items()},
        cnn_checkpoint_sha256=sha(checkpoint_path), cnn_checkpoint_epoch=checkpoint["epoch"],
        script_sha256=sha(Path(__file__)), shared_evaluation_sources_match_reference=True,
        data_files_match_reference=True, evaluation_batch_size=512, reference_prediction_sha256={})
    old_sequence, old_summary = {}, {}
    for name in ("cnn_gru", "cnn_gru_attention"):
        old_sequence.update({(r["model"], r["split"], r["sequence"]): r for r in read_csv(roots[name] / name / "sequence_metrics.csv")})
        old_summary.update({(r["model"], r["split"]): r for r in read_csv(roots[name] / name / "test_metrics.csv")})
    rows, summaries, verification = [], [], {}
    for split in SPLITS:
        folder = out / "cnn" / split; folder.mkdir()
        comparison = out / "comparison" / split; comparison.mkdir()
        filenames = [r["path"] for r in fingerprints if r["split"] == split]
        assert len(filenames) == (35 if split == "seen_test" else 32)
        totals = {name: [0., 0., 0] for name in NAMES}
        for no, filename in enumerate(filenames, 1):
            f, p, t, _, y = load_sequence(data_root / filename)
            starts = np.arange(0, len(y), 10, dtype=np.int64)
            truth = y[starts].astype(np.float64); gt = p[:, :2]; stem = Path(filename).stem
            pred = predict_windows(model, f, starts, torch.device("cuda"), 512)
            traj = reconstruct(t, p, starts, pred)
            predictions = {"cnn": pred}; trajectories = {"cnn": traj}
            for name in ("cnn_gru", "cnn_gru_attention"):
                path = roots[name] / name / split / f"{stem}_predictions.npz"
                provenance["reference_prediction_sha256"][str(path)] = sha(path)
                with np.load(path, allow_pickle=False) as saved:
                    for key, expected in dict(starts=starts, timestamps=t, target_velocity=truth,
                            ground_truth_positions=gt, prediction_timestamps=t[starts], window_end_timestamps=t[starts+200]).items():
                        np.testing.assert_array_equal(saved[key], expected)
                    predictions[name] = saved["predicted_velocity"].copy()
                    trajectories[name] = saved["predicted_positions"].copy()
                np.testing.assert_array_equal(reconstruct(t, p, starts, predictions[name]), trajectories[name])
            for name in NAMES:
                metrics = trajectory_metrics(predictions[name], truth, trajectories[name], gt)
                if name != "cnn":
                    old = old_sequence[name, split, stem]
                    for key, value in metrics.items():
                        np.testing.assert_allclose(value, float(old[key]), rtol=1e-12, atol=1e-12)
                rows.append(dict(split=split, sequence=stem, model=name, **metrics))
                error = predictions[name]-truth
                totals[name][0] += float(np.sum(error**2)); totals[name][1] += float(np.sum(abs(error))); totals[name][2] += len(truth)
            np.savez_compressed(folder / f"{stem}_predictions.npz", starts=starts,
                prediction_timestamps=t[starts], window_end_timestamps=t[starts+200],
                predicted_velocity=pred, target_velocity=truth, timestamps=t,
                predicted_positions=traj, ground_truth_positions=gt)
            plot_sequence(folder / f"{stem}.png", split, stem, t, starts, gt, truth, {"cnn": pred}, {"cnn": traj})
            plot_sequence(comparison / f"{stem}.png", split, stem, t, starts, gt, truth, predictions, trajectories)
            cnn_metrics = rows[-3]
            logging.info("TEST %s %d/%d %s CNN_RMSE=%.6f ATE=%.3f RTE=%.3f", split, no, len(filenames), stem,
                         cnn_metrics["RMSE"], cnn_metrics["ATE_official_m"], cnn_metrics["RTE_60s_official_m"])
            dump(out / "progress.json", dict(stage="evaluation", split=split, sequence=no, sequences=len(filenames)))
        for name, (sq, absolute, n) in totals.items():
            selected = [r for r in rows if r["model"] == name and r["split"] == split]
            r = dict(split=split, model=name, sequences=len(filenames), windows=n,
                MSE=sq/(2*n), RMSE=math.sqrt(sq/(2*n)), MAE=absolute/(2*n),
                ATE_official_m=float(np.mean([s["ATE_official_m"] for s in selected])),
                RTE_60s_official_m=float(np.mean([s["RTE_60s_official_m"] for s in selected])))
            if name != "cnn":
                for key in METRICS:
                    np.testing.assert_allclose(r[key], float(old_summary[name, split][key]), rtol=1e-12, atol=1e-12)
            summaries.append(r)
        # Re-open persisted CNN predictions and independently recompute the summary.
        sq = absolute = 0.; count = 0; ates = []; rtes = []
        for row in [r for r in rows if r["model"] == "cnn" and r["split"] == split]:
            with np.load(folder / f"{row['sequence']}_predictions.npz", allow_pickle=False) as saved:
                assert all(np.isfinite(saved[k]).all() for k in saved.files)
                actual = trajectory_metrics(saved["predicted_velocity"], saved["target_velocity"], saved["predicted_positions"], saved["ground_truth_positions"])
                for key, value in actual.items():
                    np.testing.assert_allclose(value, row[key], rtol=1e-12, atol=1e-12)
                e = saved["predicted_velocity"]-saved["target_velocity"]
                sq += float(np.sum(e**2)); absolute += float(np.sum(abs(e))); count += len(e)
                ates.append(actual["ATE_official_m"]); rtes.append(actual["RTE_60s_official_m"])
        computed = dict(MSE=sq/(2*count), RMSE=math.sqrt(sq/(2*count)), MAE=absolute/(2*count),
                        ATE_official_m=float(np.mean(ates)), RTE_60s_official_m=float(np.mean(rtes)))
        expected = next(r for r in summaries if r["model"] == "cnn" and r["split"] == split)
        for key, value in computed.items():
            np.testing.assert_allclose(value, expected[key], rtol=1e-12, atol=1e-12)
        verification[split] = dict(sequences_checked=len(filenames), windows=count, saved_cnn_metrics_match=True,
            reference_sequence_and_summary_metrics_recomputed=True, labels_times_indices_gt_identical_between_models=True,
            reference_trajectory_reconstruction_identical=True)
    write_csv(out / "comparison/test_metrics.csv", summaries)
    write_csv(out / "comparison/sequence_metrics.csv", rows)
    write_csv(out / "cnn/test_metrics.csv", [r for r in summaries if r["model"] == "cnn"])
    write_csv(out / "cnn/sequence_metrics.csv", [r for r in rows if r["model"] == "cnn"])
    plot_summaries(out, roots, summaries)
    write_report(out, roots, training, summaries, rows)
    dump(out / "evaluation_config.json", provenance)
    dump(out / "result_verification.json", verification)
    dump(out / "evaluation_summary.json", dict(status="completed", protocol_sha256=digest, test_metrics=summaries))
    training["cnn"].update(status="completed", test_metrics=[r for r in summaries if r["model"] == "cnn"])
    dump(out / "cnn/summary.json", training["cnn"])
    dump(out / "summary.json", dict(status="completed", protocol_sha256=digest, training=training, test_metrics=summaries))
    checks = read_json(out / "acceptance_checks.json")
    checks.update(evaluation_completed=True, all_three_complete_30_epochs=True, all_three_same_order_every_epoch=True,
                  saved_metrics_recomputed=True, test_sequences=67, reference_predictions_reused=True)
    dump(out / "acceptance_checks.json", checks)
    dump(out / "progress.json", dict(stage="completed"))
    logging.info("COMPLETED %s", json.dumps(summaries))


if __name__ == "__main__":
    main()
