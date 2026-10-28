"""Controlled ResNet18 vs CNN-GRU training and trajectory evaluation."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import hashlib
import html
import json
import logging
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn_baseline.dataset import RoNINNPZDataset, LABEL_PROTOCOL, LABEL_FORMULA
from cnn_baseline.model_cnn_gru import CNNGRU
from cnn_baseline.common import plot_training
from cnn_baseline.compare_with_resnet import trajectory_metrics, predict_windows, write_csv
from scripts.run_ronin_npz_baseline import make_model, load_sequence, reconstruct
import matplotlib.pyplot as plt

FACTORIES = {"resnet18": make_model, "cnn_gru": CNNGRU}
DISPLAY = {"resnet18": "RoNIN ResNet18", "cnn_gru": "CNN-GRU"}
PROTOCOL = dict(window_size=200, step_size=10, batch_size=256, epochs=30,
                learning_rate=1e-4, optimizer="Adam", betas=[0.9, 0.999], eps=1e-8,
                weight_decay=0., amsgrad=False, loss="MSE", seed=42,
                data_order="CPU randperm seeded with 42+epoch-1, independent of model RNG",
                augmentation=False, standardization=False, scheduler=None,
                early_stopping=False, gradient_clipping=None,
                train_filter="instantaneous speed >10 m/s anywhere in [start,end)",
                val_test_filter=False, precision="float32", tf32=False,
                cudnn_benchmark=False, cudnn_deterministic=True,
                deterministic_algorithms=True, cublas_workspace_config=":4096:8",
                label_protocol=LABEL_PROTOCOL, label_formula=LABEL_FORMULA,
                selection="minimum validation MSE, first epoch wins ties",
                evaluation_batch_size=512,
                reconstruction="same official-style reconstruct() for both models",
                trajectory_metrics="official coordinate-wise ATE and 60s RTE, sequence-macro mean")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class FixedWindows:
    """Exact CPU Dataset indices and labels; batch gather on a shared GPU pool."""
    def __init__(self, dataset, device, limit=None):
        self.details = []
        features, labels, starts = [], [], []
        offset = 0
        indices = np.asarray(dataset.index_map, dtype=np.int64)
        if limit:
            indices = indices[:limit]
        for seq_id, (f, p, t) in enumerate(dataset.sequences):
            local = indices[indices[:, 0] == seq_id, 1]
            if not len(local):
                continue
            ends = local + 200
            y = ((p[ends, :2] - p[local, :2]) / (t[ends] - t[local])[:, None]).astype(np.float32)
            features.append(f); labels.append(y); starts.append(local + offset)
            self.details.append(dict(sequence=str(dataset.files[seq_id]), frames=len(f), windows=len(local)))
            offset += len(f)
        self.features = torch.from_numpy(np.concatenate(features)).to(device)
        self.targets = torch.from_numpy(np.concatenate(labels)).to(device)
        self.starts = torch.from_numpy(np.concatenate(starts)).to(device)
        self.offsets = torch.arange(200, device=device)
        self.n = len(self.starts); self.device = device
        self.index_sha256 = hashlib.sha256(indices.tobytes()).hexdigest()
        self.target_sha256 = hashlib.sha256(np.concatenate(labels).tobytes()).hexdigest()
        # Numerical acceptance against the same CPU loader used by the models.
        for i in (0, min(123, self.n - 1), self.n - 1):
            x, y, _, _ = dataset[i]
            got_x = self.features[self.starts[i] + self.offsets].T
            np.testing.assert_array_equal(got_x.cpu().numpy(), x.numpy())
            np.testing.assert_array_equal(self.targets[i].cpu().numpy(), y.numpy())

    def batches(self, order, batch_size):
        for begin in range(0, self.n, batch_size):
            ids = order[begin:begin + batch_size].to(self.device)
            positions = self.starts[ids, None] + self.offsets
            yield self.features[positions].transpose(1, 2).contiguous(), self.targets[ids]


def setup_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train_one(name, train, val, out, protocol, common_hash, order_checks, factory=None):
    """Shared training logic; factory supports an additional ablation model.

    An existing order_checks manifest enforces the same baseline permutation.
    If no manifest exists, the first model records the canonical order.
    """
    folder = out / name; folder.mkdir()
    setup_seed(protocol["seed"])
    factory = factory or FACTORIES[name]
    model = factory().cuda()
    config = dict(model=name, protocol=protocol, protocol_sha256=common_hash,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  architecture_note="Original architecture retained; ResNet FC dropout=0.5, CNN-GRU variants have no dropout.")
    if hasattr(model, "model_config"):
        config["model_config"] = model.model_config()
    dump(folder / "config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"],
                                 betas=tuple(protocol["betas"]), eps=protocol["eps"],
                                 weight_decay=0., amsgrad=False)
    best = float("inf"); best_epoch = 0; history = []
    logging.info("START %s parameters=%d train_windows=%d val_windows=%d",
                 name, config["parameter_count"], train.n, val.n)
    with (folder / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["epoch", "train_mse", "val_mse", "train_samples", "val_samples", "seconds", "learning_rate", "order_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for epoch in range(1, protocol["epochs"] + 1):
            tic = time.perf_counter()
            generator = torch.Generator().manual_seed(protocol["seed"] + epoch - 1)
            order = torch.randperm(train.n, generator=generator)
            digest = hashlib.sha256(order.numpy().tobytes()).hexdigest()
            if str(epoch) not in order_checks:
                order_checks[str(epoch)] = digest
            elif order_checks[str(epoch)] != digest:
                raise RuntimeError("Sample order differs between model runs")
            model.train(); total = 0.; count = 0
            for batch, (x, y) in enumerate(train.batches(order, protocol["batch_size"]), 1):
                optimizer.zero_grad(set_to_none=True)
                pred = model(x)
                loss = torch.nn.functional.mse_loss(pred, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite training loss {name} epoch={epoch}")
                loss.backward(); optimizer.step()
                total += loss.item() * len(x); count += len(x)
                if batch % 500 == 0:
                    logging.info("%s epoch=%d batch=%d/%d train_mse=%.6f elapsed=%.1fs",
                                 name, epoch, batch, math.ceil(train.n / protocol["batch_size"]),
                                 total / count, time.perf_counter() - tic)
            model.eval(); val_total = 0.; val_count = 0
            with torch.inference_mode():
                for x, y in val.batches(torch.arange(val.n), protocol["batch_size"]):
                    loss = torch.nn.functional.mse_loss(model(x), y)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite validation loss")
                    val_total += loss.item() * len(x); val_count += len(x)
            assert count == train.n and val_count == val.n
            row = dict(epoch=epoch, train_mse=total/count, val_mse=val_total/val_count,
                       train_samples=count, val_samples=val_count, seconds=time.perf_counter()-tic,
                       learning_rate=protocol["learning_rate"], order_sha256=digest)
            history.append(row); writer.writerow(row); handle.flush()
            improved = row["val_mse"] < best
            if improved:
                best, best_epoch = row["val_mse"], epoch
            ckpt = dict(model_name=name, model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(), epoch=epoch,
                        label_protocol=LABEL_PROTOCOL, window_size=200, step_size=10,
                        protocol_sha256=common_hash, config=config, metrics=row,
                        best_epoch=best_epoch, best_val_mse=best,
                        torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all())
            torch.save(ckpt, folder / "checkpoint_last.pt")
            if improved:
                torch.save(ckpt, folder / "checkpoint_best.pt")
            plot_training(history, folder)
            dump(out / "data_order_manifest.json", order_checks)
            dump(out / "progress.json", dict(stage="training", model=name, epoch=epoch,
                                             total_epochs=protocol["epochs"], best_epoch=best_epoch, best_val_mse=best))
            logging.info("EPOCH_DONE %s %s best_epoch=%d", name, json.dumps(row), best_epoch)
    saved = torch.load(folder / "checkpoint_best.pt", map_location="cuda", weights_only=True)
    fresh = factory().cuda().eval()
    fresh.load_state_dict(saved["model_state_dict"], strict=True)
    model.load_state_dict(saved["model_state_dict"]); model.eval()
    x, _ = next(val.batches(torch.arange(val.n), 4))
    with torch.inference_mode():
        torch.testing.assert_close(fresh(x), model(x), rtol=0, atol=0)
    summary = dict(status="training_completed", model=name, epochs_completed=len(history),
                   best_epoch=best_epoch, best_val_mse=best,
                   parameter_count=config["parameter_count"], protocol_sha256=common_hash,
                   checkpoint_reload_passed=True, training_wall_seconds=sum(r["seconds"] for r in history))
    dump(folder / "summary.json", summary)
    logging.info("TRAINING_COMPLETED %s %s", name, json.dumps(summary))
    del model, fresh, optimizer
    torch.cuda.empty_cache()
    return summary


def plot_sequence(path, split, stem, t, starts, gt, truth, predictions, trajectories, show):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(gt[:, 0], gt[:, 1], color="black", lw=1.2, label="Ground truth")
    colors = {"resnet18": "#df8b32", "cnn_gru": "#3478b8"}
    for name in show:
        p = trajectories[name]
        ax.plot(p[:, 0], p[:, 1], color=colors[name], lw=1.1, label=DISPLAY[name])
    ax.scatter(*gt[0], c="green", s=20, label="Start")
    ax.set(title=f"{split}/{stem}", xlabel="X [m]", ylabel="Y [m]")
    ax.axis("equal"); ax.legend()
    for c, ax in ((0, axes[0, 1]), (1, axes[1, 0])):
        ax.plot(t[starts]-t[0], truth[:, c], color="black", alpha=.65, lw=.7, label="Window-mean target")
        for name in show:
            ax.plot(t[starts]-t[0], predictions[name][:, c], color=colors[name], alpha=.85, lw=.7, label=DISPLAY[name])
        ax.set(xlabel="Time [s]", ylabel=("Vx" if c == 0 else "Vy")+" [m/s]"); ax.legend()
    ax = axes[1, 1]
    for name in show:
        ax.plot(t-t[0], np.linalg.norm(trajectories[name]-gt, axis=1), color=colors[name], label=DISPLAY[name])
    ax.set(xlabel="Time [s]", ylabel="Euclidean position error [m]"); ax.legend()
    for ax in axes.flat:
        ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def evaluate_both(out, data_root, names, training, common_hash):
    models = {}
    for name, factory in FACTORIES.items():
        saved = torch.load(out/name/"checkpoint_best.pt", map_location="cuda", weights_only=True)
        assert saved["protocol_sha256"] == common_hash
        models[name] = factory().cuda().eval()
        models[name].load_state_dict(saved["model_state_dict"], strict=True)
    comparison = out/"comparison"; comparison.mkdir()
    rows = []; summaries = []; verification = {}
    for split in ("seen_test", "unseen_test"):
        for directory in [comparison/split, *[out/name/split for name in FACTORIES]]:
            directory.mkdir(parents=True)
        totals = {name: [0., 0., 0] for name in (*FACTORIES, "zero_velocity")}
        for no, file_name in enumerate(names[split], 1):
            f, p, t, _, y = load_sequence(data_root/file_name)
            starts = np.arange(0, len(y), 10, dtype=np.int64)
            truth = y[starts].astype(np.float64); gt = p[:, :2]; stem = Path(file_name).stem
            predictions = {}; trajectories = {}
            for name, model in models.items():
                pred = predict_windows(model, f, starts, torch.device("cuda"), 512)
                trajectory = reconstruct(t, p, starts, pred)
                predictions[name] = pred; trajectories[name] = trajectory
                row = dict(split=split, sequence=stem, model=name, **trajectory_metrics(pred, truth, trajectory, gt))
                rows.append(row)
                error = pred-truth
                total = totals[name]
                total[0] += float(np.sum(error**2)); total[1] += float(np.sum(abs(error))); total[2] += len(truth)
                np.savez_compressed(out/name/split/f"{stem}_predictions.npz",
                                    starts=starts, prediction_timestamps=t[starts], window_end_timestamps=t[starts+200],
                                    predicted_velocity=pred, target_velocity=truth, timestamps=t,
                                    predicted_positions=trajectory, ground_truth_positions=gt)
                plot_sequence(out/name/split/f"{stem}.png", split, stem, t, starts, gt, truth,
                              predictions, trajectories, [name])
                logging.info("TEST %s %s %d/%d %s MSE=%.6f ATE=%.3f RTE=%.3f",
                             name, split, no, len(names[split]), stem, row["MSE"],
                             row["ATE_official_m"], row["RTE_60s_official_m"])
            plot_sequence(comparison/split/f"{stem}.png", split, stem, t, starts, gt, truth,
                          predictions, trajectories, list(FACTORIES))
            total = totals["zero_velocity"]
            total[0] += float(np.sum(truth**2)); total[1] += float(np.sum(abs(truth))); total[2] += len(truth)
        for name, (sq, absolute, n) in totals.items():
            selected = [r for r in rows if r["split"] == split and r["model"] == name]
            summaries.append(dict(split=split, model=name, sequences=len(names[split]), windows=n,
                                  MSE=sq/(2*n), RMSE=math.sqrt(sq/(2*n)), MAE=absolute/(2*n),
                                  ATE_official_m=float(np.mean([r["ATE_official_m"] for r in selected])) if selected else None,
                                  RTE_60s_official_m=float(np.mean([r["RTE_60s_official_m"] for r in selected])) if selected else None))
    # Verify persisted predictions against metrics and identical labels for both models.
    for name in FACTORIES:
        write_csv(out/name/"sequence_metrics.csv", [r for r in rows if r["model"] == name])
        write_csv(out/name/"test_metrics.csv", [r for r in summaries if r["model"] in (name, "zero_velocity")])
        for split in ("seen_test", "unseen_test"):
            sq = absolute = 0.; count = 0; ates = []; rtes = []
            for r in [r for r in rows if r["model"] == name and r["split"] == split]:
                path = out/name/split/f"{r['sequence']}_predictions.npz"
                with np.load(path) as d, np.load(out/"resnet18"/split/path.name) as ref:
                    for key in ("starts", "timestamps", "target_velocity", "ground_truth_positions"):
                        np.testing.assert_array_equal(d[key], ref[key])
                    assert all(np.isfinite(d[k]).all() for k in d.files)
                    actual = trajectory_metrics(d["predicted_velocity"], d["target_velocity"],
                                                d["predicted_positions"], d["ground_truth_positions"])
                    for key, value in actual.items():
                        np.testing.assert_allclose(value, r[key], rtol=1e-12)
                    e = d["predicted_velocity"]-d["target_velocity"]
                    sq += np.sum(e**2); absolute += np.sum(abs(e)); count += len(e)
                    ates.append(actual["ATE_official_m"]); rtes.append(actual["RTE_60s_official_m"])
            summary = next(r for r in summaries if r["model"] == name and r["split"] == split)
            for key, value in dict(MSE=sq/(2*count), MAE=absolute/(2*count),
                                   ATE_official_m=np.mean(ates), RTE_60s_official_m=np.mean(rtes)).items():
                np.testing.assert_allclose(value, summary[key], rtol=1e-12)
            verification[f"{name}/{split}"] = dict(sequences_checked=len(ates), windows=count, saved_metrics_match=True,
                                                    labels_times_indices_gt_identical_between_models=True)
        training[name].update(status="completed", test_metrics=[r for r in summaries if r["model"] == name])
        dump(out/name/"summary.json", training[name])
    write_csv(comparison/"test_metrics.csv", summaries)
    write_csv(comparison/"sequence_metrics.csv", rows)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, ("RMSE", "ATE_official_m", "RTE_60s_official_m")):
        for i, name in enumerate(FACTORIES):
            values = [next(r[metric] for r in summaries if r["split"] == s and r["model"] == name)
                      for s in ("seen_test", "unseen_test")]
            bars = ax.bar(np.arange(2)+(i-.5)*.35, values, .35, label=DISPLAY[name])
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_xticks(np.arange(2), ("seen_test", "unseen_test"))
        ax.set(title=metric, ylabel="m/s" if metric == "RMSE" else "m"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(comparison/"test_summary.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in FACTORIES:
        records = list(csv.DictReader((out/name/"training_log.csv").open(encoding="utf-8")))
        for ax, metric in zip(axes, ("train_mse", "val_mse")):
            ax.plot([int(r["epoch"]) for r in records], [float(r[metric]) for r in records], label=DISPLAY[name])
            ax.set(xlabel="Epoch", ylabel=metric+" [(m/s)^2]"); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(comparison/"training_comparison.png", dpi=160); plt.close(fig)
    dump(out/"result_verification.json", verification)
    return rows, summaries


def finish_report(out, protocol, training, rows, summaries, smoke):
    table = "\n".join(f"| {r['split']} | {DISPLAY[r['model']]} | {r['RMSE']:.6f} | {r['MAE']:.6f} | {r['ATE_official_m']:.3f} | {r['RTE_60s_official_m']:.3f} |"
                      for r in summaries if r["model"] in FACTORIES)
    text = f"""# 固定训练参数：RoNIN ResNet18 与 CNN-GRU

{'本目录为小规模流程检查，不是正式实验。' if smoke else '两模型均从头完成 30 轮训练。'}

共同设置：窗口 200、步长 10、batch 256、Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0、MSE、seed=42；不使用增强、标准化、调度、早停或梯度裁剪。float32，TF32 关闭，确定性算法开启。每轮窗口顺序 SHA256 已逐轮核对一致。相同 epoch 数意味着相同样本访问与优化步数，不代表相同计算量。

监督目标为窗口平均 XY 速度。训练过滤瞬时速度 >10 m/s 的位移区间，验证和测试不筛选。采用交付的 72/13/35/32 划分。两模型共用数据缓存和批处理，仅网络结构不同。ResNet 保留官方网络的 FC Dropout=0.5；CNN-GRU 没有 Dropout，这属于保留的结构差异。

| 模型 | 参数量 | 完成轮数 | 最佳轮次 | 最佳验证 MSE |
|---|---:|---:|---:|---:|
""" + "\n".join(f"| {DISPLAY[name]} | {r['parameter_count']:,} | {r['epochs_completed']} | {r['best_epoch']} | {r['best_val_mse']:.8f} |"
               for name, r in training.items()) + f"""

| 测试集 | 模型 | 速度 RMSE (m/s) | MAE (m/s) | 官方 ATE (m) | 官方 60s RTE (m) |
|---|---|---:|---:|---:|---:|
{table}

两者均使用验证 MSE 最小的权重，未用测试集选模。速度指标对窗口和两个分量平均；ATE/RTE 沿用官方坐标误差定义，先逐序列计算再等权平均。轨迹沿用同一 reconstruct 函数，以首位置初始化，不做旋转拟合。保存预测已独立复算指标，并核对两模型的索引、时间、标签和参考轨迹一致。

每个模型目录包含配置、日志、源码引用、best/last 权重、训练曲线、汇总及逐序列指标、seen_test/、unseen_test/。comparison/ 包含共同轨迹图、训练曲线对比、测试汇总；index.html 为总览。protocol.json、dataset_manifest.json、data_order_manifest.json、data_fingerprints.csv、source_snapshot/、result_verification.json 保存复核证据。

这是单个种子在共同协议下的对照，不是论文原始条件复现，也不能独立证明结果具有统计显著性。旧实验未覆盖。
"""
    (out/"README_实验结果.md").write_text(text, encoding="utf-8")
    table_html = "".join(f"<tr><td>{r['split']}</td><td>{DISPLAY[r['model']]}</td><td>{r['RMSE']:.4f}</td><td>{r['ATE_official_m']:.3f}</td><td>{r['RTE_60s_official_m']:.3f}</td></tr>"
                         for r in summaries if r["model"] in FACTORIES)
    data = [{"split":r["split"],"sequence":r["sequence"]} for r in rows if r["model"]=="resnet18"]
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>固定协议：ResNet18 与 CNN-GRU</title><style>
body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:30px auto;background:#f4f6f8;color:#182735;padding:0 20px}
section{padding:22px;background:white;border-radius:10px;margin:18px 0}img{width:100%;height:auto}
table{width:100%;border-collapse:collapse}td,th{padding:9px;text-align:right;border-bottom:1px solid #dde3e8}td:first-child,td:nth-child(2){text-align:left}
select,button{font:inherit;padding:7px;margin:4px}a{color:#17639a}</style>
<h1>固定协议：RoNIN ResNet18 与 CNN-GRU</h1><p>__STATUS__</p>
<section><h2>测试结果</h2><table><tr><th>测试集</th><th>模型</th><th>速度 RMSE</th><th>官方 ATE</th><th>官方 RTE</th></tr>__TABLE__</table>
<p>速度单位 m/s，轨迹单位 m。共同训练预算与标签协议，保留原网络结构差异。</p>
<a href="README_实验结果.md">实验说明</a> · <a href="protocol.json">固定配置</a> · <a href="comparison/test_metrics.csv">指标 CSV</a> · <a href="result_verification.json">复核记录</a></section>
<section><h2>训练与测试汇总</h2><img src="comparison/training_comparison.png" alt="共同协议训练曲线"><img src="comparison/test_summary.png" alt="测试指标"></section>
<section><h2>逐序列三方对比</h2><select id="split"><option>seen_test</option><option>unseen_test</option></select><select id="seq"></select>
<button id="prev">上一条</button><button id="next">下一条</button><img id="chart" alt="参考轨迹、ResNet18、CNN-GRU">
<p><a id="resnet">ResNet18 预测</a> · <a id="gru">CNN-GRU 预测</a></p></section>
<script>const data=__DATA__,s=document.getElementById('split'),q=document.getElementById('seq');
function render(){document.getElementById('chart').src='comparison/'+s.value+'/'+q.value+'.png';
document.getElementById('resnet').href='resnet18/'+s.value+'/'+q.value+'_predictions.npz';
document.getElementById('gru').href='cnn_gru/'+s.value+'/'+q.value+'_predictions.npz';}
function update(){q.replaceChildren(...data.filter(r=>r.split===s.value).map(r=>{let o=document.createElement('option');o.value=o.textContent=r.sequence;return o;}));render();}
s.onchange=update;q.onchange=render;for(const [id,d] of [['prev',-1],['next',1]])document.getElementById(id).onclick=()=>{q.selectedIndex=(q.selectedIndex+d+q.length)%q.length;render();};update();</script></html>"""
    page=page.replace("__TABLE__",table_html).replace("__DATA__",json.dumps(data))
    page=page.replace("__STATUS__","小规模流程检查，非正式实验。" if smoke else "两模型均完整训练30轮，按验证集最低MSE选模；全部67条测试序列已复核。")
    (out/"index.html").write_text(page,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this experiment")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Expected CUBLAS_WORKSPACE_CONFIG=:4096:8")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),logging.FileHandler(out/"run.log",encoding="utf-8")])
    protocol=dict(PROTOCOL)
    if args.smoke:
        protocol.update(epochs=2,smoke_only=True,smoke_window_limit=384)
    else:
        protocol["smoke_only"]=False
    common_hash=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    dump(out/"protocol.json",dict(protocol=protocol,protocol_sha256=common_hash))
    dump(out/"environment.json",dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),
                                    numpy=np.__version__,gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
                                    cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"]))
    snapshot=out/"source_snapshot";snapshot.mkdir()
    files=[*Path(__file__).parent.glob("*.py"),ROOT/"scripts/run_ronin_npz_baseline.py",
           ROOT/"ronin-master/source/model_resnet1d.py",ROOT/"ronin-master/source/metric.py",
           *list((ROOT/"RoNIN_Team_Delivery/splits").glob("*"))]
    dump(out/"source_fingerprints.json",{str(p.relative_to(ROOT)):sha(p) for p in files})
    for p in files:
        shutil.copy2(p,snapshot/p.name)
    split_root=ROOT/"RoNIN_Team_Delivery/splits"
    names={k:[s.strip() for s in (split_root/f"{k}.txt").read_text().splitlines() if s.strip()]
           for k in ("train","val","seen_test","unseen_test")}
    data_root=ROOT/"RoNIN_Team_Delivery/dataset"
    if args.smoke:
        split_root=out/"smoke_splits";split_root.mkdir()
        names={k:v[:2] if k in ("train","val") else v[:1] for k,v in names.items()}
        for k,v in names.items():
            (split_root/f"{k}.txt").write_text("\n".join(v)+"\n",encoding="utf-8")
    train_ds=RoNINNPZDataset(data_root,split_root/"train.txt",filter_outliers=True)
    val_ds=RoNINNPZDataset(data_root,split_root/"val.txt",mode="val")
    assert not set(train_ds.files)&set(val_ds.files)
    train=FixedWindows(train_ds,torch.device("cuda"),384 if args.smoke else None)
    val=FixedWindows(val_ds,torch.device("cuda"),384 if args.smoke else None)
    manifest={k:dict(windows=pool.n,index_sha256=pool.index_sha256,target_sha256=pool.target_sha256,
                     sequences=pool.details) for k,pool in (("train",train),("val",val))}
    manifest["filtered_train_windows"]=train_ds.filtered_count
    dump(out/"dataset_manifest.json",manifest)
    logging.info("DATA_READY train=%d val=%d protocol_sha256=%s",train.n,val.n,common_hash)
    data_rows=[]
    for split,ls in names.items():
        for name in ls:
            path=data_root/name
            data_rows.append(dict(split=split,path=name,bytes=path.stat().st_size,sha256=sha(path)))
    write_csv(out/"data_fingerprints.csv",data_rows)
    del train_ds,val_ds
    training={};orders={}
    for name in FACTORIES:
        training[name]=train_one(name,train,val,out,protocol,common_hash,orders)
    assert all(r["epochs_completed"]==protocol["epochs"] and r["protocol_sha256"]==common_hash for r in training.values())
    del train,val
    torch.cuda.empty_cache()
    rows,summaries=evaluate_both(out,data_root,names,training,common_hash)
    finish_report(out,protocol,training,rows,summaries,args.smoke)
    # Check saved order hashes across complete logs, not just the in-memory assertion.
    logs={name:list(csv.DictReader((out/name/"training_log.csv").open(encoding="utf-8"))) for name in FACTORIES}
    assert [r["order_sha256"] for r in logs["resnet18"]]==[r["order_sha256"] for r in logs["cnn_gru"]]
    for name,records in logs.items():
        assert len(records)==protocol["epochs"]
        assert all(int(r["train_samples"])==manifest["train"]["windows"] and int(r["val_samples"])==manifest["val"]["windows"] for r in records)
    checks=dict(status="passed",same_protocol=True,same_order_every_epoch=True,complete_epochs=True,
                same_sample_counts=True,both_checkpoints_reloaded=True,smoke_only=args.smoke)
    dump(out/"acceptance_checks.json",checks)
    summary=dict(status="smoke_passed" if args.smoke else "completed",protocol_sha256=common_hash,
                 training=training,test_metrics=summaries)
    dump(out/"summary.json",summary);dump(out/"progress.json",dict(stage=summary["status"]))
    logging.info("COMPLETED %s",json.dumps(summary))


if __name__=="__main__":
    main()

