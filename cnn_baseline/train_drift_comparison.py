"""Paired CNN-GRU training: consecutive-window MSE vs residual-bias penalty.

Both conditions are new runs, selected by minimum full-validation macro ATE.
The original shuffled-window/MSE-selection experiments are historical context.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import hashlib
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn_baseline.dataset import RoNINNPZDataset, LABEL_PROTOCOL
from cnn_baseline.model_cnn_gru import CNNGRU
from cnn_baseline.sequence_dataset import ConsecutiveWindows, validation_sequences
from cnn_baseline.losses import velocity_bias_loss
from cnn_baseline.run_fixed_comparison import PROTOCOL as PREVIOUS_PROTOCOL, FixedWindows, setup_seed, dump, sha
from cnn_baseline.compare_with_resnet import trajectory_metrics, write_csv
from scripts.run_ronin_npz_baseline import reconstruct
import matplotlib.pyplot as plt

CONDITIONS = {"mse_control": 0., "bias_constraint": 1.}
PROTOCOL = dict(PREVIOUS_PROTOCOL, experiment="consecutive_window_residual_bias_v1",
    segment_length=64, segments_per_batch=4, horizons=[16, 32, 64],
    loss="velocity_MSE + lambda * mean_squared_time_normalized_prefix_residual",
    condition_weights=CONDITIONS,
    data_order="CPU randperm over identical disjoint complete segments, seed=42+epoch-1",
    incomplete_segments="drop incomplete tails per contiguous run; no padding or gap bridging",
    prediction_dt="timestamps[start+10]-timestamps[start], not window duration",
    selection="minimum full-validation sequence-macro official ATE; first epoch wins ties",
    bias_metric="sequence-macro norm of time-weighted mean velocity residual",
    interpretation="overlapping window-mean residual penalty, not exact displacement supervision")


def state_sha(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def setup_runtime():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the paired experiment")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Unexpected CUBLAS_WORKSPACE_CONFIG")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def bias_metrics(prediction, target, timestamps, starts):
    delta = timestamps[starts+10]-timestamps[starts]
    error = prediction.astype(np.float64)-target.astype(np.float64)
    mean = np.sum(error*delta[:, None], axis=0)/np.sum(delta)
    return dict(mean_velocity_error_x_mps=float(mean[0]), mean_velocity_error_y_mps=float(mean[1]),
                mean_velocity_bias_norm_mps=float(np.linalg.norm(mean)))


def score_predictions(prediction, target, metadata):
    error = prediction-target
    mse = float(np.mean(error**2))
    records = []
    for seq in metadata:
        rows = seq["rows"]; pred = prediction[rows]; truth = target[rows]
        positions = reconstruct(seq["timestamps"], seq["positions"], seq["starts"], pred)
        records.append(dict(sequence=seq["sequence"],
            **trajectory_metrics(pred, truth, positions, seq["positions"]),
            **bias_metrics(pred, truth, seq["timestamps"], seq["starts"])))
    summary = dict(MSE=mse, RMSE=math.sqrt(mse), MAE=float(np.mean(abs(error))), windows=len(error), sequences=len(records))
    for key in ("ATE_official_m", "RTE_60s_official_m", "endpoint_error_m", "mean_velocity_bias_norm_mps"):
        summary[key] = float(np.mean([r[key] for r in records]))
    if not all(math.isfinite(v) for v in summary.values()):
        raise RuntimeError("Nonfinite validation metrics")
    return summary, records


def validate(model, pool, metadata):
    model.eval(); predictions = []
    with torch.inference_mode():
        for x, _ in pool.batches(torch.arange(pool.n), 256):
            predictions.append(model(x).cpu().numpy())
    pred = np.concatenate(predictions).astype(np.float64)
    target = pool.targets.cpu().numpy().astype(np.float64)
    return score_predictions(pred, target, metadata)


def plot_history(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [r["epoch"] for r in history]
    for key, label in (("train_mse", "Velocity MSE"), ("train_bias_mse", "Bias penalty"), ("train_total_loss", "Total loss")):
        axes[0].plot(epochs, [r[key] for r in history], label=label)
    axes[0].set(xlabel="Epoch", ylabel="Online training loss [(m/s)^2]"); axes[0].legend()
    axes[1].plot(epochs, [r["val_ATE_official_m"] for r in history], label="Validation ATE")
    axes[1].set(xlabel="Epoch", ylabel="Official ATE [m]"); axes[1].legend()
    axes[2].plot(epochs, [r["val_mse"] for r in history], label="Validation velocity MSE")
    axes[2].set(xlabel="Epoch", ylabel="Velocity MSE [(m/s)^2]"); axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def train_condition(name, weight, train, val, metadata, out, protocol, digest, order_manifest, initialization):
    folder = out/name; folder.mkdir()
    setup_seed(protocol["seed"])
    model = CNNGRU().cuda()
    init_hash = state_sha(model)
    if initialization:
        assert initialization["state_sha256"] == init_hash
    else:
        initialization.update(state_sha256=init_hash, parameters=sum(p.numel() for p in model.parameters()),
                              architecture="unchanged CNNGRU; no attention or new inference parameters")
        dump(out/"initialization.json", initialization)
    config = dict(condition=name, lambda_bias=weight, model="cnn_gru", protocol=protocol,
                  protocol_sha256=digest, parameter_count=initialization["parameters"], initialization_sha256=init_hash)
    dump(folder/"config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0., amsgrad=False)
    history = []; best = float("inf"); best_epoch = 0
    logging.info("START %s lambda=%.1f segments=%d windows=%d", name, weight, train.n, train.n*64)
    with (folder/"training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = None
        for epoch in range(1, protocol["epochs"]+1):
            tic = time.perf_counter()
            order = torch.randperm(train.n, generator=torch.Generator().manual_seed(42+epoch-1))
            order_hash = hashlib.sha256(order.numpy().tobytes()).hexdigest()
            if str(epoch) in order_manifest:
                assert order_manifest[str(epoch)] == order_hash, "Condition sample order differs"
            else:
                order_manifest[str(epoch)] = order_hash
            totals = np.zeros(3); count = 0
            model.train()
            for batch, (x, y, delta) in enumerate(train.batches(order, 4), 1):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(x).reshape_as(y)
                loss, velocity, bias = velocity_bias_loss(prediction, y, delta, weight=weight)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite loss")
                loss.backward(); optimizer.step()
                totals += np.array([velocity.item(), bias.item(), loss.item()])*len(y)
                count += len(y)
                if batch % 500 == 0:
                    logging.info("%s epoch=%d batch=%d/%d mse=%.6f bias=%.6f elapsed=%.1fs", name, epoch,
                        batch, math.ceil(train.n/4), totals[0]/count, totals[1]/count, time.perf_counter()-tic)
            assert count == train.n
            metrics, sequences = validate(model, val, metadata)
            row = dict(epoch=epoch, train_mse=float(totals[0]/count), train_bias_mse=float(totals[1]/count),
                train_total_loss=float(totals[2]/count), val_mse=metrics["MSE"],
                val_ATE_official_m=metrics["ATE_official_m"], val_RTE_60s_official_m=metrics["RTE_60s_official_m"],
                val_endpoint_error_m=metrics["endpoint_error_m"], val_bias_norm_mps=metrics["mean_velocity_bias_norm_mps"],
                train_segments=count, train_windows=count*64, val_windows=val.n,
                learning_rate=1e-4, seconds=time.perf_counter()-tic, order_sha256=order_hash)
            history.append(row)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row)); writer.writeheader()
            writer.writerow(row); handle.flush()
            write_csv(folder/"validation_sequence_metrics_last.csv", sequences)
            improved = metrics["ATE_official_m"] < best
            if improved:
                best, best_epoch = metrics["ATE_official_m"], epoch
            saved = dict(model_name="cnn_gru", condition=name, lambda_bias=weight,
                model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                epoch=epoch, best_epoch=best_epoch, best_validation_ATE=best,
                label_protocol=LABEL_PROTOCOL, window_size=200, step_size=10,
                config=config, protocol_sha256=digest, metrics=row,
                torch_rng_state=torch.get_rng_state(), cuda_rng_state=torch.cuda.get_rng_state_all())
            torch.save(saved, folder/"checkpoint_last.pt")
            if improved:
                torch.save(saved, folder/"checkpoint_best.pt")
                write_csv(folder/"validation_sequence_metrics_best.csv", sequences)
            plot_history(history, folder/"training_curves.png")
            dump(out/"data_order_manifest.json", order_manifest)
            dump(out/"progress.json", dict(stage="training", condition=name, epoch=epoch,
                total_epochs=protocol["epochs"], best_epoch=best_epoch, best_validation_ATE=best))
            logging.info("EPOCH_DONE %s %s best_epoch=%d", name, json.dumps(row), best_epoch)
    saved = torch.load(folder/"checkpoint_best.pt", map_location="cuda", weights_only=True)
    fresh = CNNGRU().cuda().eval(); fresh.load_state_dict(saved["model_state_dict"], strict=True)
    model.load_state_dict(saved["model_state_dict"]); model.eval()
    x, _ = next(val.batches(torch.arange(val.n), 4))
    with torch.inference_mode():
        torch.testing.assert_close(fresh(x), model(x), rtol=0, atol=0)
    final_metrics, _ = validate(fresh, val, metadata)
    np.testing.assert_allclose(final_metrics["ATE_official_m"], best, atol=1e-12, rtol=1e-12)
    summary = dict(status="training_completed", condition=name, lambda_bias=weight, epochs_completed=len(history),
        best_epoch=best_epoch, best_validation_ATE=best, best_validation_metrics=final_metrics,
        parameter_count=initialization["parameters"], initialization_sha256=init_hash,
        protocol_sha256=digest, checkpoint_reload_passed=True, saved_validation_ATE_recomputed=True,
        training_wall_seconds=sum(r["seconds"] for r in history))
    dump(folder/"summary.json", summary)
    logging.info("TRAINING_COMPLETED %s %s", name, json.dumps(summary))
    del model, fresh, optimizer
    torch.cuda.empty_cache()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    setup_runtime()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.StreamHandler(sys.stdout), logging.FileHandler(out/"run.log", encoding="utf-8")])
    protocol = dict(PROTOCOL, smoke_only=args.smoke)
    if args.smoke:
        protocol.update(epochs=2, smoke_train_segment_limit=16)
    digest = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    dump(out/"protocol.json", dict(protocol=protocol, protocol_sha256=digest))
    dump(out/"environment.json", dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__),
        numpy=np.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"]))
    snapshot = out/"source_snapshot"; snapshot.mkdir()
    files = [*Path(__file__).parent.glob("*.py"), ROOT/"scripts/run_ronin_npz_baseline.py",
             ROOT/"ronin-master/source/metric.py", ROOT/"ronin-master/source/model_resnet1d.py",
             *list((ROOT/"RoNIN_Team_Delivery/splits").glob("*"))]
    dump(out/"source_fingerprints.json", {str(p.relative_to(ROOT)): sha(p) for p in files})
    for p in files:
        shutil.copy2(p, snapshot/p.name)
    reference = ROOT/"results/fixed_resnet18_vs_cnn_gru_20261025"
    fingerprints = list(csv.DictReader((reference/"data_fingerprints.csv").open(encoding="utf-8")))
    data_root = ROOT/"RoNIN_Team_Delivery/dataset"
    for row in fingerprints:
        p = data_root/row["path"]
        assert p.stat().st_size == int(row["bytes"]) and sha(p) == row["sha256"], f"Data changed: {p}"
    write_csv(out/"data_fingerprints.csv", fingerprints)
    baseline_sources = json.loads((reference/"source_fingerprints.json").read_text(encoding="utf-8"))
    split_root = ROOT/"RoNIN_Team_Delivery/splits"
    for path in split_root.glob("*"):
        assert sha(path) == baseline_sources[str(path.relative_to(ROOT))]
    if args.smoke:
        split_root = out/"smoke_splits"; split_root.mkdir()
        for split in ("train", "val"):
            names = [r["path"] for r in fingerprints if r["split"] == split][:2]
            (split_root/f"{split}.txt").write_text("\n".join(names)+"\n", encoding="utf-8")
    train_ds = RoNINNPZDataset(data_root, split_root/"train.txt", filter_outliers=True)
    val_ds = RoNINNPZDataset(data_root, split_root/"val.txt", mode="val")
    assert not set(train_ds.files)&set(val_ds.files)
    train = ConsecutiveWindows(train_ds, torch.device("cuda"), 64)
    if args.smoke:
        train.blocks = train.blocks[:16]; train.n = len(train.blocks)
        train.manifest.update(smoke_segment_limit=16, actual_smoke_segments=train.n)
    val = FixedWindows(val_ds, torch.device("cuda"))
    metadata = validation_sequences(val_ds)
    manifest = dict(train=train.manifest,
        validation=dict(windows=val.n, index_sha256=val.index_sha256, target_sha256=val.target_sha256,
                        sequences=[r["sequence"] for r in metadata]),
        filtered_train_windows=train_ds.filtered_count, data_files_match_previous_experiments=True)
    dump(out/"dataset_manifest.json", manifest)
    logging.info("DATA_READY %s", json.dumps(manifest))
    del train_ds, val_ds
    training = {}; orders = {}; initialization = {}
    for name, weight in CONDITIONS.items():
        training[name] = train_condition(name, weight, train, val, metadata, out, protocol, digest, orders, initialization)
    logs = {name: list(csv.DictReader((out/name/"training_log.csv").open(encoding="utf-8"))) for name in CONDITIONS}
    assert [r["order_sha256"] for r in logs["mse_control"]] == [r["order_sha256"] for r in logs["bias_constraint"]]
    assert all(len(log) == protocol["epochs"] for log in logs.values())
    checks = dict(status="passed", smoke_only=args.smoke, same_initialization=True, same_segments_and_orders=True,
        same_optimization_steps=True, same_validation_ATE_selection=True, both_checkpoints_reloaded=True,
        saved_validation_ATE_recomputed=True, data_and_splits_match_previous=True, evaluation_completed=False)
    dump(out/"acceptance_checks.json", checks)
    dump(out/"summary.json", dict(status="smoke_training_completed" if args.smoke else "training_completed",
        protocol_sha256=digest, training=training))
    dump(out/"progress.json", dict(stage="training_completed", evaluation_completed=False))
    logging.info("BOTH_TRAINING_COMPLETED; test evaluation has not run")


if __name__ == "__main__":
    main()
