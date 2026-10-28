"""Train the unchanged official RoNIN ResNet18 on the delivered NPZ data.

No raw-HDF5 dependencies are required. This adapter uses window-average XY
velocity (official StridedSequenceDataset convention), not targets[end].
"""
import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import random
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT / "ronin-master" / "source"
sys.path.insert(0, str(OFFICIAL))
from model_resnet1d import ResNet1D, BasicBlock1D, FCOutputModule
from metric import compute_ate_rte


def make_model():
    return ResNet1D(6, 2, BasicBlock1D, [2, 2, 2, 2], base_plane=64,
                    output_block=FCOutputModule, kernel_size=3, fc_dim=512,
                    in_dim=7, dropout=0.5, trans_planes=128)


def load_sequence(path, window=200):
    with np.load(path, allow_pickle=False) as d:
        f = d["features"].astype(np.float32)
        p = d["positions"].astype(np.float64)
        t = d["timestamps"].astype(np.float64)
        instant = d["targets"].astype(np.float32)
    assert f.shape == (len(t), 6) and p.shape == (len(t), 3)
    assert instant.shape == (len(t), 2) and len(t) > window
    assert all(np.isfinite(a).all() for a in (f, p, t, instant))
    assert np.all(np.diff(t) > 0)
    y = ((p[window:, :2] - p[:-window, :2]) /
         (t[window:] - t[:-window])[:, None]).astype(np.float32)
    return f, p, t, instant, y


class GPUPool:
    """Keep sequences on device; gather independent windows in batches.

    Exactly the same indices and labels as positions-based CPU slicing.
    Jitter is drawn from [-5, 5), clamped to valid starts; unlike the old
    official Strided implementation it does not clamp starts to window_size.
    """
    def __init__(self, names, data_root, device, train, smoke=False):
        fs, ys, starts, valid_maps = [], [], [], []
        offset = 0
        self.filtered = 0
        self.details = []
        for name in names[:2] if smoke else names:
            f, p, t, instant, y = load_sequence(data_root / name)
            if smoke:
                f, p, t, instant = f[:4000], p[:4000], t[:4000], instant[:4000]
                y = ((p[200:, :2] - p[:-200, :2]) /
                     (t[200:] - t[:-200])[:, None]).astype(np.float32)
            n_valid = len(y)
            good = np.ones(n_valid, dtype=bool)
            if train:
                prefix = np.r_[0, np.cumsum(np.linalg.norm(instant, axis=1) > 10)]
                good = prefix[200:len(f)] - prefix[:len(f) - 200] == 0
            base = np.arange(0, n_valid, 10, dtype=np.int64)
            self.filtered += int((~good[base]).sum())
            base = base[good[base]]
            assert len(base)
            # Map every possible jittered start to its nearest valid start.
            # For unfiltered regions this is the identity map.
            candidates = np.flatnonzero(good)
            grid = np.arange(len(f), dtype=np.int64)
            ins = np.searchsorted(candidates, grid)
            left = candidates[np.maximum(ins - 1, 0)]
            right = candidates[np.minimum(ins, len(candidates) - 1)]
            nearest = np.where(abs(grid - left) <= abs(grid - right), left, right)
            valid_maps.append(nearest + offset)
            starts.append(base + offset)
            fs.append(f)
            ys.append(np.pad(y, ((0, len(f) - len(y)), (0, 0))))
            self.details.append({"sequence": name, "frames": len(f), "windows": len(base)})
            offset += len(f)
        self.features = torch.from_numpy(np.concatenate(fs)).to(device)
        self.targets = torch.from_numpy(np.concatenate(ys)).to(device)
        self.starts = torch.from_numpy(np.concatenate(starts)).to(device)
        self.valid_map = torch.from_numpy(np.concatenate(valid_maps)).to(device)
        self.offsets = torch.arange(200, device=device)
        self.device = device
        self.train = train
        logging.info("Loaded %s: sequences=%d windows=%d filtered=%d",
                     "train" if train else "val", len(fs), len(self.starts), self.filtered)

    def batches(self, batch_size, augment=False):
        order = (torch.randperm(len(self.starts), device=self.device) if augment
                 else torch.arange(len(self.starts), device=self.device))
        for ids in order.split(batch_size):
            starts = self.starts[ids]
            if augment:
                shifted = starts + torch.randint(-5, 5, starts.shape, device=self.device)
                shifted.clamp_(0, len(self.valid_map) - 1)
                # Bound jitter to the sequence owning the original start.
                lo = self.valid_map[starts]
                # The map at the base frame provides sequence-local valid bounds.
                mapped = self.valid_map[shifted]
                starts = torch.where(abs(mapped - lo) <= 5, mapped, lo)
            x = self.features[starts[:, None] + self.offsets]
            y = self.targets[starts].clone()
            if augment:
                angle = torch.rand(len(ids), device=self.device) * (2 * math.pi)
                c, s = angle.cos(), angle.sin()
                for a, b in ((0, 1), (3, 4)):
                    u, v = x[:, :, a].clone(), x[:, :, b].clone()
                    x[:, :, a] = c[:, None] * u - s[:, None] * v
                    x[:, :, b] = s[:, None] * u + c[:, None] * v
                u, v = y[:, 0].clone(), y[:, 1].clone()
                y[:, 0], y[:, 1] = c * u - s * v, s * u + c * v
            yield x.transpose(1, 2).contiguous(), y


def plot_training(history, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for key, label in (("train_mse", "Train (augmented, online)"), ("val_mse", "Validation")):
        ax[0].plot([r["epoch"] for r in history], [r[key] for r in history], "o-", label=label)
    ax[0].set(xlabel="Epoch", ylabel="Velocity MSE [(m/s)^2]", title="RoNIN ResNet18 training")
    ax[0].legend()
    ax[1].plot([r["epoch"] for r in history], [r["learning_rate"] for r in history], "o-")
    ax[1].set(xlabel="Epoch", ylabel="Learning rate", yscale="log")
    for a in ax:
        a.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=160)
    plt.close(fig)


def reconstruct(t, p, starts, pred):
    """Official recon_traj_with_preds formula; modern integer dtype only.

    Uses fixed mean prediction spacing, cumulative velocity, zero-order
    endpoint extension and linear interpolation, matching ronin_resnet.py.
    """
    dts = np.mean(t[starts[1:]] - t[starts[:-1]])
    pos = np.zeros((len(pred) + 2, 2), dtype=np.float64)
    pos[0] = p[0, :2]
    pos[1:-1] = np.cumsum(pred[:, :2] * dts, axis=0) + pos[0]
    pos[-1] = pos[-2]
    knots = np.r_[t[0] - 1e-6, t[starts], t[-1] + 1e-6]
    return interp1d(knots, pos, axis=0)(t)


def evaluate(model, data_root, split_names, out, device, batch_size):
    rows, summaries = [], []
    model.eval()
    for split, names in split_names.items():
        folder = out / split
        folder.mkdir()
        total_sq = total_abs = zero_sq = zero_abs = 0.
        count = 0
        for no, name in enumerate(names, 1):
            f, p, t, instant, y = load_sequence(data_root / name)
            starts = np.arange(0, len(y), 10, dtype=np.int64)
            ft = torch.from_numpy(f).to(device)
            offsets = torch.arange(200, device=device)
            predictions = []
            with torch.inference_mode():
                for chunk in np.array_split(starts, max(1, math.ceil(len(starts) / batch_size))):
                    ix = torch.from_numpy(chunk).to(device)
                    x = ft[ix[:, None] + offsets].transpose(1, 2).contiguous()
                    predictions.append(model(x).cpu().numpy())
            pred = np.concatenate(predictions).astype(np.float64)
            truth = y[starts].astype(np.float64)
            err = pred - truth
            trajectory = reconstruct(t, p, starts, pred)
            gt = p[:, :2]
            ate, rte = compute_ate_rte(trajectory, gt, pred_per_min=12000)
            row = {"split": split, "sequence": Path(name).stem, "windows": len(starts),
                   "duration_s": float(t[-1] - t[0]), "MSE": float(np.mean(err ** 2)),
                   "RMSE": float(np.sqrt(np.mean(err ** 2))), "MAE": float(np.mean(abs(err))),
                   "ATE_official_m": float(ate), "RTE_60s_official_m": float(rte),
                   "ATE_euclidean_m": float(np.sqrt(np.mean(np.sum((trajectory - gt) ** 2, axis=1)))),
                   "endpoint_error_m": float(np.linalg.norm(trajectory[-1] - gt[-1]))}
            rows.append(row)
            total_sq += float(np.sum(err ** 2)); total_abs += float(np.sum(abs(err)))
            zero_sq += float(np.sum(truth ** 2)); zero_abs += float(np.sum(abs(truth)))
            count += len(starts)
            stem = Path(name).stem
            np.savez_compressed(folder / f"{stem}_predictions.npz", starts=starts,
                                prediction_timestamps=t[starts], window_end_timestamps=t[starts + 200],
                                predicted_velocity=pred, target_velocity=truth,
                                timestamps=t, predicted_positions=trajectory, ground_truth_positions=gt)
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            axes[0, 0].plot(gt[:, 0], gt[:, 1], label="Ground truth", lw=1.4)
            axes[0, 0].plot(trajectory[:, 0], trajectory[:, 1], label="RoNIN ResNet18", lw=1.2)
            axes[0, 0].scatter(*gt[0], c="black", s=25, label="Start")
            axes[0, 0].set(xlabel="X [m]", ylabel="Y [m]", title=f"{split}/{stem}")
            axes[0, 0].axis("equal"); axes[0, 0].legend()
            for ax, component in ((axes[0, 1], 0), (axes[1, 0], 1)):
                ts = t[starts] - t[0]
                ax.plot(ts, truth[:, component], label="Target", alpha=.8, lw=.8)
                ax.plot(ts, pred[:, component], label="Prediction", alpha=.8, lw=.8)
                ax.set(xlabel="Time [s]", ylabel=f"V{'x' if component == 0 else 'y'} [m/s]")
                ax.legend()
            axes[1, 1].plot(t - t[0], np.linalg.norm(trajectory - gt, axis=1))
            axes[1, 1].set(xlabel="Time [s]", ylabel="Position error [m]",
                           title=f"Official ATE={ate:.2f}m, RTE={rte:.2f}m")
            for a in axes.flat:
                a.grid(alpha=.25)
            fig.tight_layout(); fig.savefig(folder / f"{stem}.png", dpi=130); plt.close(fig)
            logging.info("TEST %s %d/%d %s MSE=%.6f ATE=%.3f RTE=%.3f",
                         split, no, len(names), stem, row["MSE"], ate, rte)
            del ft
        selected = [r for r in rows if r["split"] == split]
        summary = {"split": split, "model": "RoNIN ResNet18", "sequences": len(names),
                   "windows": count, "MSE": total_sq / (2 * count),
                   "RMSE": math.sqrt(total_sq / (2 * count)), "MAE": total_abs / (2 * count),
                   "ATE_official_m": float(np.mean([r["ATE_official_m"] for r in selected])),
                   "RTE_60s_official_m": float(np.mean([r["RTE_60s_official_m"] for r in selected]))}
        summaries.append(summary)
        summaries.append({"split": split, "model": "Zero velocity", "sequences": len(names),
                          "windows": count, "MSE": zero_sq / (2 * count),
                          "RMSE": math.sqrt(zero_sq / (2 * count)), "MAE": zero_abs / (2 * count),
                          "ATE_official_m": None, "RTE_60s_official_m": None})
    for filename, data in (("sequence_metrics.csv", rows), ("test_metrics.csv", summaries)):
        with (out / filename).open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key in zip(axes, ("RMSE", "ATE_official_m", "RTE_60s_official_m")):
        selected = [r for r in summaries if r["model"] == "RoNIN ResNet18"]
        bars = ax.bar([r["split"] for r in selected], [r[key] for r in selected], color=["#3478b8", "#d98435"])
        ax.bar_label(bars, fmt="%.3f")
        ax.set(title=key, ylabel="m/s" if key == "RMSE" else "m"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "test_summary.png", dpi=170); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    for split in split_names:
        values = sorted(r["ATE_official_m"] for r in rows if r["split"] == split)
        ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=split)
    ax.set(xlabel="Official ATE [m]", ylabel="Fraction of sequences", title="Trajectory error distribution")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out / "ate_distribution.png", dpi=160); plt.close(fig)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", encoding="utf-8")])
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA unavailable; refusing unintended long CPU training")
    data_root = ROOT / "RoNIN_Team_Delivery" / "dataset"
    split_root = ROOT / "RoNIN_Team_Delivery" / "splits"
    lists = {k: [s.strip() for s in (split_root / f"{k}.txt").read_text().splitlines() if s.strip()]
             for k in ("train", "val", "seen_test", "unseen_test")}
    source_snapshot = out / "source_snapshot"; source_snapshot.mkdir()
    manifest = {}
    for p in [Path(__file__), OFFICIAL / "model_resnet1d.py", OFFICIAL / "metric.py",
              OFFICIAL / "data_glob_speed.py", OFFICIAL / "ronin_resnet.py", *split_root.glob("*")]:
        shutil.copy2(p, source_snapshot / p.name)
        manifest[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    config = {"architecture": "official ResNet1D resnet18", "seed": seed,
              "window_size": 200, "step_size": 10, "batch_size": args.batch_size,
              "max_epochs": args.epochs, "early_stopping_patience": args.patience,
              "optimizer": "Adam", "learning_rate": 1e-4, "loss": "MSELoss",
              "scheduler": "ReduceLROnPlateau(factor=0.1, patience=10, eps=1e-12)",
              "label": "(positions[start+200,:2]-positions[start,:2])/(timestamps[start+200]-timestamps[start])",
              "train_filter": "exclude windows containing instantaneous target speed >10 m/s",
              "validation_and_test_filter": False, "standardization": False,
              "augmentation": "random global XY rotation [0,2pi); integer start jitter [-5,5), valid/filter-safe bounds",
              "precision": "float32; TF32 disabled", "device": str(device),
              "python": sys.version, "python_executable": sys.executable,
              "torch": str(torch.__version__), "numpy": np.__version__,
              "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
              "cudnn_benchmark": True, "deterministic_algorithms": False,
              "split_counts": {k: len(v) for k, v in lists.items()},
              "smoke_only": args.smoke, "source_sha256": manifest,
              "protocol_notes": ["Uses delivered splits, not the official repository lists.",
                                 "All NPZ features were exported using game_rv, unlike official adaptive training orientation selection.",
                                 "Final raw position was truncated during NPZ export; boundary windows can differ from raw HDF5.",
                                 "Training speed filter is a project-specific addition.",
                                 "Official architecture and official ATE/RTE functions are imported unchanged.",
                                 "GPU-resident window batching replaces CPU Dataset/DataLoader.",
                                 "Official trajectory reconstruction formula retained; no SE(2) fit or heading alignment.",
                                 "Official ATE/RTE average squared coordinate errors, not Euclidean distances."]}
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("ENV torch=%s device=%s gpu=%s", torch.__version__, device, config["gpu"])
    train = GPUPool(lists["train"], data_root, device, True, args.smoke)
    val = GPUPool(lists["val"], data_root, device, False, args.smoke)
    (out / "dataset_manifest.json").write_text(json.dumps({"train": train.details, "val": val.details,
                                                           "filtered_train_windows": train.filtered}, indent=2), encoding="utf-8")
    model = make_model().to(device)
    logging.info("Official ResNet18 parameters=%d", sum(p.numel() for p in model.parameters()))
    # Check GPU adapter against direct CPU window/label calculations.
    test_x, test_y = next(train.batches(4))
    f, p, t, _, y = load_sequence(data_root / lists["train"][0])
    np.testing.assert_allclose(test_x[0].cpu().numpy(), f[:200].T, rtol=0, atol=0)
    np.testing.assert_allclose(test_y[0].cpu().numpy(), y[0], rtol=0, atol=0)
    model.eval()
    with torch.no_grad():
        assert model(test_x).shape == (4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.1, patience=10, eps=1e-12)
    best = float("inf"); best_epoch = 0; stale = 0; history = []
    start_total = time.perf_counter()
    with (out / "training_log.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=["epoch", "train_mse", "val_mse", "learning_rate", "seconds", "train_windows", "val_windows"])
        writer.writeheader()
        for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
            start = time.perf_counter(); model.train(); total = 0.; n = 0
            for batch, (x, y) in enumerate(train.batches(args.batch_size, augment=True), 1):
                optimizer.zero_grad(set_to_none=True)
                pred = model(x); loss = torch.nn.functional.mse_loss(pred, y)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch} batch {batch}")
                loss.backward(); optimizer.step()
                total += loss.item() * len(x); n += len(x)
                if batch % 100 == 0:
                    logging.info("Epoch %d batch %d/%d train_MSE=%.6f elapsed=%.1fs", epoch, batch,
                                 math.ceil(len(train.starts) / args.batch_size), total / n, time.perf_counter() - start)
                if args.smoke and batch == 10:
                    break
            model.eval(); val_total = 0.; vn = 0
            with torch.inference_mode():
                for x, y in val.batches(512):
                    loss = torch.nn.functional.mse_loss(model(x), y)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite validation loss")
                    val_total += loss.item() * len(x); vn += len(x)
            row = {"epoch": epoch, "train_mse": total / n, "val_mse": val_total / vn,
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "seconds": time.perf_counter() - start, "train_windows": n, "val_windows": vn}
            history.append(row); writer.writerow(row); h.flush()
            scheduler.step(row["val_mse"])
            improved = row["val_mse"] < best
            if improved:
                best, best_epoch, stale = row["val_mse"], epoch, 0
            else:
                stale += 1
            checkpoint = {"epoch": epoch, "model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                          "config": config, "metrics": row, "best_val_mse": best, "best_epoch": best_epoch,
                          "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all()}
            torch.save(checkpoint, out / "checkpoint_last.pt")
            if improved:
                torch.save(checkpoint, out / "checkpoint_best.pt")
            plot_training(history, out)
            logging.info("EPOCH_DONE %s best_epoch=%d", json.dumps(row), best_epoch)
            if stale >= args.patience:
                logging.info("Early stopping: %d epochs without validation improvement", stale)
                break
    del train, val
    torch.cuda.empty_cache()
    ckpt = torch.load(out / "checkpoint_best.pt", map_location=device, weights_only=True)
    reloaded = make_model().to(device)
    reloaded.load_state_dict(ckpt["model_state_dict"], strict=True)
    reloaded.eval(); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(test_x), reloaded(test_x), rtol=0, atol=0)
    logging.info("Checkpoint strict load and identical inference: PASS")
    del model
    if args.smoke:
        summary = {"status": "smoke_passed", "checkpoint_reload": True, "history": history}
    else:
        summaries = evaluate(reloaded, data_root, {k: lists[k] for k in ("seen_test", "unseen_test")}, out, device, 512)
        summary = {"status": "completed", "best_epoch": best_epoch, "best_val_mse": best,
                   "epochs_completed": len(history), "training_seconds": sum(r["seconds"] for r in history),
                   "total_seconds": time.perf_counter() - start_total, "test_metrics": summaries,
                   "checkpoint_reload": True}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("COMPLETED %s", json.dumps(summary))


if __name__ == "__main__":
    main()
