"""Evaluate new mean-velocity checkpoints; reject legacy label protocols."""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
sys.path.insert(0, str(ROOT))
from cnn_baseline.dataset import RoNINNPZDataset, LABEL_PROTOCOL, validate_checkpoint
from cnn_baseline.common import MODELS, select_device
import matplotlib.pyplot as plt


def velocity_metrics(pred, target):
    if pred.shape != target.shape or target.ndim != 2 or target.shape[1] != 2 or not len(target):
        raise ValueError("Expected matching nonempty [N,2] predictions and labels")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite predictions or labels")
    error = pred.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(error ** 2))
    return dict(MSE=mse, RMSE=float(np.sqrt(mse)), MAE=float(np.mean(abs(error))), samples=len(target))


def predict(model, loader, device, return_attention=False):
    model.eval(); preds = []; targets = []; attention_weights = []
    if return_attention and not hasattr(model, "forward_with_attention"):
        raise ValueError("This model has no temporal attention output")
    with torch.inference_mode():
        for x, y, _, _ in loader:
            x = x.to(device, non_blocking=True)
            if return_attention:
                prediction, weights = model.forward_with_attention(x)
                attention_weights.append(weights.cpu().numpy())
            else:
                prediction = model(x)
            preds.append(prediction.cpu().numpy())
            targets.append(y.numpy())
    if return_attention:
        return np.concatenate(preds), np.concatenate(targets), np.concatenate(attention_weights)
    return np.concatenate(preds), np.concatenate(targets)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", nargs="+", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "RoNIN_Team_Delivery/dataset")
    parser.add_argument("--split-root", type=Path, default=ROOT / "RoNIN_Team_Delivery/splits")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--save-attention", action="store_true",
                        help="Save all attention weights and one attention plot per sequence (attention model only)")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    device = select_device(args.device); torch.set_num_threads(4)
    loaded = []; seen_names = set(); shared = None
    # Validate metadata before loading test data or creating output files.
    for path in args.checkpoint:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        name = checkpoint.get("model_name")
        if name not in MODELS:
            raise ValueError(f"Unknown model: {name!r}")
        window = checkpoint.get("window_size"); step = checkpoint.get("step_size")
        validate_checkpoint(checkpoint, name, window, step)
        if not isinstance(window, int) or window < 4 or not isinstance(step, int) or step <= 0:
            raise ValueError("Invalid window/step metadata")
        if shared is not None and shared != (window, step):
            raise ValueError("Comparison requires identical window and step sizes")
        if name in seen_names:
            raise ValueError("Use one checkpoint per model per comparison")
        seen_names.add(name); shared = (window, step)
        model = MODELS[name]().to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        loaded.append((name, model, path.resolve()))
    if args.save_attention and not any(hasattr(model, "forward_with_attention") for _, model, _ in loaded):
        parser.error("--save-attention requires a cnn_gru_attention checkpoint")
    out = (args.out or args.checkpoint[0].resolve().parent /
           f"evaluation_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(dict(
        label_protocol=LABEL_PROTOCOL, window_size=shared[0], step_size=shared[1],
        checkpoints=[str(p) for _, _, p in loaded], data_root=str(args.data_root.resolve()),
        split_root=str(args.split_root.resolve()), filter_outliers=False, device=str(device),
        save_attention=args.save_attention,
        metric="Coordinate-wise velocity errors, averaged over all windows and both XY components",
        plots="Velocity curves and metric bars; no trajectory metrics in this evaluator"),
        indent=2), encoding="utf-8")
    rows = []; per_sequence = []
    for split in ("seen_test", "unseen_test"):
        ds = RoNINNPZDataset(args.data_root, args.split_root / f"{split}.txt",
                             shared[0], shared[1], mode=split)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            pin_memory=device.type == "cuda")
        indices = np.array(ds.index_map, dtype=np.int64)
        starts = indices[:, 1]; sequence_ids = indices[:, 0]
        for model_name, model, checkpoint_path in loaded:
            weights = None
            if args.save_attention and hasattr(model, "forward_with_attention"):
                pred, target, weights = predict(model, loader, device, return_attention=True)
            else:
                pred, target = predict(model, loader, device)
            rows.append(dict(split=split, model=model_name, **velocity_metrics(pred, target)))
            if not any(r["split"] == split and r["model"] == "zero_velocity" for r in rows):
                rows.append(dict(split=split, model="zero_velocity",
                                 **velocity_metrics(np.zeros_like(target), target)))
            folder = out / split / model_name; folder.mkdir(parents=True)
            extras = {"attention_weights": weights} if weights is not None else {}
            np.savez_compressed(folder / "window_predictions.npz",
                                predicted_velocity=pred, target_velocity=target,
                                sequence_ids=sequence_ids, starts=starts,
                                sequence_files=np.array([str(p) for p in ds.files]), **extras)
            for seq_id, path in enumerate(ds.files):
                mask = sequence_ids == seq_id
                if not mask.any():
                    continue
                per_sequence.append(dict(split=split, model=model_name, sequence=path.stem,
                                         **velocity_metrics(pred[mask], target[mask])))
                times = ds.sequences[seq_id][2]
                relative_times = times[starts[mask]] - times[0]
                fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                for component, ax in enumerate(axes):
                    ax.plot(relative_times, target[mask, component], label="Window-mean target", lw=.8)
                    ax.plot(relative_times, pred[mask, component], label="Prediction", lw=.8, alpha=.85)
                    ax.set(ylabel=("Vx" if component == 0 else "Vy") + " [m/s]")
                    ax.grid(alpha=.25); ax.legend()
                axes[0].set_title(f"{split}/{path.stem} - {model_name}")
                axes[-1].set_xlabel("Window start time [s]")
                fig.tight_layout(); fig.savefig(folder / f"{path.stem}_velocity.png", dpi=130); plt.close(fig)
                if weights is not None:
                    selected = np.flatnonzero(mask)[mask.sum() // 2]
                    fig, ax = plt.subplots(figsize=(9, 3))
                    ax.bar(np.arange(weights.shape[1]), weights[selected], width=.9)
                    ax.set(xlabel="GRU timestep after CNN downsampling", ylabel="Attention weight",
                           title=f"{path.stem}: window start frame {starts[selected]}")
                    ax.grid(axis="y", alpha=.25); fig.tight_layout()
                    fig.savefig(folder / f"{path.stem}_attention.png", dpi=140); plt.close(fig)
            print(f"{split} | {model_name} | {rows[-1] if rows[-1]['model'] == model_name else rows[-2]}", flush=True)
        del ds, loader
    for filename, data in (("test_metrics.csv", rows), ("sequence_metrics.csv", per_sequence)):
        with (out / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    names = ["zero_velocity", *[name for name, _, _ in loaded]]
    width = .8 / len(names); x = np.arange(2)
    for ax, metric in zip(axes, ("MSE", "RMSE", "MAE")):
        for i, name in enumerate(names):
            values = [next(r[metric] for r in rows if r["split"] == split and r["model"] == name)
                      for split in ("seen_test", "unseen_test")]
            ax.bar(x + (i - (len(names) - 1) / 2) * width, values, width, label=name)
        ax.set_xticks(x, ("seen_test", "unseen_test"))
        ax.set(title=metric, ylabel="(m/s)^2" if metric == "MSE" else "m/s")
        ax.grid(axis="y", alpha=.25); ax.legend()
    fig.tight_layout(); fig.savefig(out / "test_comparison.png", dpi=160); plt.close(fig)
    (out / "summary.json").write_text(json.dumps(dict(status="completed", label_protocol=LABEL_PROTOCOL,
                                                      test_metrics=rows), indent=2), encoding="utf-8")
    print(f"Evaluation saved: {out}")


if __name__ == "__main__":
    main()

