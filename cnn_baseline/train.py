"""Train CNN/CNN-GRU/CNN-GRU-Attention with window-average velocity labels."""
import argparse
import csv
import hashlib
import json
import logging
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
sys.path.insert(0, str(ROOT))
from cnn_baseline.dataset import RoNINNPZDataset, LABEL_PROTOCOL, LABEL_FORMULA
from cnn_baseline.common import MODELS, select_device, plot_training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default="cnn")
    parser.add_argument("--data-root", type=Path, default=ROOT / "RoNIN_Team_Delivery/dataset")
    parser.add_argument("--split-root", type=Path, default=ROOT / "RoNIN_Team_Delivery/splits")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Run exactly --epochs complete epochs; still select best checkpoint by validation")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if min(args.epochs, args.patience, args.batch_size, args.step_size) <= 0 or args.window_size < 4:
        parser.error("epochs/patience/batch/step must be positive; window must be >=4")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be positive and finite")
    device = select_device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    out = (args.out or PACKAGE / "runs" / f"{args.model}_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "run.log", encoding="utf-8")])
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(out=str(out), device=str(device), python=sys.version,
                  executable=sys.executable, torch=str(torch.__version__), numpy=np.__version__,
                  label_protocol=LABEL_PROTOCOL, label_formula=LABEL_FORMULA,
                  optimizer="Adam", loss="MSE", train_outlier_threshold=10.0,
                  validation_filter=False, normalization=False, augmentation=False,
                  data_order_seed=args.seed,
                  notes="Shared mean-velocity labels; no rotation/jitter augmentation. Do not compare training budgets directly with ResNet.")
    snapshot = out / "source_snapshot"; snapshot.mkdir()
    hashes = {}
    for path in sorted(PACKAGE.glob("*.py")) + sorted(args.split_root.glob("*.txt")) + sorted(args.split_root.glob("*.json")):
        shutil.copy2(path, snapshot / path.name)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    config["source_and_split_sha256"] = hashes
    (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    train_set = RoNINNPZDataset(args.data_root, args.split_root / "train.txt",
                               args.window_size, args.step_size, filter_outliers=True)
    val_set = RoNINNPZDataset(args.data_root, args.split_root / "val.txt",
                             args.window_size, args.step_size, mode="val")
    train_paths = {p.resolve() for p in train_set.files}
    if train_paths.intersection(p.resolve() for p in val_set.files):
        raise ValueError("Training and validation contain the same sequence")
    manifest = {key: {"files": [str(p) for p in ds.files], "windows": len(ds),
                      "filtered_windows": ds.filtered_count}
                for key, ds in (("train", train_set), ("val", val_set))}
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    loaders = {key: DataLoader(ds, batch_size=args.batch_size, shuffle=key == "train",
                               num_workers=0, pin_memory=device.type == "cuda",
                               generator=torch.Generator().manual_seed(args.seed if key == "train" else args.seed + 1))
               for key, ds in (("train", train_set), ("val", val_set))}
    model = MODELS[args.model]().to(device)
    if hasattr(model, "model_config"):
        config["model_config"] = model.model_config()
        (out / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best = float("inf"); best_epoch = 0; stale = 0; history = []
    logging.info("model=%s parameters=%d device=%s label=%s", args.model,
                 sum(p.numel() for p in model.parameters()), device, LABEL_PROTOCOL)
    with (out / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("epoch", "train_mse", "val_mse", "train_samples", "val_samples", "seconds")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter(); losses = {}; counts = {}
            for mode in ("train", "val"):
                model.train(mode == "train"); total = 0.; count = 0
                with torch.set_grad_enabled(mode == "train"):
                    for batch, (x, y, _, _) in enumerate(loaders[mode], 1):
                        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
                        if mode == "train":
                            optimizer.zero_grad(set_to_none=True)
                        pred = model(x)
                        if pred.shape != y.shape:
                            raise RuntimeError(f"Prediction {pred.shape} != target {y.shape}")
                        loss = torch.nn.functional.mse_loss(pred, y)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Nonfinite loss: epoch={epoch} mode={mode}")
                        if mode == "train":
                            loss.backward(); optimizer.step()
                        total += loss.item() * len(x); count += len(x)
                        if mode == "train" and batch % 500 == 0:
                            logging.info("epoch=%d batch=%d/%d train_mse=%.6f",
                                         epoch, batch, len(loaders[mode]), total / count)
                losses[mode] = total / count; counts[mode] = count
            row = dict(epoch=epoch, train_mse=losses["train"], val_mse=losses["val"],
                       train_samples=counts["train"], val_samples=counts["val"],
                       seconds=time.perf_counter() - start)
            history.append(row); writer.writerow(row); handle.flush()
            improved = losses["val"] < best
            if improved:
                best = losses["val"]; best_epoch = epoch; stale = 0
            else:
                stale += 1
            checkpoint = dict(model_name=args.model, label_protocol=LABEL_PROTOCOL,
                              label_formula=LABEL_FORMULA, window_size=args.window_size,
                              step_size=args.step_size, epoch=epoch, config=config,
                              model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                              train_mse=losses["train"], val_mse=losses["val"],
                              best_epoch=best_epoch, best_val_mse=best)
            torch.save(checkpoint, out / "checkpoint_last.pt")
            if improved:
                torch.save(checkpoint, out / "checkpoint_best.pt")
            plot_training(history, out)
            logging.info("EPOCH_DONE %s best_epoch=%d", json.dumps(row), best_epoch)
            if not args.no_early_stopping and stale >= args.patience:
                logging.info("Early stopping"); break
    # Verify the persisted best model, rather than merely checking file existence.
    saved = torch.load(out / "checkpoint_best.pt", map_location=device, weights_only=True)
    restored = MODELS[args.model]().to(device)
    restored.load_state_dict(saved["model_state_dict"], strict=True)
    model.load_state_dict(saved["model_state_dict"]); model.eval(); restored.eval()
    x = next(iter(loaders["val"]))[0][:4].to(device)
    with torch.inference_mode():
        expected = model(x); actual = restored(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if not torch.isfinite(actual).all():
            raise RuntimeError("Reloaded checkpoint produced nonfinite output")
    summary = dict(status="completed", model=args.model, label_protocol=LABEL_PROTOCOL,
                   epochs_completed=len(history), best_epoch=best_epoch, best_val_mse=best,
                   checkpoint_reload_passed=True, output=str(out))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("COMPLETED %s", json.dumps(summary))


if __name__ == "__main__":
    main()

