"""Train-only BN buffer intervention on three existing CNN-GRU checkpoints.

No backward pass, optimizer, checkpoint selection, or test-set access.
Two fixed layouts use exactly the same full-size training batches.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import copy
import csv
import hashlib
import inspect
import json
import logging
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
from cnn_baseline.dataset import RoNINNPZDataset, validate_checkpoint
from cnn_baseline.model_cnn_gru import CNNGRU
from cnn_baseline.sequence_dataset import partition_segments, validation_sequences
from cnn_baseline.run_fixed_comparison import FixedWindows, dump, sha, setup_seed
from cnn_baseline.train_drift_comparison import setup_runtime, score_predictions, state_sha
from cnn_baseline.compare_with_resnet import write_csv

OLD = ROOT / "results/fixed_resnet18_vs_cnn_gru_20261025"
NEW = ROOT / "results/drift_loss_cnn_gru_20261026"
SOURCES = {
    "old_random_mse": OLD / "cnn_gru/checkpoint_best.pt",
    "consecutive_mse": NEW / "mse_control/checkpoint_best.pt",
    "consecutive_bias": NEW / "bias_constraint/checkpoint_best.pt",
}
SEED = 20261026
LAYOUTS = ("mixed_windows", "consecutive_segments")
METRICS = ("RMSE", "ATE_official_m", "RTE_60s_official_m",
           "endpoint_error_m", "mean_velocity_bias_norm_mps")


def tensor_hash(items):
    h = hashlib.sha256()
    for name, tensor in items:
        h.update(name.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def bn_summary(model):
    return {name: dict(mean=module.running_mean.detach().cpu().tolist(),
                       variance=module.running_var.detach().cpu().tolist(),
                       num_batches_tracked=int(module.num_batches_tracked),
                       momentum=module.momentum, eps=module.eps)
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.BatchNorm1d)}


def evaluate(model, pool, metadata, folder):
    model.eval()
    before = state_sha(model)
    parts = []
    with torch.inference_mode():
        for x, _ in pool.batches(torch.arange(pool.n), 256):
            assert not model.training
            assert all(not m.training for m in model.modules())
            parts.append(model(x).cpu().numpy())
    predictions = np.concatenate(parts)
    targets = pool.targets.cpu().numpy()
    metrics, records = score_predictions(predictions.astype(np.float64),
                                         targets.astype(np.float64), metadata)
    assert state_sha(model) == before, "Validation changed parameters or buffers"
    # Full predictions are retained for independent recomputation.
    np.savez_compressed(folder / "validation_predictions.npz", prediction=predictions,
                        target=targets)
    write_csv(folder / "validation_sequence_metrics.csv", records)
    dump(folder / "metrics.json", metrics)
    return metrics, records


def calibrate(model, pool, order):
    """Cumulative average of batch statistics, with equal full batches only.

    BN layers use batch statistics during calibration. Deeper-layer activation
    statistics therefore remain dependent on batch composition; this is an
    explicit sensitivity intervention, not an oracle population estimate.
    """
    model.eval()
    modules = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert len(modules) == 3
    old_momenta = [m.momentum for m in modules]
    for p in model.parameters():
        p.requires_grad_(False)
        assert p.grad is None
    for m in modules:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    count = 0
    with torch.inference_mode():
        for begin in range(0, len(order), 256):
            rows = order[begin:begin + 256].to(pool.device)
            assert len(rows) == 256
            x = pool.features[pool.starts[rows, None] + pool.offsets].transpose(1, 2).contiguous()
            assert not model.training and model.gru.training is False
            assert all(m.training for m in modules)
            model(x)
            count += 1
            if count % 1000 == 0:
                logging.info("Calibration batch %d/%d", count, len(order) // 256)
    for m, momentum in zip(modules, old_momenta):
        assert int(m.num_batches_tracked) == count
        m.momentum = momentum
    model.eval()
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    return count


def plots(rows, buffer_rows, out):
    names = list(SOURCES)
    variants = ("original", *LAYOUTS)
    labels = ("Original BN", "Mixed-window calibration", "Consecutive calibration")
    colors = ("#777777", "#2478b5", "#db8a23")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    lookup = {(r["model"], r["variant"]): r for r in rows}
    for ax, key, ylabel in zip(axes, ("ATE_official_m", "RMSE", "mean_velocity_bias_norm_mps"),
                             ("Validation ATE [m]", "Velocity RMSE [m/s]", "Mean velocity bias [m/s]")):
        for j, (variant, label, color) in enumerate(zip(variants, labels, colors)):
            values = [lookup[n, variant][key] for n in names]
            bars = ax.bar(np.arange(3) + (j - 1)*.25, values, width=.24, label=label, color=color)
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_xticks(range(3), ("Old random\nMSE selection", "Consecutive MSE\nATE selection", "Consecutive bias\nATE selection"))
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.2); ax.set_axisbelow(True)
        ax.margins(y=.18)
    axes[0].legend(fontsize=8)
    fig.suptitle("BN-only intervention | same training windows | full validation | lower is better")
    fig.tight_layout(); fig.savefig(out / "bn_validation_comparison.png", dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, names):
        for variant, color in zip(LAYOUTS, colors[1:]):
            values = [r for r in buffer_rows if r["model"] == name and r["variant"] == variant]
            ax.plot(range(3), [r["mean_shift_in_original_std_rms"] for r in values],
                    "o-", label=variant, color=color)
        ax.set_xticks(range(3), ("BN1", "BN2", "BN3")); ax.set_title(name)
        ax.set_ylabel("RMS mean shift / original running std"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "bn_buffer_changes.png", dpi=170); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    setup_runtime(); setup_seed(42)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", encoding="utf-8")])
    started = time.perf_counter()
    protocol = dict(experiment="bn_calibration_diagnosis_v1", calibration_split="train only",
                    evaluation_split="full validation only; no test access", batch_size=256,
                    seed=SEED, segment_length=64, layouts=list(LAYOUTS),
                    window_set="identical complete segments from original training pool; discard two permuted segments for full batches",
                    bn_update="reset buffers, momentum=None, one pass; only BN layers train, all learned parameters frozen",
                    selection="no checkpoint, layout or calibration hyperparameter selection; report all nine conditions",
                    validation_batch_size=256, precision="FP32", deterministic=True,
                    interpretation="inference BN buffer intervention; cannot isolate BN effects on training gradients")
    dump(out / "protocol.json", protocol)
    dump(out / "environment.json", dict(python=sys.version, executable=sys.executable,
         torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda,
         gpu=torch.cuda.get_device_name(0)))
    (out / "local_batchnorm_forward.txt").write_text(
        inspect.getsource(torch.nn.modules.batchnorm._BatchNorm.forward), encoding="utf-8")
    # Preserve and verify the actual original artifacts rather than editing them.
    watched = {p for root in (OLD, NEW) for p in root.rglob("*") if p.is_file()}
    expected_sources = json.loads((NEW / "source_fingerprints.json").read_text(encoding="utf-8"))
    for rel, digest in expected_sources.items():
        p = ROOT / rel
        assert sha(p) == digest, f"Original source/split changed: {rel}"
        watched.add(p)
    fingerprints = list(csv.DictReader((NEW / "data_fingerprints.csv").open(encoding="utf-8")))
    train_val_files = []
    for row in fingerprints:
        if row["split"] in ("train", "val"):
            p = ROOT / "RoNIN_Team_Delivery/dataset" / row["path"]
            assert p.stat().st_size == int(row["bytes"]) and sha(p) == row["sha256"]
            watched.add(p); train_val_files.append(str(p.relative_to(ROOT)))
    before = {str(p.relative_to(ROOT)): sha(p) for p in sorted(watched)}
    dump(out / "original_artifact_hashes.json", before)
    snapshot = out / "source_snapshot"; snapshot.mkdir()
    for p in [Path(__file__), *[ROOT / rel for rel in expected_sources]]:
        shutil.copy2(p, snapshot / p.name)
    dump(out / "source_fingerprints.json", {str(p.relative_to(ROOT)): sha(p)
          for p in [Path(__file__), *[ROOT / rel for rel in expected_sources]]})
    data = ROOT / "RoNIN_Team_Delivery/dataset"; splits = ROOT / "RoNIN_Team_Delivery/splits"
    train_ds = RoNINNPZDataset(data, splits / "train.txt", filter_outliers=True)
    val_ds = RoNINNPZDataset(data, splits / "val.txt", mode="val")
    assert not set(train_ds.files) & set(val_ds.files)
    train = FixedWindows(train_ds, torch.device("cuda"))
    val = FixedWindows(val_ds, torch.device("cuda"))
    meta = validation_sequences(val_ds)
    blocks, counts = partition_segments(train_ds.index_map)
    previous = json.loads((NEW / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert train.index_sha256 == previous["train"]["index_sha256"]
    assert train.target_sha256 == previous["train"]["target_sha256"]
    assert hashlib.sha256(blocks.tobytes()).hexdigest() == previous["train"]["segment_sha256"]
    assert val.index_sha256 == previous["validation"]["index_sha256"]
    assert val.target_sha256 == previous["validation"]["target_sha256"]
    # Independently seeded CPU generators; nothing depends on model state/RNG.
    perm = torch.randperm(len(blocks), generator=torch.Generator().manual_seed(SEED)).numpy()
    selected_segments = perm[:len(blocks) // 4 * 4]
    grouped = torch.from_numpy(blocks[selected_segments].reshape(-1).copy())
    mixed = grouped[torch.randperm(len(grouped), generator=torch.Generator().manual_seed(SEED))]
    orders = dict(mixed_windows=mixed, consecutive_segments=grouped)
    assert torch.equal(torch.sort(grouped).values, torch.sort(mixed).values)
    assert len(grouped) == len(torch.unique(grouped)) and len(grouped) % 256 == 0
    indices = np.asarray(train_ds.index_map, dtype=np.int64)
    assert len(np.unique(indices[grouped.numpy(), 0])) == len(train_ds.files)
    np.savez_compressed(out / "calibration_orders.npz", **{k: v.numpy() for k, v in orders.items()},
                        validation_indices=np.asarray(val_ds.index_map, dtype=np.int64))
    manifest = dict(**counts, calibration_windows=len(grouped), calibration_batches=len(grouped)//256,
                    calibration_sequences=len(train_ds.files), validation_sequences=len(meta),
                    validation_windows=val.n, dropped_additional_segments=len(blocks)-len(selected_segments),
                    same_window_set=True, orders_sha256={k: hashlib.sha256(v.numpy().tobytes()).hexdigest() for k, v in orders.items()},
                    validation_index_sha256=val.index_sha256, validation_target_sha256=val.target_sha256,
                    train_val_data_files_verified=train_val_files)
    dump(out / "dataset_manifest.json", manifest)
    del train_ds, val_ds
    logging.info("DATA_READY %s", json.dumps(manifest))
    rows = []; sequence_rows = []; buffer_rows = []; checks = []
    for name, path in SOURCES.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint(checkpoint, "cnn_gru", 200, 10)
        base = CNNGRU().cuda().eval()
        base.load_state_dict(checkpoint["model_state_dict"], strict=True)
        for p in base.parameters():
            p.requires_grad_(False)
        initial_params = tensor_hash(base.named_parameters())
        base_buffers = bn_summary(base)
        for variant in ("original", *LAYOUTS):
            tic = time.perf_counter()
            model = copy.deepcopy(base)
            model.gru.flatten_parameters()
            assert state_sha(model) == state_sha(base)
            folder = out / name / variant; folder.mkdir(parents=True)
            logging.info("START %s %s checkpoint_epoch=%d", name, variant, checkpoint["epoch"])
            original_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if variant != "original":
                batches = calibrate(model, train, orders[variant])
                assert batches == manifest["calibration_batches"]
            assert tensor_hash(model.named_parameters()) == initial_params
            changed = [k for k, v in model.state_dict().items() if not torch.equal(v, original_state[k])]
            allowed = {f"{n}.{k}" for n, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm1d)
                       for k in ("running_mean", "running_var", "num_batches_tracked")}
            assert set(changed) <= allowed
            assert (len(changed) == 0) if variant == "original" else (len(changed) == 9)
            buffers = bn_summary(model); dump(folder / "bn_buffers.json", buffers)
            for layer, current in buffers.items():
                a = base_buffers[layer]
                shift = (np.array(current["mean"]) - np.array(a["mean"])) / np.sqrt(np.array(a["variance"]) + a["eps"])
                buffer_rows.append(dict(model=name, variant=variant, layer=layer,
                    mean_shift_in_original_std_rms=float(np.sqrt(np.mean(shift**2))),
                    variance_ratio_geometric_mean=float(np.exp(np.mean(np.log(np.array(current["variance"]) / np.array(a["variance"]))))),
                    num_batches_tracked=current["num_batches_tracked"]))
            metrics, records = evaluate(model, val, meta, folder)
            if variant == "original":
                if name == "old_random_mse":
                    np.testing.assert_allclose(metrics["MSE"], checkpoint["metrics"]["val_mse"], rtol=2e-6, atol=1e-8)
                else:
                    np.testing.assert_allclose(metrics["ATE_official_m"], checkpoint["metrics"]["val_ATE_official_m"], rtol=1e-12, atol=1e-12)
            else:
                saved = dict(model_name="cnn_gru", model_state_dict=model.state_dict(),
                             label_protocol=checkpoint["label_protocol"], window_size=200, step_size=10,
                             source_checkpoint=str(path.relative_to(ROOT)), source_sha256=sha(path),
                             epoch=checkpoint["epoch"], intervention=variant, calibration_protocol=protocol,
                             learned_parameters_sha256=initial_params, validation_metrics=metrics)
                torch.save(saved, folder / "checkpoint_bn_calibrated.pt")
                reloaded = torch.load(folder / "checkpoint_bn_calibrated.pt", map_location="cuda", weights_only=True)
                fresh = CNNGRU().cuda().eval(); fresh.load_state_dict(reloaded["model_state_dict"], strict=True)
                x, _ = next(val.batches(torch.arange(val.n), 256))
                with torch.inference_mode():
                    torch.testing.assert_close(fresh(x), model(x), rtol=0, atol=0)
                assert state_sha(fresh) == state_sha(model)
                assert tensor_hash(fresh.named_parameters()) == initial_params
                del fresh, reloaded
            row = dict(model=name, variant=variant, source_epoch=checkpoint["epoch"], **metrics)
            rows.append(row)
            sequence_rows.extend(dict(model=name, variant=variant, **r) for r in records)
            checks.append(dict(model=name, variant=variant, learned_parameters_unchanged=True,
                               parameter_sha256=initial_params, changed_state_keys=changed,
                               validation_did_not_change_state=True, calibrated_checkpoint_reload_passed=variant != "original"))
            dump(out / "progress.json", dict(completed=len(rows), total=9, current=row))
            logging.info("DONE %s %s seconds=%.1f metrics=%s", name, variant, time.perf_counter()-tic, json.dumps(metrics))
            del model
        del base, checkpoint
    comparisons = []
    for name in SOURCES:
        base_row = next(r for r in rows if r["model"] == name and r["variant"] == "original")
        for variant in LAYOUTS:
            row = next(r for r in rows if r["model"] == name and r["variant"] == variant)
            comparisons.append(dict(model=name, variant=variant,
                **{k + "_error_reduction_percent": 100*(base_row[k]-row[k])/base_row[k] for k in METRICS}))
    write_csv(out / "validation_metrics.csv", rows)
    write_csv(out / "validation_sequence_metrics.csv", sequence_rows)
    write_csv(out / "bn_buffer_changes.csv", buffer_rows)
    write_csv(out / "calibration_effect.csv", comparisons)
    dump(out / "summary.json", dict(status="completed", validation_metrics=rows, effect=comparisons,
                                    no_test_evaluation=True, wall_seconds=time.perf_counter()-started))
    plots(rows, buffer_rows, out)
    # Recompute all nine evaluations from persisted predictions.
    for row in rows:
        folder = out / row["model"] / row["variant"]
        with np.load(folder / "validation_predictions.npz", allow_pickle=False) as persisted:
            assert np.isfinite(persisted["prediction"]).all()
            np.testing.assert_array_equal(persisted["target"], val.targets.cpu().numpy())
            recomputed, _ = score_predictions(persisted["prediction"].astype(np.float64),
                                              persisted["target"].astype(np.float64), meta)
        for key in METRICS:
            np.testing.assert_allclose(row[key], recomputed[key], rtol=1e-12, atol=1e-12)
    changed_files = [rel for rel, digest in before.items() if sha(ROOT / rel) != digest]
    assert not changed_files
    from PIL import Image
    for p in out.glob("*.png"):
        with Image.open(p) as image:
            image.verify()
    audit = dict(status="passed", no_backward_or_optimizer=True, no_test_set_access=True,
                 training_data_only_for_calibration=True, identical_windows_and_full_batches=True,
                 validation_used_eval_mode_and_frozen_state=True,
                 all_learned_parameters_unchanged=True, six_calibrated_checkpoints_reloaded=True,
                 nine_persisted_prediction_metrics_recomputed=True,
                 original_artifacts_verified_unchanged=len(before), changed_original_files=changed_files,
                 checks=checks)
    dump(out / "final_audit.json", audit)
    dump(out / "progress.json", dict(status="completed", completed=9, total=9))
    logging.info("COMPLETED wall_seconds=%.1f originals_unchanged=%d", time.perf_counter()-started, len(before))


if __name__ == "__main__":
    main()
