"""Train only CNN-GRU-Attention under the verified fixed baseline protocol.

No training runs on import. --check-only validates the baseline protocol and
model metadata without writing outputs or starting training. Evaluation is
available through cnn_baseline/evaluate.py --save-attention.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from cnn_baseline.dataset import RoNINNPZDataset
from cnn_baseline.model_cnn_gru_attention import CNNGRUAttention
from cnn_baseline.run_fixed_comparison import PROTOCOL, FixedWindows, train_one, dump, sha


def load_reference(reference):
    protocol_record = json.loads((reference / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((reference / "summary.json").read_text(encoding="utf-8"))
    checks = json.loads((reference / "acceptance_checks.json").read_text(encoding="utf-8"))
    expected = dict(PROTOCOL, smoke_only=False)
    if protocol_record["protocol"] != expected:
        raise ValueError("Baseline protocol differs from current fixed protocol; reconcile it explicitly")
    digest = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    if protocol_record["protocol_sha256"] != digest or summary["protocol_sha256"] != digest:
        raise ValueError("Reference protocol SHA256 mismatch")
    if summary["status"] != "completed" or checks.get("smoke_only") is not False:
        raise ValueError("Reference must be a completed full experiment, not a smoke test")
    if not all(checks.get(k) is True for k in ("same_protocol", "same_order_every_epoch", "complete_epochs", "same_sample_counts")):
        raise ValueError("Baseline acceptance checks did not pass")
    if any(summary["training"][name]["epochs_completed"] != expected["epochs"] for name in ("resnet18", "cnn_gru")):
        raise ValueError("Baseline did not complete all specified epochs")
    orders = json.loads((reference / "data_order_manifest.json").read_text(encoding="utf-8"))
    if set(orders) != {str(e) for e in range(1, expected["epochs"] + 1)}:
        raise ValueError("Reference window-order manifest is incomplete")
    return expected, digest, orders


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path,
                        default=ROOT / "results/fixed_resnet18_vs_cnn_gru_20261025")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    reference = args.reference_run.resolve()
    protocol, digest, orders = load_reference(reference)
    model = CNNGRUAttention()
    metadata = dict(model="cnn_gru_attention", parameter_count=sum(p.numel() for p in model.parameters()),
                    model_config=model.model_config(), protocol_sha256=digest,
                    reference_run=str(reference), status="configuration_checked_only",
                    note="Current data SHA256 is verified when real training starts.")
    if args.check_only:
        print(json.dumps(metadata, indent=2))
        return
    metadata.update(status="reference_protocol_validated",
                    note="Data, split, window and order checks are recorded separately during training.")
    del model
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the shared fixed-protocol trainer")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != protocol["cublas_workspace_config"]:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG differs from the frozen protocol")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = protocol["tf32"]
    torch.backends.cudnn.allow_tf32 = protocol["tf32"]
    torch.backends.cudnn.benchmark = protocol["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = protocol["cudnn_deterministic"]
    torch.use_deterministic_algorithms(protocol["deterministic_algorithms"])
    out = (args.out or ROOT / "results" / f"cnn_gru_attention_fixed_{datetime.now():%Y%m%d_%H%M%S}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "run.log", encoding="utf-8")])
    dump(out / "protocol.json", dict(protocol=protocol, protocol_sha256=digest))
    dump(out / "reference.json", metadata)
    dump(out / "environment.json", dict(python=sys.version, executable=sys.executable,
         torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    snapshot = out / "source_snapshot"; snapshot.mkdir()
    files = [*Path(__file__).parent.glob("*.py"), *list((ROOT / "RoNIN_Team_Delivery/splits").glob("*"))]
    dump(out / "source_fingerprints.json", {str(p.relative_to(ROOT)): sha(p) for p in files})
    for p in files:
        shutil.copy2(p, snapshot / p.name)
    data_root = ROOT / "RoNIN_Team_Delivery/dataset"
    split_root = ROOT / "RoNIN_Team_Delivery/splits"
    fingerprints = list(csv.DictReader((reference / "data_fingerprints.csv").open(encoding="utf-8")))
    for row in fingerprints:
        path = data_root / row["path"]
        if path.stat().st_size != int(row["bytes"]) or sha(path) != row["sha256"]:
            raise ValueError(f"Data changed since baseline: {path}")
    shutil.copy2(reference / "data_fingerprints.csv", out / "data_fingerprints.csv")
    old_hashes = json.loads((reference / "source_fingerprints.json").read_text(encoding="utf-8"))
    for path in split_root.glob("*"):
        if sha(path) != old_hashes[str(path.relative_to(ROOT))]:
            raise ValueError(f"Split changed since baseline: {path}")
    train_ds = RoNINNPZDataset(data_root, split_root / "train.txt", filter_outliers=True)
    val_ds = RoNINNPZDataset(data_root, split_root / "val.txt", mode="val")
    if set(train_ds.files) & set(val_ds.files):
        raise ValueError("Training and validation sequences overlap")
    train = FixedWindows(train_ds, torch.device("cuda"))
    val = FixedWindows(val_ds, torch.device("cuda"))
    expected = json.loads((reference / "dataset_manifest.json").read_text(encoding="utf-8"))
    manifest = {}
    for name, pool in (("train", train), ("val", val)):
        actual = dict(windows=pool.n, index_sha256=pool.index_sha256, target_sha256=pool.target_sha256)
        for key, value in actual.items():
            if value != expected[name][key]:
                raise ValueError(f"{name} {key} differs from baseline")
        manifest[name] = actual
    dump(out / "dataset_manifest.json", manifest)
    del train_ds, val_ds
    summary = train_one("cnn_gru_attention", train, val, out, protocol, digest,
                        dict(orders), factory=CNNGRUAttention)
    dump(out / "summary.json", summary)
    dump(out / "acceptance_checks.json", dict(status="passed",same_protocol_as_reference=True,
         data_files_match_reference=True,splits_match_reference=True,
         window_indices_and_labels_match_reference=True,same_order_every_epoch=True,
         checkpoint_reload_passed=summary["checkpoint_reload_passed"],evaluation_completed=False))
    logging.info("Training completed; evaluation has not run. Summary: %s", json.dumps(summary))


if __name__ == "__main__":
    main()

