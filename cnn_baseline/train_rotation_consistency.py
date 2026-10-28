"""Controlled 30-epoch yaw augmentation vs augmentation + equivariance loss."""
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn_baseline.dataset import RoNINNPZDataset, LABEL_PROTOCOL, validate_checkpoint
from cnn_baseline.model_cnn_gru import CNNGRU
from cnn_baseline.rotation_training import rotate_features, paired_loss
from cnn_baseline.run_fixed_comparison import PROTOCOL as OLD_PROTOCOL, FixedWindows, setup_seed, dump, sha
from cnn_baseline.sequence_dataset import validation_sequences
from cnn_baseline.train_drift_comparison import setup_runtime, state_sha, validate
from cnn_baseline.compare_with_resnet import write_csv

OLD = ROOT / "results/fixed_resnet18_vs_cnn_gru_20261025"
CONDITIONS = {"yaw_augmentation": 0., "yaw_consistency": .1}
PROTOCOL = dict(OLD_PROTOCOL, experiment="random_windows_so2_supervision_equivariance_v1",
    augmentation="paired original and random-yaw copy; both gyro and acceleration XY rotated together with velocity target",
    anchor_batch_size=256, network_batch_size="2*anchor_batch_size; originals then rotated copies in one forward",
    loss="0.5*(MSE(f(x),y)+MSE(f(Rx),Ry)) + lambda*MSE(f(Rx),R*f(x))",
    conditions=CONDITIONS, rotations="one independent angle per anchor, uniform [0,2*pi), FP32",
    rotation_rng="CPU torch.Generator seed=20261027+epoch-1; generated in epoch sample order before training",
    gradient="both prediction branches receive gradients; no detach",
    selection="minimum unaugmented full-validation velocity MSE; first ties; best-ATE checkpoint diagnostic only",
    save_checkpoints="best MSE (primary), best ATE (diagnostic), last; never choose on test metrics",
    initialization="fresh CNN-GRU seed=42, same as old controlled CNN-GRU; not fine-tuning",
    evaluation="native training BN, single and fixed C4; no calibration or test-time training")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    ints = {"epoch", "train_anchors", "network_windows", "optimizer_steps", "val_windows"}
    hashes = {"order_sha256", "angles_sha256"}
    return [{k: v if k in hashes else int(v) if k in ints else float(v) for k, v in r.items()} for r in rows]


def plot_history(rows, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = [r["epoch"] for r in rows]
    for key in ("train_supervised_mse", "train_consistency_mse", "train_total_loss"):
        axes[0].plot(epochs, [r[key] for r in rows], label=key)
    axes[0].set(xlabel="Epoch", ylabel="Online loss [(m/s)^2]"); axes[0].legend(fontsize=7)
    axes[1].plot(epochs, [r["val_mse"] for r in rows]); axes[1].set(xlabel="Epoch", ylabel="Validation MSE (primary)")
    axes[2].plot(epochs, [r["val_ATE_official_m"] for r in rows]); axes[2].set(xlabel="Epoch", ylabel="Validation ATE [m]")
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def train_condition(name, weight, train, val, metadata, out, protocol, digest, orders, initialization, resume):
    folder = out/name
    if resume and (folder/"summary.json").exists():
        prior = read_json(folder/"summary.json")
        if prior["status"] == "training_completed":
            assert prior["epochs_completed"] == protocol["epochs"]
            return prior
    folder.mkdir(exist_ok=resume)
    setup_seed(42); model = CNNGRU().cuda()
    initial_hash = state_sha(model)
    assert initial_hash == read_json(ROOT/"results/drift_loss_cnn_gru_20261026/initialization.json")["state_sha256"]
    if initialization:
        assert initial_hash == initialization["state_sha256"]
    else:
        initialization.update(state_sha256=initial_hash, parameters=sum(p.numel() for p in model.parameters()),
                              seed=42, architecture="unchanged CNN-GRU")
        dump(out/"initialization.json", initialization)
    config = dict(condition=name, lambda_equivariance=weight, protocol=protocol,
                  protocol_sha256=digest, initialization_sha256=initial_hash,
                  parameter_count=sum(p.numel() for p in model.parameters()))
    if (folder/"config.json").exists():
        assert read_json(folder/"config.json") == config
    else:
        dump(folder/"config.json", config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0., amsgrad=False)
    history = read_history(folder/"training_log.csv") if resume else []
    best_mse = float("inf"); best_mse_epoch = 0; best_ate = float("inf"); best_ate_epoch = 0; first = 1
    if history:
        saved = torch.load(folder/"checkpoint_last.pt", map_location="cpu", weights_only=True)
        assert saved["epoch"] == history[-1]["epoch"] and saved["protocol_sha256"] == digest
        model.load_state_dict(saved["model_state_dict"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        torch.set_rng_state(saved["torch_rng_state"]); torch.cuda.set_rng_state_all(saved["cuda_rng_state"])
        first = saved["epoch"]+1; best_mse=saved["best_val_mse"]; best_mse_epoch=saved["best_epoch"]
        best_ate=saved["best_val_ATE"]; best_ate_epoch=saved["best_ATE_epoch"]
    mode = "a" if history else "w"
    with (folder/"training_log.csv").open(mode, newline="", encoding="utf-8") as handle:
        writer = None
        for epoch in range(first, protocol["epochs"]+1):
            tic = time.perf_counter()
            order = torch.randperm(train.n, generator=torch.Generator().manual_seed(42+epoch-1))
            angles = torch.rand(train.n, generator=torch.Generator().manual_seed(20261027+epoch-1))*float(2*math.pi)
            manifest = dict(order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),
                            angles_sha256=hashlib.sha256(angles.numpy().tobytes()).hexdigest())
            if str(epoch) in orders:
                assert orders[str(epoch)] == manifest
            else:
                orders[str(epoch)] = manifest
            if not protocol["smoke_only"]:
                assert manifest["order_sha256"] == read_json(OLD/"data_order_manifest.json")[str(epoch)]
            cosine = angles.cos().cuda(); sine = angles.sin().cuda()
            totals = np.zeros(3); count = 0; steps = 0; model.train()
            for batch, (x, y) in enumerate(train.batches(order, 256), 1):
                n = len(x); c=cosine[count:count+n]; s=sine[count:count+n]
                xr = rotate_features(x, c, s)
                optimizer.zero_grad(set_to_none=True)
                output = model(torch.cat((x, xr), dim=0))
                loss, supervised, consistency = paired_loss(output[:n], output[n:], y, c, s, weight)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward(); optimizer.step()
                totals += np.array([supervised.item(), consistency.item(), loss.item()])*n
                count += n; steps += 1
                if batch % 500 == 0:
                    logging.info("BATCH %s epoch=%d %d/%d supervised=%.6f equivariance=%.6f elapsed=%.1fs", name, epoch,
                                 batch, math.ceil(train.n/256), totals[0]/count, totals[1]/count, time.perf_counter()-tic)
            assert count == train.n
            metrics, sequences = validate(model, val, metadata)
            row = dict(epoch=epoch, train_supervised_mse=float(totals[0]/count), train_consistency_mse=float(totals[1]/count),
                       train_total_loss=float(totals[2]/count), val_mse=metrics["MSE"], val_ATE_official_m=metrics["ATE_official_m"],
                       val_RTE_60s_official_m=metrics["RTE_60s_official_m"], val_endpoint_error_m=metrics["endpoint_error_m"],
                       val_bias_norm_mps=metrics["mean_velocity_bias_norm_mps"], train_anchors=count, network_windows=2*count,
                       optimizer_steps=steps, val_windows=val.n, learning_rate=1e-4, seconds=time.perf_counter()-tic, **manifest)
            improved = metrics["MSE"] < best_mse; improved_ate = metrics["ATE_official_m"] < best_ate
            if improved:
                best_mse=metrics["MSE"]; best_mse_epoch=epoch
            if improved_ate:
                best_ate=metrics["ATE_official_m"]; best_ate_epoch=epoch
            saved = dict(model_name="cnn_gru", condition=name, lambda_equivariance=weight, epoch=epoch,
                         model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                         best_epoch=best_mse_epoch, best_val_mse=best_mse, best_ATE_epoch=best_ate_epoch, best_val_ATE=best_ate,
                         label_protocol=LABEL_PROTOCOL, window_size=200, step_size=10, config=config,
                         protocol_sha256=digest, metrics=row, torch_rng_state=torch.get_rng_state(),
                         cuda_rng_state=torch.cuda.get_rng_state_all())
            torch.save(saved, folder/"checkpoint_last.pt")
            write_csv(folder/"validation_sequence_metrics_last.csv", sequences)
            if improved:
                torch.save(saved, folder/"checkpoint_best.pt")
                write_csv(folder/"validation_sequence_metrics_best.csv", sequences)
            if improved_ate:
                torch.save(saved, folder/"checkpoint_best_ATE_diagnostic.pt")
                write_csv(folder/"validation_sequence_metrics_best_ATE_diagnostic.csv", sequences)
            if writer is None:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if not history:
                    writer.writeheader()
            writer.writerow(row); handle.flush(); history.append(row)
            plot_history(history, folder/"training_curves.png")
            dump(out/"data_order_and_rotation_manifest.json", orders)
            dump(out/"progress.json", dict(stage="training", condition=name, epoch=epoch, total_epochs=protocol["epochs"],
                  best_epoch=best_mse_epoch, best_val_mse=best_mse, last_epoch_seconds=row["seconds"]))
            logging.info("EPOCH_DONE %s %s best_MSE_epoch=%d best_ATE_epoch=%d", name, json.dumps(row), best_mse_epoch, best_ate_epoch)
    checkpoint = torch.load(folder/"checkpoint_best.pt", map_location="cuda", weights_only=True)
    fresh = CNNGRU().cuda().eval(); fresh.load_state_dict(checkpoint["model_state_dict"], strict=True)
    validate_checkpoint(checkpoint, "cnn_gru", 200, 10)
    recomputed, records = validate(fresh, val, metadata)
    np.testing.assert_allclose(recomputed["MSE"], best_mse, rtol=1e-12, atol=1e-12)
    x, _ = next(val.batches(torch.arange(val.n), 256))
    model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(x), fresh(x), rtol=0, atol=0)
    result = dict(status="training_completed", condition=name, epochs_completed=len(history), best_epoch=best_mse_epoch,
                  best_val_mse=best_mse, best_validation_metrics=recomputed, best_ATE_epoch_diagnostic=best_ate_epoch,
                  best_ATE_diagnostic=best_ate, parameter_count=initialization["parameters"], initialization_sha256=initial_hash,
                  protocol_sha256=digest, lambda_equivariance=weight, checkpoint_reload_passed=True,
                  validation_metrics_recomputed=True, training_wall_seconds=sum(r["seconds"] for r in history))
    dump(folder/"summary.json", result)
    logging.info("CONDITION_COMPLETED %s %s", name, json.dumps(result))
    del model, fresh, optimizer; torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); setup_runtime()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=args.resume)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler(sys.stdout),
                       logging.FileHandler(out/"run.log", mode="a" if args.resume else "w", encoding="utf-8")])
    protocol = dict(PROTOCOL, smoke_only=args.smoke)
    if args.smoke:
        protocol.update(epochs=2, smoke_anchor_limit=2048)
    digest=hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if args.resume:
        assert read_json(out/"protocol.json") == dict(protocol=protocol, protocol_sha256=digest)
        for rel, expected in read_json(out/"source_fingerprints.json").items():
            assert sha(ROOT/rel) == expected, f"Source changed since training: {rel}"
    else:
        dump(out/"protocol.json", dict(protocol=protocol, protocol_sha256=digest))
        dump(out/"environment.json", dict(python=sys.version, executable=sys.executable, torch=str(torch.__version__),
             numpy=np.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
        (out/"PLAN_执行计划.md").write_text("""# 旋转一致性训练：预先固定的实验计划

两组各训练30轮，A为原始+旋转样本的双分支监督，B增加lambda=0.1的旋转一致性MSE。
均从seed42初始化，不从旧最佳权重微调。沿用旧随机窗口训练索引、顺序、Adam lr1e-4、batch256原始窗口；每批拼接旋转副本，一次网络前向处理512窗口。
每窗口角度独立均匀采样[0,2pi)，训练顺序和角度使用独立CPU RNG预生成，两组完全相同。同步旋转gyro/acc XY及速度标签；Z与时间不变。
主选模指标为未增强的完整验证集速度MSE；另存最佳ATE诊断权重，主实验不依据其测试表现换选权重。
全部30轮，无早停/调度/标准化/额外噪声或BN重校准。全量验证、seen和unseen，保存单次/C4两种推理。
主对照B_single vs A_single隔离一致性损失；与旧CNN-GRU_single比较总训练方案；与旧CNN-GRU_C4比较是否接近四次推理精度。
同时报告RMSE/MAE/ATE/60sRTE/终点误差/残差偏置、逐序列变化、尾部、成功失败图和计算开销。
相对A的整体改善定义：seen/unseen ATE均降低，unseen终点误差/偏置/RMSE不劣于A。与旧路线也按同样规则分别判断。
所有条件、结果全部报告，lambda不根据测试修改；当前测试集为重复使用的探索性评估，不宣称独立盲测或统计显著性。
""", encoding="utf-8")
        (out/"source_snapshot").mkdir()
        sources=[*Path(__file__).parent.glob("*.py"), ROOT/"scripts/run_ronin_npz_baseline.py", ROOT/"ronin-master/source/metric.py",
                 *list((ROOT/"RoNIN_Team_Delivery/splits").glob("*"))]
        dump(out/"source_fingerprints.json", {str(p.relative_to(ROOT)):sha(p) for p in sources})
        for p in sources:
            shutil.copy2(p, out/"source_snapshot"/p.name)
        watched={p for ref in (OLD, ROOT/"results/cnn_gru_rotation_consistency_20261026") for p in ref.rglob("*") if p.is_file()}
        watched.update(sources)
        dump(out/"original_artifact_hashes.json", {str(p.relative_to(ROOT)):sha(p) for p in sorted(watched)})
    fingerprints=list(csv.DictReader((OLD/"data_fingerprints.csv").open(encoding="utf-8")))
    data=ROOT/"RoNIN_Team_Delivery/dataset"
    for row in fingerprints:
        p=data/row["path"]
        assert p.stat().st_size == int(row["bytes"]) and sha(p)==row["sha256"]
    write_csv(out/"data_fingerprints.csv", fingerprints)
    splits=ROOT/"RoNIN_Team_Delivery/splits"
    if args.smoke:
        splits=out/"smoke_splits"; splits.mkdir(exist_ok=args.resume)
        for split in ("train", "val"):
            names=[r["path"] for r in fingerprints if r["split"]==split][:2]
            (splits/f"{split}.txt").write_text("\n".join(names)+"\n", encoding="utf-8")
    train_ds=RoNINNPZDataset(data, splits/"train.txt", filter_outliers=True)
    val_ds=RoNINNPZDataset(data, splits/"val.txt", mode="val")
    assert not set(train_ds.files)&set(val_ds.files)
    train=FixedWindows(train_ds, torch.device("cuda"), limit=2048 if args.smoke else None)
    val=FixedWindows(val_ds, torch.device("cuda")); metadata=validation_sequences(val_ds)
    manifest=dict(train_windows=train.n, val_windows=val.n, train_sequences=len(train_ds.files), val_sequences=len(val_ds.files),
                  train_index_sha256=train.index_sha256, train_targets_sha256=train.target_sha256,
                  val_index_sha256=val.index_sha256, val_targets_sha256=val.target_sha256, filtered_windows=train_ds.filtered_count)
    if not args.smoke:
        previous=read_json(ROOT/"results/drift_loss_cnn_gru_20261026/dataset_manifest.json")
        assert train.index_sha256==previous["train"]["index_sha256"] and train.target_sha256==previous["train"]["target_sha256"]
        assert val.index_sha256==previous["validation"]["index_sha256"] and val.target_sha256==previous["validation"]["target_sha256"]
    dump(out/"dataset_manifest.json", manifest); logging.info("DATA_READY %s", json.dumps(manifest))
    del train_ds, val_ds
    orders=read_json(out/"data_order_and_rotation_manifest.json") if args.resume and (out/"data_order_and_rotation_manifest.json").exists() else {}
    initialization=read_json(out/"initialization.json") if args.resume and (out/"initialization.json").exists() else {}
    results={name:train_condition(name, w, train, val, metadata, out, protocol, digest, orders, initialization, args.resume)
             for name,w in CONDITIONS.items()}
    logs={n:read_history(out/n/"training_log.csv") for n in CONDITIONS}
    assert all(len(rows)==protocol["epochs"] for rows in logs.values())
    for epoch in range(protocol["epochs"]):
        a,b=(logs[n][epoch] for n in CONDITIONS)
        assert all(a[k]==b[k] for k in ("order_sha256", "angles_sha256", "train_anchors", "network_windows", "optimizer_steps"))
    checks=dict(status="passed", smoke_only=args.smoke, same_initialization=True, same_anchor_order=True, same_rotations=True,
                same_network_batch_organization=True, same_optimization_steps=True, same_MSE_selection=True,
                checkpoints_reloaded=True, full_validation_metrics_recomputed=True)
    dump(out/"training_acceptance.json", checks)
    dump(out/"summary.json", dict(status="training_completed", smoke_only=args.smoke, protocol_sha256=digest, training=results))
    dump(out/"progress.json", dict(stage="training_completed"))
    logging.info("BOTH_TRAINING_COMPLETED %s", json.dumps(results))


if __name__ == "__main__":
    main()
