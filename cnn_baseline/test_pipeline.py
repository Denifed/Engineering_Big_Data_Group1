"""Contract and end-to-end smoke tests; synthetic results are not experiments."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cnn_baseline.dataset import RoNINNPZDataset, LABEL_PROTOCOL, validate_checkpoint
from cnn_baseline.common import MODELS
from cnn_baseline.model_cnn_gru_attention import CNNGRUAttention, TemporalAttention

torch.set_num_threads(2)


class MeanVelocityContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="cnn_mean_test_", dir=Path(__file__).resolve().parent)
        self.root = Path(self.tmp.name)
        t = np.cumsum(np.linspace(.004, .006, 80))
        p = np.column_stack((t + 2 * t ** 2, .5 * t, np.zeros_like(t)))
        f = np.random.default_rng(42).normal(size=(80, 6)).astype(np.float32)
        self.arrays = dict(features=f, positions=p, timestamps=t,
                           targets=np.tile([-3., -4.], (80, 1)).astype(np.float32))
        self.save()
        (self.root / "list.txt").write_text("sequence.npz\n", encoding="utf-8")

    def tearDown(self):
        assert self.root.resolve().parent == Path(__file__).resolve().parent
        self.tmp.cleanup()

    def save(self):
        np.savez(self.root / "sequence.npz", **self.arrays)

    def dataset(self, **kwargs):
        return RoNINNPZDataset(self.root, self.root / "list.txt", window_size=20, step_size=7, **kwargs)

    def test_nonuniform_time_endpoint_label_and_window_boundary(self):
        ds = self.dataset()
        for index in (0, len(ds) - 1):
            x, y, aux, seq = ds[index]
            start = ds.index_map[index][1]; end = start + 20
            expected = ((self.arrays["positions"][end, :2] - self.arrays["positions"][start, :2]) /
                        (self.arrays["timestamps"][end] - self.arrays["timestamps"][start]))
            np.testing.assert_allclose(y.numpy(), expected, rtol=1e-6)
            np.testing.assert_array_equal(x.numpy(), self.arrays["features"][start:end].T)
            self.assertFalse(np.allclose(y.numpy(), self.arrays["targets"][end]))
            self.assertEqual(aux.dtype, torch.float64)
            self.assertLess(end, len(self.arrays["timestamps"]))
        self.assertEqual([i[1] for i in ds.index_map], list(range(0, 60, 7)))

    def test_filter_covers_displacement_interval_not_future_label(self):
        self.arrays["targets"][20] = [100, 0]
        self.save()
        ds = self.dataset(filter_outliers=True)
        starts = [start for _, start in ds.index_map]
        self.assertIn(0, starts)  # [0,20) excludes interval 20 -> 21.
        self.assertNotIn(7, starts)
        self.assertNotIn(14, starts)
        self.assertEqual(ds.filtered_count, 2)

    def test_reject_invalid_time(self):
        self.arrays["timestamps"][5] = self.arrays["timestamps"][4]
        self.save()
        with self.assertRaisesRegex(ValueError, "时间戳"):
            self.dataset()

    def test_reject_legacy_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "label_protocol"):
            validate_checkpoint(dict(model_name="cnn", window_size=200, step_size=10), "cnn", 200, 10)
        validate_checkpoint(dict(model_name="cnn", window_size=200, step_size=10,
                                 label_protocol=LABEL_PROTOCOL), "cnn", 200, 10)

    def test_all_models_forward_backward(self):
        ds = self.dataset()
        x = torch.stack([ds[0][0], ds[1][0]])
        y = torch.stack([ds[0][1], ds[1][1]])
        for name, factory in MODELS.items():
            model = factory()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            before = next(model.parameters()).detach().clone()
            pred = model(x)
            self.assertEqual(tuple(pred.shape), (2, 2))
            loss = torch.nn.functional.mse_loss(pred, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            self.assertTrue(torch.isfinite(loss).item())
            self.assertFalse(torch.equal(before, next(model.parameters()).detach()), name)

    def test_attention_normalizes_over_time_and_uniform_scores_mean_pool(self):
        attention = TemporalAttention(hidden_size=4, attention_size=3)
        sequence = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
        context, weights = attention(sequence)
        self.assertEqual(tuple(weights.shape), (2, 3))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
        self.assertTrue((weights >= 0).all())
        for parameter in attention.parameters():
            torch.nn.init.zeros_(parameter)
        context, weights = attention(sequence)
        torch.testing.assert_close(weights, torch.full((2, 3), 1 / 3))
        torch.testing.assert_close(context, sequence.mean(dim=1))

    def test_attention_parameters_receive_velocity_loss_gradients(self):
        torch.manual_seed(42)
        model = CNNGRUAttention()
        ds = self.dataset()
        x = torch.stack([ds[0][0], ds[1][0]])
        y = torch.stack([ds[0][1], ds[1][1]])
        prediction, weights = model.forward_with_attention(x)
        self.assertEqual(tuple(weights.shape), (2, 5))
        loss = torch.nn.functional.mse_loss(prediction, y)
        loss.backward()
        for name, parameter in model.attention.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.assertGreater(model.gru.weight_ih_l0.grad.abs().sum().item(), 0)
        model.eval()
        with torch.inference_mode():
            prediction, weights = model.forward_with_attention(x)
            torch.testing.assert_close(prediction, model(x), rtol=0, atol=0)
            direct_prediction, direct_weights = model(x, return_attention=True)
            torch.testing.assert_close(direct_prediction, prediction, rtol=0, atol=0)
            torch.testing.assert_close(direct_weights, weights, rtol=0, atol=0)

    def test_attention_reuses_control_backbone_and_pools_all_gru_outputs(self):
        torch.manual_seed(42)
        control = MODELS["cnn_gru"]().eval()
        torch.manual_seed(42)
        attention = CNNGRUAttention().eval()
        for name, tensor in control.state_dict().items():
            torch.testing.assert_close(attention.state_dict()[name], tensor, rtol=0, atol=0)
        for parameter in attention.attention.parameters():
            torch.nn.init.zeros_(parameter)
        x = torch.randn(2, 6, 200)
        with torch.inference_mode():
            sequence, _ = control.gru(control.cnn(x).transpose(1, 2))
            expected = control.regressor(sequence.mean(dim=1))
            prediction, weights = attention(x, return_attention=True)
        torch.testing.assert_close(weights, torch.full((2, 50), .02))
        torch.testing.assert_close(prediction, expected)
        self.assertEqual(attention.model_config()["attention_size"], 32)
        self.assertEqual(sum(p.numel() for p in attention.parameters()), 80482)

    def test_train_reload_evaluate_from_another_working_directory(self):
        data = self.root / "data"; data.mkdir()
        splits = self.root / "splits"; splits.mkdir()
        for split in ("train", "val", "seen_test", "unseen_test"):
            np.savez(data / f"{split}.npz", **self.arrays)
            (splits / f"{split}.txt").write_text(f"{split}.npz\n", encoding="utf-8")
        checkpoints = []
        for model_name in ("cnn", "cnn_gru_attention"):
            out = self.root / model_name
            command = [sys.executable, "-B", str(ROOT / "cnn_baseline/train.py"),
                       "--model", model_name, "--data-root", str(data), "--split-root", str(splits),
                       "--out", str(out), "--epochs", "1", "--window-size", "20", "--step-size", "7",
                       "--batch-size", "4", "--device", "cpu"]
            result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            summary = json.loads((out / "summary.json").read_text())
            self.assertTrue(summary["checkpoint_reload_passed"])
            checkpoints.append(str(out / "checkpoint_best.pt"))
        evaluation = self.root / "evaluation"
        result = subprocess.run([sys.executable, "-B", str(ROOT / "cnn_baseline/evaluate.py"),
                                 "--checkpoint", *checkpoints, "--save-attention",
                                 "--data-root", str(data), "--split-root", str(splits),
                                 "--out", str(evaluation), "--device", "cpu"],
                                cwd=self.root, capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((evaluation / "test_comparison.png").is_file())
        summary = json.loads((evaluation / "summary.json").read_text())
        self.assertEqual(len(summary["test_metrics"]), 6)
        for split in ("seen_test", "unseen_test"):
            with np.load(evaluation / split / "cnn/window_predictions.npz") as d:
                mse = float(np.mean((d["predicted_velocity"].astype(float) -
                                     d["target_velocity"].astype(float)) ** 2))
            row = next(r for r in summary["test_metrics"] if r["split"] == split and r["model"] == "cnn")
            self.assertAlmostEqual(mse, row["MSE"], places=12)
            attention_folder = evaluation / split / "cnn_gru_attention"
            self.assertTrue((attention_folder / f"{split}_attention.png").is_file())
            with np.load(attention_folder / "window_predictions.npz") as d:
                weights = d["attention_weights"]
                self.assertEqual(weights.shape, (len(d["starts"]), 5))
                np.testing.assert_allclose(weights.sum(axis=1), 1, rtol=1e-6)

    def test_real_npz_agrees_with_a_loader_and_resnet_formula(self):
        real = ROOT / "RoNIN_Team_Delivery/dataset/train/a000_1.npz"
        if not real.is_file():
            self.skipTest("Delivered real sequence unavailable")
        listing = self.root / "real.txt"
        listing.write_text("train/a000_1.npz\n", encoding="utf-8")
        ds = RoNINNPZDataset(real.parents[1], listing)
        spec = importlib.util.spec_from_file_location("a_loader", ROOT / "RoNIN_Team_Delivery/08_ronin_dataset.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        reference = module.RoNINDataset(real.parents[1], listing, layout="cnn")
        for index in (0, 10, len(ds) - 1):
            x, y, _, _ = ds[index]; ax, ay = reference[index]
            torch.testing.assert_close(x, ax, rtol=0, atol=0)
            torch.testing.assert_close(y, ay, rtol=0, atol=0)
            start = ds.index_map[index][1]
            f, p, t = ds.sequences[0]
            expected = ((p[start + 200, :2] - p[start, :2]) /
                        (t[start + 200] - t[start])).astype(np.float32)
            np.testing.assert_array_equal(y.numpy(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)

