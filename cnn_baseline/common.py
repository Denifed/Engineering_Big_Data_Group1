"""Model registry and small shared runtime helpers."""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model_cnn1d import CNN1D
from .model_cnn_gru import CNNGRU
from .model_cnn_gru_attention import CNNGRUAttention

MODELS = {"cnn": CNN1D, "cnn_gru": CNNGRU, "cnn_gru_attention": CNNGRUAttention}


def select_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available()
                        else "cpu" if name == "auto" else name)


def plot_training(history, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = [r["epoch"] for r in history]
    ax.plot(epochs, [r["train_mse"] for r in history], "o-", label="Train (online)")
    ax.plot(epochs, [r["val_mse"] for r in history], "o-", label="Validation")
    ax.set(xlabel="Epoch", ylabel="Mean velocity MSE [(m/s)^2]")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=160)
    plt.close(fig)

