"""Deep autoencoder detector (Section IV-C, eq. 1).

Encoder widths 34-24-16-8 with a mirrored decoder, ReLU, batch normalisation and
dropout 0.2. Trained on benign traffic only by minimising the mean squared
reconstruction error; the per-request score is the L2 reconstruction error
e(x) = ||x - g(f(x))||_2, large exactly when x lies off the manifold of behaviour
seen in training.

Training follows Section V: Adam at 1e-3, batch 512, up to 120 epochs, cosine
annealing, early stopping with patience 10 on the held-out benign validation loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ztb.config import (
    AE_BATCH_SIZE,
    AE_DROPOUT,
    AE_EARLY_STOPPING_PATIENCE,
    AE_ENCODER_WIDTHS,
    AE_EPOCHS,
    AE_LEARNING_RATE,
)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class AEConfig:
    widths: tuple[int, ...] = AE_ENCODER_WIDTHS   # first entry is the input dim
    dropout: float = AE_DROPOUT
    epochs: int = AE_EPOCHS
    batch_size: int = AE_BATCH_SIZE
    lr: float = AE_LEARNING_RATE
    patience: int = AE_EARLY_STOPPING_PATIENCE


class AutoEncoder(nn.Module):
    """34-24-16-8 encoder, mirrored decoder. Output dim == input dim."""

    def __init__(self, widths: tuple[int, ...] = AE_ENCODER_WIDTHS, dropout: float = AE_DROPOUT):
        super().__init__()
        if len(widths) < 2:
            raise ValueError("need at least input and code widths")
        self.widths = tuple(widths)
        self.dropout = dropout
        enc: list[nn.Module] = []
        for a, b in zip(widths[:-1], widths[1:], strict=True):
            enc.append(nn.Linear(a, b))
            if b != widths[-1]:                  # no BN/ReLU/dropout on the code layer
                enc += [nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(dropout)]
        dec: list[nn.Module] = []
        rev = tuple(reversed(widths))
        for a, b in zip(rev[:-1], rev[1:], strict=True):
            dec.append(nn.Linear(a, b))
            if b != rev[-1]:                     # linear output layer
                dec += [nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(dropout)]
        self.encoder = nn.Sequential(*enc)
        self.decoder = nn.Sequential(*dec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    @torch.no_grad()
    def reconstruction_error(self, x: np.ndarray, batch_size: int = 65_536) -> np.ndarray:
        """e(x) = ||x - g(f(x))||_2 per row, computed in eval mode."""
        self.eval()
        device = next(self.parameters()).device
        out = np.empty(len(x), dtype=np.float32)
        for i in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[i:i + batch_size], dtype=torch.float32, device=device)
            out[i:i + batch_size] = torch.linalg.vector_norm(xb - self(xb), dim=1).cpu().numpy()
        return out


@dataclass
class TrainLog:
    epochs_run: int
    best_epoch: int
    best_val_loss: float
    train_loss: list[float]
    val_loss: list[float]


def train_autoencoder(
    x_train: np.ndarray,
    x_val: np.ndarray,
    *,
    config: AEConfig | None = None,
    seed: int = 0,
    device: str = "auto",
    verbose: bool = False,
) -> tuple[AutoEncoder, TrainLog]:
    """Fit on standardised benign rows; early-stop on benign validation loss."""
    config = config or AEConfig()
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = pick_device(device)
    widths = (x_train.shape[1],) + tuple(config.widths[1:])
    model = AutoEncoder(widths, config.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epochs)
    loss_fn = nn.MSELoss(reduction="none")

    xt = torch.as_tensor(x_train, dtype=torch.float32, device=dev)
    xv = torch.as_tensor(x_val, dtype=torch.float32, device=dev)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    best_state, best_val, best_epoch, bad = None, float("inf"), -1, 0
    tl, vl = [], []
    for epoch in range(config.epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen).to(dev)
        total = 0.0
        for i in range(0, len(xt), config.batch_size):
            xb = xt[perm[i:i + config.batch_size]]
            if len(xb) < 2:                     # BatchNorm needs > 1 row
                continue
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), xb).sum(dim=1).mean()   # eq. (1)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        sched.step()
        tl.append(total / max(1, len(xt)))

        model.eval()
        with torch.no_grad():
            v = 0.0
            for i in range(0, len(xv), 65_536):
                xb = xv[i:i + 65_536]
                v += loss_fn(model(xb), xb).sum(dim=1).sum().item()
            v /= max(1, len(xv))
        vl.append(v)
        if verbose:
            print(f"    epoch {epoch + 1:3d}  train {tl[-1]:.4f}  val {v:.4f}", flush=True)

        if v < best_val - 1e-6:
            best_val, best_epoch, bad = v, epoch, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, TrainLog(len(tl), best_epoch + 1, best_val, tl, vl)


def save_autoencoder(model: AutoEncoder, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"widths": model.widths, "dropout": model.dropout,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, path)


def load_autoencoder(path: Path, device: str = "cpu") -> AutoEncoder:
    payload = torch.load(path, map_location="cpu")
    model = AutoEncoder(tuple(payload["widths"]), payload["dropout"])
    model.load_state_dict(payload["state_dict"])
    return model.to(pick_device(device)).eval()


def config_dict(config: AEConfig) -> dict:
    return asdict(config)
