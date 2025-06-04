from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    import torch
    from torch import nn, optim
except Exception:  # noqa: BLE001
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore

try:
    import tensorflow as tf
except Exception:  # noqa: BLE001
    tf = None


class Trainer:
    """Simple trainer that supports PyTorch or TensorFlow backends."""

    def __init__(self, checkpoint: Optional[str] = None, use_gpu: bool = True) -> None:
        self.checkpoint = checkpoint
        self.use_gpu = use_gpu and torch and torch.cuda.is_available()
        self.model = None
        if checkpoint:
            self.load(checkpoint)
        else:
            self._init_model()

    def _init_model(self) -> None:
        if torch:
            self.model = nn.Linear(10, 1)
            if self.use_gpu:
                self.model.cuda()
            self.optimizer = optim.Adam(self.model.parameters())
            self.loss_fn = nn.MSELoss()
        elif tf:
            self.model = tf.keras.Sequential([
                tf.keras.layers.Dense(1, input_shape=(10,)),
            ])
            self.model.compile(optimizer="adam", loss="mse")
        else:
            raise ImportError("Neither PyTorch nor TensorFlow is installed")

    def load(self, path: str) -> None:
        if torch and Path(path).with_suffix(".pt").exists():
            self.model = torch.load(path)
        elif tf and Path(path).with_suffix(".h5").exists():
            self.model = tf.keras.models.load_model(path)
        else:
            self._init_model()

    def train(self, data) -> None:
        """Train the model on provided data. Data is a placeholder."""
        if torch and isinstance(self.model, nn.Module):
            self.model.train()
            for x, y in data:
                if self.use_gpu:
                    x = x.cuda()
                    y = y.cuda()
                self.optimizer.zero_grad()
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.optimizer.step()
        elif tf:
            x, y = zip(*data)
            self.model.fit(tf.stack(x), tf.stack(y))
        else:
            raise RuntimeError("No ML backend available")

    def save(self, path: str) -> None:
        Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
        if torch and isinstance(self.model, nn.Module):
            torch.save(self.model, path if path.endswith(".pt") else f"{path}.pt")
        elif tf:
            self.model.save(path if path.endswith(".h5") else f"{path}.h5")
        else:
            raise RuntimeError("No ML backend available")
