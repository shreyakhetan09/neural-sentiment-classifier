"""Shared training and evaluation loop for both architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import SentimentTorchDataset, SSTDataset


@dataclass
class TrainConfig:
    lr: float = 0.01
    batch_size: int = 128
    epochs: int = 5
    weight_decay: float = 1e-5
    word_dropout: float = 0.1
    max_len: int = 40
    is_root: bool = True
    seed: int = 1234


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collate(batch):
    import numpy as np

    ids, lengths, masks, labels = zip(*batch)
    return (
        torch.from_numpy(np.stack(ids)).long(),
        torch.as_tensor(lengths, dtype=torch.long),
        torch.from_numpy(np.stack(masks)).long(),
        torch.as_tensor(labels, dtype=torch.long),
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device, criterion) -> Tuple[float, float]:
    model.eval()
    correct = total = 0
    losses = []
    for ids, lengths, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        logits = model(ids, mask, lengths)
        losses.append(criterion(logits, labels).item())
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / max(total, 1), sum(losses) / max(len(losses), 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device) -> List[int]:
    model.eval()
    preds: List[int] = []
    for ids, lengths, mask, _ in loader:
        ids, mask = ids.to(device), mask.to(device)
        logits = model(ids, mask, lengths)
        preds.extend(logits.argmax(1).tolist())
    return preds


def train(model: nn.Module, sst: SSTDataset, config: TrainConfig | None = None):
    """Train ``model`` on ``sst`` and return (best_dev_accuracy, trained_model)."""
    config = config or TrainConfig()
    torch.manual_seed(config.seed)
    device = get_device()
    model.to(device)

    def make_loader(split: str, shuffle: bool, wd: float) -> DataLoader:
        ds = SentimentTorchDataset(
            sst, split, max_len=config.max_len,
            word_dropout=wd, is_root=config.is_root,
        )
        return DataLoader(ds, batch_size=config.batch_size, shuffle=shuffle, collate_fn=_collate)

    train_loader = make_loader("train", True, config.word_dropout)
    dev_loader = make_loader("dev", False, 0.0)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    best_acc, best_state = 0.0, None
    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        for ids, lengths, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(ids, mask, lengths), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        acc, val_loss = evaluate(model, dev_loader, device, criterion)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch {epoch:2d} | train loss {running / len(train_loader):.4f} "
            f"| dev loss {val_loss:.4f} | dev acc {acc:5.2f} | best {best_acc:5.2f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc, model
