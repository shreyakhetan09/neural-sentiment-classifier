"""Neural architectures for binary sentiment classification.

* :class:`DAN` - a Deep Averaging Network. Average the (masked) word embeddings,
  then pass the result through a feed-forward stack. Order agnostic but fast and
  surprisingly strong on sentence-level sentiment.
* :class:`RNNClassifier` - a (bi)LSTM or GRU over the token embeddings whose
  final hidden state feeds a linear classifier. Order aware.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def embedding_from_matrix(matrix: np.ndarray, freeze: bool = False) -> nn.Embedding:
    """Build an ``nn.Embedding`` initialised from a numpy embedding matrix."""
    tensor = torch.tensor(matrix, dtype=torch.float32)
    layer = nn.Embedding(*tensor.shape)
    layer.weight.data.copy_(tensor)
    layer.weight.requires_grad = not freeze
    return layer


class DAN(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        d_hidden: int = 300,
        d_out: int = 2,
        depth: int = 2,
        dropout: float = 0.2,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.embed = embedding_from_matrix(embedding_matrix, freeze_embeddings)
        d_embed = embedding_matrix.shape[1]

        layers: list[nn.Module] = []
        in_dim = d_embed
        for _ in range(depth):
            layers += [nn.Linear(in_dim, d_hidden), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = d_hidden
        layers.append(nn.Linear(in_dim, d_out))
        self.classifier = nn.Sequential(*layers)

    @staticmethod
    def masked_mean(vectors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = vectors.size()
        mask = mask.view(batch, seq, 1).expand(batch, seq, dim).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (vectors * mask).sum(dim=1) / denom

    def forward(self, text: torch.Tensor, mask: torch.Tensor, lengths=None):
        embedded = self.embed(text)
        pooled = self.masked_mean(embedded, mask)
        return self.classifier(pooled)


class RNNClassifier(nn.Module):
    def __init__(
        self,
        embedding_matrix: np.ndarray,
        d_hidden: int = 150,
        d_out: int = 2,
        num_layers: int = 2,
        bidirectional: bool = True,
        cell: str = "lstm",
        dropout: float = 0.3,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()
        self.embed = embedding_from_matrix(embedding_matrix, freeze_embeddings)
        d_embed = embedding_matrix.shape[1]
        self.bidirectional = bidirectional
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell.lower()]
        self.rnn = rnn_cls(
            input_size=d_embed,
            hidden_size=d_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if num_layers == 1 else dropout,
        )
        self.fc = nn.Linear(d_hidden * (2 if bidirectional else 1), d_out)

    def forward(self, text: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor):
        embedded = self.embed(text)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output = self.rnn(packed)
        hidden = output[1][0] if isinstance(output[1], tuple) else output[1]
        if self.bidirectional:
            final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            final = hidden[-1]
        return self.fc(final)
