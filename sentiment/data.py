"""Stanford Sentiment Treebank loading, padding, and batching.

Two entry points:

* :meth:`SSTDataset.from_parquet` reads the processed SST parquet splits used in
  the Stanford Sentiment Treebank release (columns ``tokens``, ``ids``,
  ``label``, ``is_root``) together with a token-list file for the vocabulary.
* :meth:`SSTDataset.toy` builds a small synthetic sentiment corpus so the models
  can be trained end to end without any downloads.

The binary task keeps only ``label`` values 0 (negative) and 1 (positive).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .vocab import Vocabulary, canonicalize_word


@dataclass
class Example:
    tokens: List[str]
    ids: List[int]
    label: int
    is_root: bool = True


def pad_id_lists(
    id_lists: List[List[int]], max_len: int = 40, pad_id: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Pad / truncate a list of id sequences to a dense ``(n, max_len)`` array."""
    arr = np.full((len(id_lists), max_len), pad_id, dtype=np.int64)
    lengths = np.zeros(len(id_lists), dtype=np.int64)
    for i, ids in enumerate(id_lists):
        n = min(len(ids), max_len)
        arr[i, :n] = ids[:n]
        lengths[i] = max(n, 1)
    return arr, lengths


@dataclass
class SSTDataset:
    vocab: Vocabulary
    train: List[Example] = field(default_factory=list)
    dev: List[Example] = field(default_factory=list)
    test: List[Example] = field(default_factory=list)

    def split(self, name: str) -> List[Example]:
        return {"train": self.train, "dev": self.dev, "test": self.test}[name]

    def as_padded_array(
        self, split: str = "train", max_len: int = 40, is_root: bool = True
    ):
        rows = self.split(split)
        if is_root:
            rows = [r for r in rows if r.is_root]
        x, lengths = pad_id_lists([r.ids for r in rows], max_len=max_len)
        y = np.array([r.label for r in rows], dtype=np.int64)
        return x, lengths, y

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_parquet(
        cls,
        train_file: str,
        dev_file: str,
        test_file: str,
        vocab_file: str,
        size: int = 20000,
    ) -> "SSTDataset":
        import pandas as pd

        with open(vocab_file, encoding="utf8") as handle:
            tokens = [w.strip() for w in handle if w.strip()]
        vocab = Vocabulary(tokens, size=size)

        def to_examples(df) -> List[Example]:
            out = []
            for row in df.itertuples(index=False):
                out.append(
                    Example(
                        tokens=list(getattr(row, "tokens", []) or []),
                        ids=list(row.ids),
                        label=int(getattr(row, "label", 0)),
                        is_root=bool(getattr(row, "is_root", True)),
                    )
                )
            return out

        return cls(
            vocab=vocab,
            train=to_examples(pd.read_parquet(train_file)),
            dev=to_examples(pd.read_parquet(dev_file)),
            test=to_examples(pd.read_parquet(test_file)),
        )

    @classmethod
    def toy(cls, n_per_split: int = 400, seed: int = 0) -> "SSTDataset":
        """A small templated sentiment corpus for smoke tests and demos."""
        rng = random.Random(seed)
        pos_adj = ["brilliant", "warm", "funny", "gorgeous", "touching",
                   "clever", "charming", "stunning", "delightful", "engaging"]
        neg_adj = ["dull", "clumsy", "boring", "lifeless", "pretentious",
                   "tedious", "forgettable", "shallow", "awful", "muddled"]
        subj = ["the film", "this movie", "the story", "the cast", "the script",
                "the sequel", "the ending", "the drama", "the comedy", "the plot"]
        pos_t = ["{s} is {a} and genuinely {a2}",
                 "a {a} , {a2} film that works",
                 "{s} is {a} from start to finish"]
        neg_t = ["{s} is {a} and painfully {a2}",
                 "a {a} , {a2} mess that never works",
                 "{s} is {a} from start to finish"]

        def make(n: int) -> List[Example]:
            texts = []
            for _ in range(n):
                if rng.random() < 0.5:
                    t = rng.choice(pos_t).format(
                        s=rng.choice(subj), a=rng.choice(pos_adj), a2=rng.choice(pos_adj))
                    label = 1
                else:
                    t = rng.choice(neg_t).format(
                        s=rng.choice(subj), a=rng.choice(neg_adj), a2=rng.choice(neg_adj))
                    label = 0
                texts.append((t, label))
            return texts  # type: ignore[return-value]

        raw = {s: make(n_per_split if s == "train" else n_per_split // 4)
               for s in ("train", "dev", "test")}

        all_tokens = [tok for (t, _) in raw["train"] for tok in t.split()]
        vocab = Vocabulary(all_tokens, size=5000)

        def to_examples(pairs) -> List[Example]:
            out = []
            for text, label in pairs:
                toks = [canonicalize_word(w, vocab.wordset) for w in text.split()]
                out.append(Example(tokens=toks, ids=vocab.words_to_ids(toks), label=label))
            return out

        return cls(
            vocab=vocab,
            train=to_examples(raw["train"]),
            dev=to_examples(raw["dev"]),
            test=to_examples(raw["test"]),
        )


class SentimentTorchDataset:
    """``torch.utils.data.Dataset`` yielding ``(ids, length, mask, label)``.

    During training a fraction of tokens is randomly dropped (word dropout),
    a cheap regulariser that works well with averaging models.
    """

    def __init__(
        self,
        sst: SSTDataset,
        split: str = "train",
        max_len: int = 40,
        word_dropout: float = 0.3,
        is_root: bool = True,
    ) -> None:
        self.split = split
        self.word_dropout = word_dropout if split == "train" else 0.0
        self.x, self.lengths, self.y = sst.as_padded_array(split, max_len, is_root)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        ids = self.x[idx]
        length = int(self.lengths[idx])
        mask = np.zeros(len(ids), dtype=np.int64)
        kept = 0
        for j in range(length):
            if self.word_dropout and random.random() < self.word_dropout:
                continue
            mask[j] = 1
            kept += 1
        if kept == 0:  # never drop everything
            mask[0] = 1
        label = int(self.y[idx]) if idx < len(self.y) else 0
        return ids, length, mask, label
