#!/usr/bin/env python3
"""Train a sentiment classifier (DAN, LSTM, or GRU) on SST.

Run on the bundled synthetic corpus (no downloads):

    python scripts/train.py --demo --model dan

Or on the real Stanford Sentiment Treebank parquet splits + GloVe:

    python scripts/train.py --model lstm \
        --data-dir data --glove data/glove.840B.300d.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from sentiment.data import SSTDataset
from sentiment.embeddings import build_embedding_matrix, load_glove
from sentiment.models import DAN, RNNClassifier
from sentiment.trainer import TrainConfig, train


def build_model(name: str, matrix):
    name = name.lower()
    if name == "dan":
        return DAN(matrix, d_hidden=300, depth=2, dropout=0.2)
    if name in ("lstm", "gru"):
        return RNNClassifier(matrix, d_hidden=150, num_layers=2, bidirectional=True, cell=name)
    raise ValueError(f"unknown model '{name}' (choose dan, lstm, or gru)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dan", choices=["dan", "lstm", "gru"])
    parser.add_argument("--demo", action="store_true", help="use the synthetic toy corpus")
    parser.add_argument("--data-dir", default="data", help="dir with train/dev/test parquet")
    parser.add_argument("--glove", default=None, help="path to a GloVe text file")
    parser.add_argument("--dim", type=int, default=300, help="embedding dimension")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--out", default="checkpoint.pt")
    args = parser.parse_args()

    if args.demo:
        print("building synthetic SST-style corpus ...")
        sst = SSTDataset.toy()
        dim = 100
    else:
        d = Path(args.data_dir)
        sst = SSTDataset.from_parquet(
            d / "train.parquet", d / "dev.parquet", d / "test.parquet", d / "train_tokens.txt"
        )
        dim = args.dim

    glove = load_glove(args.glove, sst.vocab) if args.glove else None
    matrix = build_embedding_matrix(sst.vocab, dim=dim, glove=glove)
    print(f"vocab={sst.vocab.size} | embeddings={matrix.shape} | train={len(sst.train)}")

    model = build_model(args.model, matrix)

    cfg = TrainConfig()
    if args.model == "dan":
        cfg.lr, cfg.batch_size, cfg.epochs = 0.01, 128, 5
    else:
        cfg.lr, cfg.batch_size, cfg.epochs = 0.01, 64, 3
    cfg.is_root = not args.demo  # toy corpus has only root sentences
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.lr = args.lr

    best_acc, model = train(model, sst, cfg)
    print(f"\nbest dev accuracy: {best_acc:.2f}")

    torch.save(
        {
            "model_name": args.model,
            "state_dict": model.state_dict(),
            "embedding_matrix": matrix,
            "vocab_words": sst.vocab.ordered_words(),
            "max_len": cfg.max_len,
        },
        args.out,
    )
    print(f"saved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
