#!/usr/bin/env python3
"""Classify the sentiment of sentences with a trained checkpoint.

    python scripts/predict.py --checkpoint checkpoint.pt \
        --text "a warm , funny and genuinely engaging film"

    python scripts/predict.py --checkpoint checkpoint.pt --input reviews.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from sentiment.data import pad_id_lists
from sentiment.models import DAN, RNNClassifier
from sentiment.vocab import Vocabulary


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    matrix = ckpt["embedding_matrix"]
    name = ckpt["model_name"]
    if name == "dan":
        model = DAN(matrix)
    else:
        model = RNNClassifier(matrix, cell=name)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    vocab = Vocabulary.__new__(Vocabulary)
    words = ckpt["vocab_words"]
    vocab.id_to_word = dict(enumerate(words))
    vocab.word_to_id = {w: i for i, w in vocab.id_to_word.items()}
    vocab.wordset = set(vocab.word_to_id)
    vocab.size = len(words)
    vocab.UNK_ID = vocab.word_to_id.get("<unk>", 0)
    return model, vocab, ckpt["max_len"]


def classify(model, vocab, max_len, sentences):
    id_lists = [vocab.encode(s) or [vocab.UNK_ID] for s in sentences]
    x, lengths = pad_id_lists(id_lists, max_len=max_len)
    mask = (x != 0).astype(np.int64)
    with torch.no_grad():
        logits = model(
            torch.as_tensor(x), torch.as_tensor(mask), torch.as_tensor(lengths)
        )
        probs = torch.softmax(logits, dim=1)[:, 1].tolist()
    return probs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", default=None, help="a single sentence to classify")
    parser.add_argument("--input", default=None, help="file with one sentence per line")
    args = parser.parse_args()

    model, vocab, max_len = load_checkpoint(args.checkpoint)

    if args.text is not None:
        sentences = [args.text]
    elif args.input:
        sentences = [l.strip() for l in open(args.input, encoding="utf8") if l.strip()]
    else:
        parser.error("provide --text or --input")

    for sentence, p in zip(sentences, classify(model, vocab, max_len, sentences)):
        label = "positive" if p >= 0.5 else "negative"
        print(f"[{label} {p:.2f}] {sentence}")


if __name__ == "__main__":
    main()
