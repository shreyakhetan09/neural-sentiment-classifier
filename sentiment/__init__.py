"""Neural sentiment classification.

Binary sentiment (positive / negative) over the Stanford Sentiment Treebank
with two complementary neural architectures:

* a **Deep Averaging Network** (order-agnostic, fast), and
* a **(Bi)LSTM / GRU** sequence classifier (order-aware).

Both sit on top of pretrained GloVe embeddings and share a single training
loop, so they can be compared head to head.
"""

from .vocab import Vocabulary
from .models import DAN, RNNClassifier

__all__ = ["Vocabulary", "DAN", "RNNClassifier"]

__version__ = "1.0.0"
