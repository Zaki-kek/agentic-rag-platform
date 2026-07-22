"""Linear-probe demo: a pure-numpy logistic regression over frozen embeddings.

Small, honest demo of a *trained* model on top of the RAG stack: a toy
linearly-separable text dataset (:mod:`app.ml.dataset`) is embedded with the
offline hash embedder and a single-layer logistic-regression probe
(:mod:`app.ml.probe`) is fitted with full-batch gradient descent. Everything is
offline, seeded and reproducible; the only dependency is ``numpy``.
"""

from app.ml.dataset import ProbeDataset, build_dataset, build_texts
from app.ml.probe import LogisticProbe, accuracy, bce_loss, sigmoid

__all__ = [
    "ProbeDataset",
    "build_dataset",
    "build_texts",
    "LogisticProbe",
    "accuracy",
    "bce_loss",
    "sigmoid",
]
