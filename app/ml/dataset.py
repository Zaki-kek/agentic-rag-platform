"""Toy linearly-separable text dataset for the linear probe (offline).

This is a deliberately small, neutral demo dataset: ~40 short texts split into
two classes whose vocabularies **do not overlap** (class ``A`` uses a fixed set
of technology words, class ``B`` a fixed set of nature words). Embedded with the
deterministic :class:`app.rag.embed.HashEmbedder`, the two classes land in
linearly-separable regions of the hashed bag-of-tokens space, which is exactly
what a linear probe should be able to recover (train/test accuracy ``1.0`` on a
70/30 split with the pinned seed).

Nothing here touches the network or an API key. The word banks are fixed and the
sentence assembly is driven by ``numpy.random.default_rng`` with an explicit
seed, so the whole dataset is reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.rag.embed import HashEmbedder

# Two disjoint vocabularies. Class A = technology, class B = nature. There is no
# shared token between the banks, so a bag-of-tokens representation is linearly
# separable by construction.
_CLASS_A_WORDS: tuple[str, ...] = (
    "processor",
    "database",
    "compiler",
    "network",
    "algorithm",
    "kernel",
    "server",
    "pipeline",
    "cache",
    "protocol",
    "runtime",
    "buffer",
)
_CLASS_B_WORDS: tuple[str, ...] = (
    "river",
    "forest",
    "meadow",
    "mountain",
    "glacier",
    "orchard",
    "canyon",
    "prairie",
    "wetland",
    "coastline",
    "woodland",
    "valley",
)

# Fixed connector tokens shared by both classes. Because they appear in every
# sentence they carry no class signal (they cancel out), so they do not break
# separability - they just make the texts read like sentences.
_CONNECTORS: tuple[str, ...] = ("the", "a", "this", "that", "some", "every")

_DEFAULT_SEED = 42
_DEFAULT_EMBED_DIM = 256
_SAMPLES_PER_CLASS = 20
_WORDS_PER_TEXT = 4
_DEFAULT_TEST_FRACTION = 0.3


@dataclass(frozen=True)
class ProbeDataset:
    """A ready-to-train linear-probe dataset with a train/test split.

    Attributes:
        x_train: Training feature matrix, shape ``(n_train, dim)``.
        y_train: Training labels (``0``/``1``), shape ``(n_train,)``.
        x_test: Test feature matrix, shape ``(n_test, dim)``.
        y_test: Test labels (``0``/``1``), shape ``(n_test,)``.
        texts: The raw texts, in the original (pre-split) order.
        labels: The raw labels aligned to ``texts``.
        dim: Embedding dimension of the feature matrices.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    texts: list[str]
    labels: np.ndarray
    dim: int


def _make_sentence(words: tuple[str, ...], rng: np.random.Generator) -> str:
    """Assemble one short sentence from a class word bank plus connectors.

    Args:
        words: The class-specific vocabulary to draw content words from.
        rng: The seeded random generator driving the sampling.

    Returns:
        A lowercase space-joined sentence of ``_WORDS_PER_TEXT`` content words
        interleaved with fixed connector tokens.
    """
    content = rng.choice(np.asarray(words), size=_WORDS_PER_TEXT, replace=True)
    connectors = rng.choice(np.asarray(_CONNECTORS), size=_WORDS_PER_TEXT, replace=True)
    tokens: list[str] = []
    for connector, word in zip(connectors, content, strict=True):
        tokens.append(str(connector))
        tokens.append(str(word))
    return " ".join(tokens)


def build_texts(seed: int = _DEFAULT_SEED) -> tuple[list[str], np.ndarray]:
    """Build the raw ``(texts, labels)`` corpus before embedding.

    The corpus is balanced: ``_SAMPLES_PER_CLASS`` class-``A`` texts (label
    ``0``) followed by the same number of class-``B`` texts (label ``1``).

    Args:
        seed: Seed for the ``numpy`` generator, for reproducibility.

    Returns:
        A ``(texts, labels)`` pair where ``labels`` is an ``int64`` array
        aligned to ``texts``.
    """
    rng = np.random.default_rng(seed)
    texts: list[str] = []
    labels: list[int] = []
    for _ in range(_SAMPLES_PER_CLASS):
        texts.append(_make_sentence(_CLASS_A_WORDS, rng))
        labels.append(0)
    for _ in range(_SAMPLES_PER_CLASS):
        texts.append(_make_sentence(_CLASS_B_WORDS, rng))
        labels.append(1)
    return texts, np.asarray(labels, dtype=np.int64)


def _train_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    test_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split features/labels into train and test partitions.

    Args:
        features: Feature matrix, shape ``(n, dim)``.
        labels: Labels aligned to ``features``, shape ``(n,)``.
        test_fraction: Fraction of samples held out for the test set, in
            ``(0, 1)``.
        rng: The seeded random generator driving the shuffle.

    Returns:
        A ``(x_train, y_train, x_test, y_test)`` tuple.
    """
    n_samples = features.shape[0]
    order = rng.permutation(n_samples)
    n_test = max(1, int(round(n_samples * test_fraction)))
    test_idx = order[:n_test]
    train_idx = order[n_test:]
    return (
        features[train_idx],
        labels[train_idx],
        features[test_idx],
        labels[test_idx],
    )


async def build_dataset(
    seed: int = _DEFAULT_SEED,
    dim: int = _DEFAULT_EMBED_DIM,
    test_fraction: float = _DEFAULT_TEST_FRACTION,
) -> ProbeDataset:
    """Build the embedded, split linear-probe dataset (offline, reproducible).

    Texts are generated from the two disjoint word banks, embedded with a
    :class:`app.rag.embed.HashEmbedder`, and split 70/30 (by default) into train
    and test partitions with a seeded shuffle.

    Args:
        seed: Seed for text generation and the split shuffle.
        dim: Embedding dimension for the hash embedder.
        test_fraction: Fraction of samples held out for testing, in ``(0, 1)``.

    Returns:
        A :class:`ProbeDataset` with train/test matrices and the raw corpus.

    Raises:
        ValueError: If ``test_fraction`` is not strictly in ``(0, 1)``.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    texts, labels = build_texts(seed)
    embedder = HashEmbedder(dim=dim)
    embeddings = await embedder.embed(texts)
    features = np.asarray(embeddings, dtype=np.float64)

    rng = np.random.default_rng(seed)
    x_train, y_train, x_test, y_test = _train_test_split(features, labels, test_fraction, rng)
    return ProbeDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        texts=texts,
        labels=labels,
        dim=dim,
    )
