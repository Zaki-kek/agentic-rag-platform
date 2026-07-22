"""Tests for the pure-numpy linear probe and its toy dataset (offline).

The invariants pinned here are what make this a *real* (if tiny) training demo,
not a stub:

* the loss decreases monotonically under full-batch gradient descent;
* the fitted probe separates the linearly-separable toy classes on both the
  train and the held-out test split (generalisation, not just memorisation);
* training is deterministic for a fixed seed;
* the analytic gradient matches a finite-difference numeric gradient. The
  finite-difference check is taken at the *active* coordinate
  ``j = argmax(|grad|)`` and asserts ``|grad[j]| > 1e-6`` first, so it cannot be
  satisfied trivially on a dead (zero-gradient) bucket.

All computation is offline (hash embedder), so there are no network calls.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml.dataset import build_dataset, build_texts
from app.ml.probe import LogisticProbe, accuracy, bce_loss, sigmoid

_SEED = 42
_DIM = 256
_LR = 0.5
_EPOCHS = 300


async def _fitted_probe() -> tuple[
    LogisticProbe, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]
]:
    """Build the toy dataset and fit a probe on it (shared test helper).

    Returns:
        A ``(probe, x_train, y_train, x_test, y_test, history)`` tuple.
    """
    dataset = await build_dataset(seed=_SEED, dim=_DIM)
    probe = LogisticProbe(seed=_SEED)
    history = probe.fit(dataset.x_train, dataset.y_train, lr=_LR, epochs=_EPOCHS)
    return (
        probe,
        dataset.x_train,
        dataset.y_train,
        dataset.x_test,
        dataset.y_test,
        history,
    )


def test_dataset_classes_have_disjoint_vocab() -> None:
    """The two classes must draw from non-overlapping word banks."""
    texts, labels = build_texts(seed=_SEED)
    assert len(texts) == labels.size
    assert set(labels.tolist()) == {0, 1}
    tokens_a: set[str] = set()
    tokens_b: set[str] = set()
    for text, label in zip(texts, labels.tolist(), strict=True):
        target = tokens_a if label == 0 else tokens_b
        target.update(text.split())
    # Connectors are shared on purpose; the *content* vocabularies must be
    # disjoint, which is what makes the problem linearly separable.
    connectors = {"the", "a", "this", "that", "some", "every"}
    content_a = tokens_a - connectors
    content_b = tokens_b - connectors
    assert content_a
    assert content_b
    assert content_a.isdisjoint(content_b)


async def test_loss_decreases_monotonically() -> None:
    """Full-batch GD must drive the BCE loss down every step (anchor)."""
    _, _, _, _, _, history = await _fitted_probe()
    assert len(history) == _EPOCHS
    # Monotone non-increasing across the whole run.
    assert all(history[i] >= history[i + 1] for i in range(len(history) - 1))
    # And a real, non-trivial drop from the ~ln(2) start.
    assert history[0] > history[-1]
    assert history[0] == pytest.approx(np.log(2.0), abs=0.05)


async def test_train_and_test_accuracy_generalise() -> None:
    """Fitted probe separates both splits well above a below-fact threshold."""
    probe, x_train, y_train, x_test, y_test, _ = await _fitted_probe()
    train_acc = accuracy(probe.predict(x_train), y_train)
    test_acc = accuracy(probe.predict(x_test), y_test)
    # Fact is 1.0/1.0; assert a threshold strictly below the fact so the test
    # is robust, and require the held-out set too (generalisation).
    assert train_acc >= 0.9
    assert test_acc >= 0.9


async def test_training_is_deterministic() -> None:
    """A fixed seed reproduces identical weights, bias and loss history."""
    dataset = await build_dataset(seed=_SEED, dim=_DIM)
    probe_a = LogisticProbe(seed=_SEED)
    history_a = probe_a.fit(dataset.x_train, dataset.y_train, lr=_LR, epochs=_EPOCHS)
    probe_b = LogisticProbe(seed=_SEED)
    history_b = probe_b.fit(dataset.x_train, dataset.y_train, lr=_LR, epochs=_EPOCHS)
    assert probe_a.weights is not None
    assert probe_b.weights is not None
    assert np.array_equal(probe_a.weights, probe_b.weights)
    assert probe_a.bias == probe_b.bias
    assert history_a == history_b


async def test_analytic_gradient_matches_finite_difference() -> None:
    """Analytic BCE gradient must match a numeric one at the active coordinate.

    Guards against a silently-broken gradient (the thing that would make
    "training" fake). Uses ``j = argmax(|grad|)`` so the check lands on a bucket
    that actually carries signal, and asserts ``|grad[j]| > 1e-6`` before the
    agreement check - otherwise the test would pass trivially on a dead bucket
    (e.g. an unused hash bin whose gradient is exactly ``0.0``).
    """
    dataset = await build_dataset(seed=_SEED, dim=_DIM)
    probe = LogisticProbe(seed=_SEED)
    # Partially fit so we evaluate at a non-degenerate interior point.
    probe.fit(dataset.x_train, dataset.y_train, lr=_LR, epochs=10)
    assert probe.weights is not None

    grad_weights, _ = probe.gradients(dataset.x_train, dataset.y_train)
    j = int(np.argmax(np.abs(grad_weights)))

    # The bucket must be active, otherwise finite-diff agreement is vacuous.
    assert abs(grad_weights[j]) > 1e-6

    eps = 1e-6
    w_plus = probe.weights.copy()
    w_minus = probe.weights.copy()
    w_plus[j] += eps
    w_minus[j] -= eps
    loss_plus = probe.loss_at(w_plus, probe.bias, dataset.x_train, dataset.y_train)
    loss_minus = probe.loss_at(w_minus, probe.bias, dataset.x_train, dataset.y_train)
    numeric = (loss_plus - loss_minus) / (2.0 * eps)

    assert abs(grad_weights[j] - numeric) < 1e-4


def test_sigmoid_and_bce_are_well_behaved() -> None:
    """Sigmoid is stable and monotone; BCE is minimal on a perfect predictor."""
    # Large-magnitude inputs must stay finite (no overflow / nan) and bounded
    # in [0, 1]; at |z|=50 float64 saturates to exactly 0.0 / 1.0, which is fine.
    saturated = sigmoid(np.array([-50.0, 50.0], dtype=np.float64))
    assert np.all(np.isfinite(saturated))
    assert np.all(saturated >= 0.0)
    assert np.all(saturated <= 1.0)
    # Away from saturation it is strictly inside (0, 1), monotone and hits 0.5.
    z = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
    probs = sigmoid(z)
    assert np.all(probs > 0.0)
    assert np.all(probs < 1.0)
    assert probs[1] == pytest.approx(0.5)
    assert probs[0] < probs[1] < probs[2]
    # Perfect predictions -> near-zero loss; confident-wrong -> large loss.
    targets = np.array([1.0, 0.0], dtype=np.float64)
    perfect = np.array([0.999, 0.001], dtype=np.float64)
    wrong = np.array([0.001, 0.999], dtype=np.float64)
    assert bce_loss(perfect, targets) < bce_loss(wrong, targets)
    assert bce_loss(perfect, targets) < 0.05
