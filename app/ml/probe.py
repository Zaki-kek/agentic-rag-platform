"""Pure-numpy logistic-regression linear probe (offline, reproducible).

This is a small demo classifier - a single linear layer trained with full-batch
gradient descent on the binary cross-entropy loss - used to check that a linear
decision boundary can be recovered from frozen embeddings. It is intentionally
minimal: no ``scipy``/``sklearn``, only ``numpy``.

The implementation is written so the training is *verifiable*: :func:`bce_loss`
and :meth:`LogisticProbe.gradients` are exposed so a finite-difference check can
confirm the analytic gradient matches a numeric one. Weights are initialised
from a seeded generator (``rules/experiment-reproducibility.md``), so a given
seed reproduces the same fit exactly.
"""

from __future__ import annotations

import numpy as np

from app.core import get_logger

logger = get_logger(__name__)

_DEFAULT_SEED = 42
_DEFAULT_LR = 0.5
_DEFAULT_EPOCHS = 300
_LOG_EVERY = 50
# Clamp for the log terms in BCE so a saturated sigmoid never yields log(0).
_EPS = 1e-12
# Small std for the seeded weight init; small enough to start near loss ~0.693.
_INIT_STD = 0.01


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid.

    Computed branch-wise so large-magnitude inputs do not overflow ``exp``.

    Args:
        z: Pre-activation array of any shape.

    Returns:
        ``1 / (1 + exp(-z))`` elementwise, same shape as ``z``, in ``(0, 1)``.
    """
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def bce_loss(probs: np.ndarray, targets: np.ndarray) -> float:
    """Mean binary cross-entropy between predicted probabilities and targets.

    Args:
        probs: Predicted probabilities in ``(0, 1)``, shape ``(n,)``.
        targets: Binary targets (``0``/``1``), shape ``(n,)``.

    Returns:
        The mean BCE loss as a Python ``float``. Probabilities are clamped to
        ``[_EPS, 1 - _EPS]`` so the log terms stay finite.
    """
    clipped = np.clip(probs, _EPS, 1.0 - _EPS)
    losses = -(targets * np.log(clipped) + (1.0 - targets) * np.log(1.0 - clipped))
    return float(np.mean(losses))


class LogisticProbe:
    """Single-layer logistic-regression probe trained with full-batch GD.

    Attributes:
        weights: Learned weight vector, shape ``(dim,)`` (``None`` before fit).
        bias: Learned bias scalar (``0.0`` before fit).
        seed: Seed used for the weight initialisation.
    """

    def __init__(self, seed: int = _DEFAULT_SEED) -> None:
        """Initialise an unfitted probe.

        Args:
            seed: Seed for the weight initialisation, for reproducibility.
        """
        self.seed = seed
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0

    def _init_params(self, dim: int) -> None:
        """Seed-initialise ``weights`` (small std) and ``bias`` (zero).

        Args:
            dim: Feature dimension of the inputs.
        """
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, _INIT_STD, size=dim).astype(np.float64)
        self.bias = 0.0

    def _logits(self, features: np.ndarray) -> np.ndarray:
        """Linear pre-activations ``X @ w + b``.

        Args:
            features: Feature matrix, shape ``(n, dim)``.

        Returns:
            Logit array, shape ``(n,)``.

        Raises:
            RuntimeError: If the probe has not been fitted / initialised.
        """
        if self.weights is None:
            raise RuntimeError("probe is not fitted; call fit() first")
        return features @ self.weights + self.bias

    def gradients(
        self, features: np.ndarray, targets: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Analytic BCE gradients w.r.t. weights and bias at current params.

        For logistic regression with mean BCE the gradient is the familiar
        ``Xᵀ(p - y) / n`` for the weights and ``mean(p - y)`` for the bias.

        Args:
            features: Feature matrix, shape ``(n, dim)``.
            targets: Binary targets, shape ``(n,)``.

        Returns:
            A ``(grad_weights, grad_bias)`` pair, where ``grad_weights`` has
            shape ``(dim,)`` and ``grad_bias`` is a Python ``float``.
        """
        probs = sigmoid(self._logits(features))
        residual = probs - targets
        n_samples = float(features.shape[0])
        grad_weights = (features.T @ residual) / n_samples
        grad_bias = float(np.sum(residual) / n_samples)
        return grad_weights, grad_bias

    def loss_at(
        self, weights: np.ndarray, bias: float, features: np.ndarray, targets: np.ndarray
    ) -> float:
        """BCE loss for an *arbitrary* weight/bias pair (for finite-diff checks).

        This does not touch the probe's own parameters; it evaluates the loss at
        the supplied ``weights``/``bias`` so a caller can perturb one coordinate
        and measure the numeric gradient.

        Args:
            weights: Weight vector to evaluate, shape ``(dim,)``.
            bias: Bias scalar to evaluate.
            features: Feature matrix, shape ``(n, dim)``.
            targets: Binary targets, shape ``(n,)``.

        Returns:
            The mean BCE loss at ``(weights, bias)``.
        """
        logits = features @ weights + bias
        return bce_loss(sigmoid(logits), targets)

    def fit(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        lr: float = _DEFAULT_LR,
        epochs: int = _DEFAULT_EPOCHS,
    ) -> list[float]:
        """Train the probe with full-batch gradient descent on BCE.

        Weights are seed-initialised, then updated for ``epochs`` steps. The
        loss is recorded every step and a subsample is logged at ``INFO`` so the
        (monotone) decrease is visible.

        Args:
            features: Training feature matrix, shape ``(n, dim)``.
            targets: Training targets (``0``/``1``), shape ``(n,)``.
            lr: Learning rate (gradient-descent step size). Must be positive.
            epochs: Number of full-batch update steps. Must be positive.

        Returns:
            The per-epoch loss history (length ``epochs``).

        Raises:
            ValueError: If ``lr`` or ``epochs`` is not positive.
        """
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")

        features = np.asarray(features, dtype=np.float64)
        targets = np.asarray(targets, dtype=np.float64)
        self._init_params(features.shape[1])
        assert self.weights is not None  # narrowed for type-checkers

        history: list[float] = []
        for epoch in range(epochs):
            probs = sigmoid(self._logits(features))
            history.append(bce_loss(probs, targets))
            grad_weights, grad_bias = self.gradients(features, targets)
            self.weights -= lr * grad_weights
            self.bias -= lr * grad_bias
            if epoch % _LOG_EVERY == 0 or epoch == epochs - 1:
                logger.info("probe epoch %d/%d loss=%.4f", epoch, epochs, history[-1])
        return history

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predicted positive-class probabilities.

        Args:
            features: Feature matrix, shape ``(n, dim)``.

        Returns:
            Probability array in ``(0, 1)``, shape ``(n,)``.
        """
        return sigmoid(self._logits(np.asarray(features, dtype=np.float64)))

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Hard class predictions at a probability threshold.

        Args:
            features: Feature matrix, shape ``(n, dim)``.
            threshold: Decision threshold on the positive-class probability.

        Returns:
            Integer label array (``0``/``1``), shape ``(n,)``.
        """
        return (self.predict_proba(features) >= threshold).astype(np.int64)


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Fraction of predictions equal to the targets.

    Args:
        predictions: Predicted labels, shape ``(n,)``.
        targets: True labels, shape ``(n,)``.

    Returns:
        Accuracy in ``[0, 1]``. Returns ``0.0`` for empty input (guard).
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if predictions.size == 0:
        return 0.0
    return float(np.mean(predictions == targets))
