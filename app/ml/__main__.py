"""CLI entry point for the linear-probe demo (``python -m app.ml``).

Builds the toy dataset, trains the pure-numpy logistic-regression probe with
full-batch gradient descent, and prints the (decreasing) loss curve together
with **train and test** accuracy. Everything runs offline with a fixed seed, so
repeated runs print identical numbers.

Run with::

    python -m app.ml [--seed INT] [--dim INT] [--lr FLOAT] [--epochs INT]
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.core import get_logger
from app.ml.dataset import build_dataset
from app.ml.probe import LogisticProbe, accuracy

logger = get_logger(__name__)

_DEFAULT_SEED = 42
_DEFAULT_DIM = 256
_DEFAULT_LR = 0.5
_DEFAULT_EPOCHS = 300
# How many loss checkpoints to print from the full history.
_LOSS_CHECKPOINTS = 6


def _sample_loss_curve(history: list[float], checkpoints: int) -> list[tuple[int, float]]:
    """Pick evenly-spaced ``(epoch, loss)`` points from a loss history.

    Args:
        history: Per-epoch loss values.
        checkpoints: How many points to sample (always includes first and
            last).

    Returns:
        A list of ``(epoch, loss)`` pairs in ascending epoch order.
    """
    if not history:
        return []
    if len(history) <= checkpoints:
        return list(enumerate(history))
    step = (len(history) - 1) / (checkpoints - 1)
    indices = sorted({int(round(i * step)) for i in range(checkpoints)})
    return [(idx, history[idx]) for idx in indices]


def _format_report(
    history: list[float], train_acc: float, test_acc: float
) -> str:
    """Render the training report (loss curve + train/test accuracy).

    Args:
        history: Per-epoch loss history.
        train_acc: Accuracy on the training split.
        test_acc: Accuracy on the held-out test split.

    Returns:
        A human-readable multi-line report.
    """
    lines = [
        "=== Linear-probe demo (logistic regression, pure numpy, offline) ===",
        f"epochs        : {len(history)}",
        "loss curve (subsampled):",
    ]
    for epoch, loss in _sample_loss_curve(history, _LOSS_CHECKPOINTS):
        lines.append(f"  epoch {epoch:>4}: loss={loss:.4f}")
    lines.append(f"train accuracy: {train_acc:.3f}")
    lines.append(f"test  accuracy: {test_acc:.3f}")
    return "\n".join(lines)


async def _run(seed: int, dim: int, lr: float, epochs: int) -> tuple[list[float], float, float]:
    """Build the dataset, fit the probe, and score both splits.

    Args:
        seed: Seed for dataset generation and probe init.
        dim: Embedding dimension.
        lr: Learning rate.
        epochs: Number of training epochs.

    Returns:
        A ``(loss_history, train_accuracy, test_accuracy)`` triple.
    """
    dataset = await build_dataset(seed=seed, dim=dim)
    probe = LogisticProbe(seed=seed)
    history = probe.fit(dataset.x_train, dataset.y_train, lr=lr, epochs=epochs)
    train_acc = accuracy(probe.predict(dataset.x_train), dataset.y_train)
    test_acc = accuracy(probe.predict(dataset.x_test), dataset.y_test)
    return history, train_acc, test_acc


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse CLI arguments for the probe demo.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        The parsed argument namespace.
    """
    parser = argparse.ArgumentParser(prog="app.ml", description="Offline linear-probe demo")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--dim", type=int, default=_DEFAULT_DIM, help="Embedding dimension.")
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS, help="Training epochs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the probe demo, print the report and exit ``0``.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Always ``0`` (the demo has no failure gate).
    """
    args = _parse_args(argv)
    history, train_acc, test_acc = asyncio.run(
        _run(args.seed, args.dim, args.lr, args.epochs)
    )
    print(_format_report(history, train_acc, test_acc))
    logger.info(
        "probe demo done: train_acc=%.3f test_acc=%.3f final_loss=%.4f",
        train_acc,
        test_acc,
        history[-1] if history else float("nan"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
