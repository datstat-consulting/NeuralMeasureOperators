"""Finite feature measures and quadrature evaluation of kernel operators."""

from dataclasses import dataclass

import numpy as np


def points(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("Expected a finite nonempty array of points as rows.")
    return values


@dataclass(frozen=True)
class FiniteMeasure:
    """Sites, nonnegative integration weights and a feature vector at each site."""

    sites: np.ndarray
    features: np.ndarray
    weights: np.ndarray = None

    def __post_init__(self):
        sites, features = points(self.sites), points(self.features)
        weights = np.full(len(sites), 1 / len(sites)) if self.weights is None else np.asarray(
            self.weights, dtype=float
        )
        if len(features) != len(sites) or weights.shape != (len(sites),):
            raise ValueError("Provide one feature row and one weight per site.")
        if not np.all(np.isfinite(weights) & (weights >= 0)):
            raise ValueError("Weights must be finite and nonnegative.")
        for name, value in (("sites", sites), ("features", features), ("weights", weights)):
            copy = np.array(value, copy=True)
            copy.flags.writeable = False
            object.__setattr__(self, name, copy)

    def integrate(self):
        """Integrate the feature field against this measure."""
        return self.weights @ self.features


class KernelOperator:
    """Apply a scalar or matrix-valued kernel to a finite feature measure."""

    def __init__(self, kernel, *, normalize=False, block_size=512):
        if not isinstance(block_size, int) or block_size < 1:
            raise ValueError("block_size must be a positive integer.")
        self.kernel = kernel
        self.normalize = normalize
        self.block_size = block_size

    def __call__(self, measure, targets=None):
        targets = measure.sites if targets is None else points(targets)
        weighted = measure.weights[:, None] * measure.features
        blocks = []
        for start in range(0, len(targets), self.block_size):
            target = targets[start:start + self.block_size]
            kernel = np.asarray(self.kernel(target, measure.sites), dtype=float)
            if kernel.shape[:2] != (len(target), len(measure.sites)) or not np.isfinite(kernel).all():
                raise ValueError("Kernel must return finite values for each target/site pair.")
            if kernel.ndim == 2:
                if self.normalize:
                    mass = kernel @ measure.weights
                    if np.any(kernel < 0) or np.any(mass <= 0):
                        raise ValueError("Normalization requires nonnegative kernels and positive row mass.")
                    kernel = kernel / mass[:, None]
                blocks.append(kernel @ weighted)
            elif kernel.ndim == 4 and kernel.shape[-1] == weighted.shape[-1]:
                if self.normalize:
                    raise ValueError("Normalization requires a scalar kernel.")
                blocks.append(np.einsum("yxoi,xi->yo", kernel, weighted))
            else:
                raise ValueError("Expected kernel shape (target, site[, out_features, in_features]).")
        return np.concatenate(blocks)
