# NeuralMeasureOperators/representations.py

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.metrics.pairwise import rbf_kernel

try:
    from scipy.fft import dctn, idctn
except ImportError:
    dctn = None
    idctn = None


class RBFClassMeanClassifier(BaseEstimator, ClassifierMixin):
    """
    Empirical RBF class-kernel-mean classifier.

    Each class c is represented by the empirical measure

        mu_c = (1 / n_c) sum_{i:y_i=c} delta_{x_i}.

    Given an RBF kernel

        k(x, z) = exp(-gamma ||x - z||^2),

    the class score is

        s_c(x) = integral k(x, z) dmu_c(z)
               = (1 / n_c) sum_{i:y_i=c} k(x, x_i).

    Prediction is argmax_c s_c(x).
    """

    def __init__(
        self,
        gamma: Any = "scale",
        max_median_samples: int = 800,
        random_state: int = 0,
    ) -> None:
        self.gamma = gamma
        self.max_median_samples = max_median_samples
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBFClassMeanClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")

        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]

        if self.gamma == "scale":
            var = float(X.var())
            self.gamma_ = 1.0 / (X.shape[1] * var) if var > 0 else 1.0
        elif self.gamma == "median":
            self.gamma_ = self._median_gamma(X)
        else:
            self.gamma_ = float(self.gamma)

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, n_features).")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        K = rbf_kernel(X, self.X_, gamma=self.gamma_)

        scores = []
        for cls in self.classes_:
            mask = self.y_ == cls
            scores.append(K[:, mask].mean(axis=1))

        return np.column_stack(scores)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def _median_gamma(self, X: np.ndarray) -> float:
        rng = np.random.default_rng(self.random_state)

        m = min(int(self.max_median_samples), len(X))
        if m <= 1:
            return 1.0

        idx = rng.choice(len(X), size=m, replace=False)
        Xs = X[idx]

        d2 = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(axis=2)
        vals = d2[d2 > 0]

        if len(vals) == 0:
            return 1.0

        med = float(np.median(vals))
        return 1.0 / med if med > 0 else 1.0

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "X_"):
            raise RuntimeError("RBFClassMeanClassifier has not been fitted.")


class DCT2DLowFreq(BaseEstimator, TransformerMixin):
    """
    Low-frequency 2D DCT representation for flattened image fields.

    Input shape:

        (n_samples, height * width)

    Output shape:

        (n_samples, n_components)

    Coefficients are ordered by increasing spatial frequency i + j.
    """

    def __init__(
        self,
        image_shape: Tuple[int, int] = (8, 8),
        n_components: int = 16,
    ) -> None:
        self.image_shape = image_shape
        self.n_components = n_components

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> "DCT2DLowFreq":
        if dctn is None:
            raise ImportError("scipy is required for DCT2DLowFreq.")

        X = np.asarray(X, dtype=float)
        h, w = self.image_shape
        total = h * w

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, height * width).")

        if X.shape[1] != total:
            raise ValueError(f"Expected {total} features, got {X.shape[1]}.")

        if not 1 <= int(self.n_components) <= total:
            raise ValueError(f"n_components must be between 1 and {total}.")

        coords = [(i, j) for i in range(h) for j in range(w)]
        self.coords_ = sorted(
            coords,
            key=lambda ij: (ij[0] + ij[1], ij[0], ij[1]),
        )[: int(self.n_components)]

        self.n_features_in_ = total

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()

        if dctn is None:
            raise ImportError("scipy is required for DCT2DLowFreq.")

        X = np.asarray(X, dtype=float)
        h, w = self.image_shape

        if X.ndim != 2:
            raise ValueError("X must be a 2D array with shape (n_samples, height * width).")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        imgs = X.reshape(-1, h, w)
        Z = np.empty((len(imgs), len(self.coords_)), dtype=float)

        for n, img in enumerate(imgs):
            coeff = dctn(img, type=2, norm="ortho")
            Z[n] = [coeff[i, j] for i, j in self.coords_]

        return Z

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        self._check_is_fitted()

        if idctn is None:
            raise ImportError("scipy is required for inverse DCT.")

        Z = np.asarray(Z, dtype=float)
        h, w = self.image_shape

        if Z.ndim != 2:
            raise ValueError("Z must be a 2D array.")

        if Z.shape[1] != len(self.coords_):
            raise ValueError(
                f"Expected {len(self.coords_)} DCT coefficients, got {Z.shape[1]}."
            )

        out = np.zeros((Z.shape[0], h, w), dtype=float)

        for n, row in enumerate(Z):
            coeff = np.zeros((h, w), dtype=float)
            for value, (i, j) in zip(row, self.coords_):
                coeff[i, j] = value
            out[n] = idctn(coeff, type=2, norm="ortho")

        return out.reshape(Z.shape[0], h * w)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "coords_"):
            raise RuntimeError("DCT2DLowFreq has not been fitted.")
