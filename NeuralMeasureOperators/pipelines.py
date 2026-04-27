# NeuralMeasureOperators/pipelines.py

from __future__ import annotations

from typing import Any, Tuple

from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .representations import DCT2DLowFreq, RBFClassMeanClassifier


def raw_logistic(
    C: float = 1.0,
    max_iter: int = 5000,
    random_state: int = 0,
) -> Pipeline:
    """
    Logistic regression on the raw finite coordinate field.
    """
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def pca_logistic(
    n_components: int = 8,
    C: float = 1.0,
    max_iter: int = 5000,
    random_state: int = 0,
) -> Pipeline:
    """
    PCA projection followed by logistic regression.
    """
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=random_state)),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def dct_logistic(
    n_components: int = 32,
    image_shape: Tuple[int, int] = (8, 8),
    C: float = 1.0,
    max_iter: int = 5000,
    random_state: int = 0,
) -> Pipeline:
    """
    Low-frequency 2D DCT projection followed by logistic regression.
    """
    return Pipeline(
        steps=[
            (
                "dct",
                DCT2DLowFreq(
                    image_shape=image_shape,
                    n_components=n_components,
                ),
            ),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )


def kernel_mean(
    gamma: Any = "scale",
    max_median_samples: int = 800,
    random_state: int = 0,
) -> Pipeline:
    """
    Empirical RBF class-kernel-mean classifier.
    """
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                RBFClassMeanClassifier(
                    gamma=gamma,
                    max_median_samples=max_median_samples,
                    random_state=random_state,
                ),
            ),
        ]
    )


def nystroem_ridge(
    n_components: int = 128,
    gamma: Any = 0.02,
    alpha: float = 1.0,
    random_state: int = 0,
) -> Pipeline:
    """
    Nyström finite-rank RBF approximation followed by a ridge classifier.
    """
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "features",
                Nystroem(
                    kernel="rbf",
                    gamma=gamma,
                    n_components=n_components,
                    random_state=random_state,
                ),
            ),
            ("clf", RidgeClassifier(alpha=alpha)),
        ]
    )


def exact_rbf_svm(
    C: float = 10.0,
    gamma: Any = "scale",
) -> Pipeline:
    """
    Exact RBF-kernel SVM baseline.
    """
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                SVC(
                    C=C,
                    gamma=gamma,
                    kernel="rbf",
                ),
            ),
        ]
    )
