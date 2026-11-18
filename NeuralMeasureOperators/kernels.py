"""Data-derived RBF kernels and sparse symbolic kernel regression."""

import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.base import BaseEstimator
from sklearn.linear_model import OrthogonalMatchingPursuit, Ridge
from sklearn.utils.validation import check_is_fitted

from .operators import points


class RBFKernel(BaseEstimator):
    """Fit an isotropic RBF kernel with the median positive pairwise distance."""

    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth

    def fit(self, sites, *, max_sites=2048, random_state=0):
        sites = points(sites)
        bandwidth = self.bandwidth
        if bandwidth is None:
            if not isinstance(max_sites, int) or max_sites < 2:
                raise ValueError("max_sites must be an integer of at least two.")
            if len(sites) > max_sites:
                indices = np.random.default_rng(random_state).choice(len(sites), max_sites, replace=False)
                sites = sites[indices]
            distances = pdist(sites)
            positive = distances[distances > 0]
            if not len(positive):
                raise ValueError("Bandwidth estimation needs two distinct sites.")
            bandwidth = np.median(positive)
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth must be finite and positive.")
        self.bandwidth_ = float(bandwidth)
        return self

    def __call__(self, targets, sites):
        check_is_fitted(self, "bandwidth_")
        squared = cdist(points(targets), points(sites), metric="sqeuclidean")
        return np.exp(-0.5 * squared / self.bandwidth_**2)


class SymbolicKernel(BaseEstimator):
    """Regress a scalar interaction on a sparse library of symmetric expressions."""

    def __init__(self, *, max_terms=3, ridge=1e-8):
        if not isinstance(max_terms, int) or not 1 <= max_terms <= 7:
            raise ValueError("max_terms must be an integer from 1 to 7.")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative.")
        self.max_terms = max_terms
        self.ridge = ridge

    terms = ("1", "dot", "dot^2", "distance^2", "distance^4",
             "exp(-distance^2)", "exp(-distance^2/4)")

    @staticmethod
    def _library(squared_distance, dot):
        return np.stack((np.ones_like(dot), dot, dot**2, squared_distance,
                         squared_distance**2, np.exp(-squared_distance),
                         np.exp(-squared_distance / 4)), axis=-1)

    def _regress(self, design, values):
        if not np.isfinite(design).all() or not np.any(design):
            raise ValueError("Kernel basis values must be finite with some nonzero information.")
        scale = np.linalg.norm(design, axis=0)
        scale[scale == 0] = 1
        normalized = design / scale
        informative = int(np.linalg.matrix_rank(normalized))
        if np.linalg.norm(normalized.T @ values) == 0:
            coefficients = np.zeros(len(self.terms))
        else:
            estimator = OrthogonalMatchingPursuit(
                n_nonzero_coefs=min(self.max_terms, informative), fit_intercept=False
            )
            # Fit sparse terms by residual reduction, then regularize the selected fit.
            estimator.fit(normalized, values)
            selected = np.flatnonzero(estimator.coef_)
            coefficients = np.zeros(len(self.terms))
            if len(selected):
                coefficients[selected] = Ridge(alpha=self.ridge, fit_intercept=False).fit(
                    normalized[:, selected], values
                ).coef_ / scale[selected]
        self.coef_ = coefficients
        return self

    def fit(self, left, right, observations):
        """Fit from observed scalar interactions for aligned pairs of points."""
        left, right = points(left), points(right)
        values = np.asarray(observations, dtype=float)
        if left.shape != right.shape or values.shape != (len(left),) or not np.isfinite(values).all():
            raise ValueError("Expected aligned point pairs and finite scalar observations.")
        return self._regress(self._library(((left - right)**2).sum(axis=1),
                                           (left * right).sum(axis=1)), values)

    def fit_operator(self, experiments):
        """Infer a shared kernel from (measure, targets, observed field) triples."""
        designs, outputs = [], []
        for measure, targets, field in experiments:
            targets = points(targets)
            field = points(field)
            if field.shape != (len(targets), measure.features.shape[1]):
                raise ValueError("Observed field shape does not match targets and input feature size.")
            basis = self._library(cdist(targets, measure.sites, metric="sqeuclidean"),
                                  targets @ measure.sites.T)
            weighted = measure.weights[:, None] * measure.features
            designs.append(np.einsum("yxt,xf->yft", basis, weighted).reshape(-1, len(self.terms)))
            outputs.append(field.ravel())
        if not designs:
            raise ValueError("Provide at least one operator experiment.")
        return self._regress(np.concatenate(designs), np.concatenate(outputs))

    @property
    def expression(self):
        check_is_fitted(self, "coef_")
        return " ".join(f"{coefficient:+.6g} * {term}" for term, coefficient in
                        zip(self.terms, self.coef_) if coefficient) or "0"

    def __call__(self, targets, sites):
        check_is_fitted(self, "coef_")
        targets, sites = points(targets), points(sites)
        basis = self._library(cdist(targets, sites, metric="sqeuclidean"), targets @ sites.T)
        return basis @ self.coef_
