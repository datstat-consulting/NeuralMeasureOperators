import numpy as np

from NeuralMeasureOperators import (
    FiniteMeasure, KernelOperator, RBFKernel, SymbolicKernel, TokenMeasureEncoder,
)


def test_weighted_operator_and_token_positions():
    encoder = TokenMeasureEncoder().fit(["x y x", "z"])
    measure = encoder.encode("x z x", weights=[0.1, 0.2, 0.7])
    np.testing.assert_allclose(measure.sites[:, 0], [0, 0.5, 1])
    assert np.isclose(measure.integrate()[encoder.vocabulary_["x"]], 0.8)
    kernel = RBFKernel().fit(measure.sites)
    result = KernelOperator(kernel, normalize=True, block_size=1)(measure)
    np.testing.assert_allclose(result.sum(axis=1), 1)
    matrix = KernelOperator(lambda y, x: np.broadcast_to(
        np.eye(measure.features.shape[1]), (len(y), len(x), 4, 4)
    ))(measure)
    np.testing.assert_allclose(matrix, np.tile(measure.integrate(), (3, 1)))


def test_symbolic_regression_recovers_kernel_from_aggregates():
    rng = np.random.default_rng(7)
    experiment = []
    targets = rng.normal(size=(25, 1))
    truth = KernelOperator(lambda y, x: np.exp(-(y - x.T)**2))
    for _ in range(4):
        measure = FiniteMeasure(rng.normal(size=6), rng.normal(size=6))
        experiment.append((measure, targets, truth(measure, targets)))
    kernel = SymbolicKernel(max_terms=1).fit_operator(experiment)
    reconstructed = KernelOperator(kernel)(measure, targets)
    np.testing.assert_allclose(reconstructed, truth(measure, targets), atol=1e-7)
    assert "exp(-distance^2)" in kernel.expression
