import numpy as np


def assert_allclose(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-12, rtol: float = 0.0) -> None:
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=atol, rtol=rtol), (a, b)
