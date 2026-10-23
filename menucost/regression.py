from typing import NamedTuple

import numpy as np


class Regression(NamedTuple):
    coef: np.ndarray
    stderr: np.ndarray
    r2: float
    n: int
    resid_std: float


def law_of_motion(chi, log_P) -> Regression:
    """OLS fit of log(P/S) on a constant and chi, with standard errors."""
    x = np.asarray(chi, dtype=float).ravel()
    y = np.asarray(log_P, dtype=float).ravel()
    X = np.column_stack([np.ones_like(x), x])

    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    n, k = X.shape

    ssr = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    s2 = ssr / (n - k)
    stderr = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))

    return Regression(
        coef=coef,
        stderr=stderr,
        r2=1.0 - ssr / sst,
        n=n,
        resid_std=float(np.sqrt(s2)),
    )
