from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


def _register_pytree(cls):
    """Register the dataclass fields as JAX pytree children."""
    names = tuple(cls.__dataclass_fields__)

    def flatten(obj):
        return tuple(getattr(obj, name) for name in names), None

    def unflatten(_aux, children):
        return cls(*children)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls


@_register_pytree
@dataclass(frozen=True)
class ModelParameters:
    rhoz: float
    sigz: float
    pz: float
    ω: float
    f: float
    α: float
    θ: float
    β: float
    μ: float
    sigs: float
    tauchen_spread: float
    zn: int
    pn: int
    chin: int
    mn: int
    chi_min: float = -0.05
    chi_max: float = 0.03


@_register_pytree
@dataclass(frozen=True)
class VFIParameters:
    vfi_tol: float
    vfi_maxiter: int


@_register_pytree
@dataclass(frozen=True)
class KSParameters:
    nfirm: int
    ntime: int
    nburn: int
    damp: float
    ks_tol: float
    ks_maxiter: int


@_register_pytree
@dataclass(frozen=True)
class CoefParameters:
    """Forecasting rule: log(P/S) = gam0 + gam1 * chi."""
    gam0: float
    gam1: float

    def as_array(self) -> np.ndarray:
        return np.array([self.gam0, self.gam1], dtype=float)


class ShockGrids(NamedTuple):
    zgrid: jnp.ndarray
    Fz: jnp.ndarray
    zdist: jnp.ndarray
    sgrid: jnp.ndarray
    sdist: jnp.ndarray
    pgrid: jnp.ndarray
    chigrid: jnp.ndarray
    mgrid: jnp.ndarray


class DecisionRule(NamedTuple):
    optimal_price: jnp.ndarray   # (chi, price, z)
    optimal0_price: jnp.ndarray  # (chi, z)


class SimDraws(NamedTuple):
    """Random draws held fixed across KS iterations."""
    free_adjust: jnp.ndarray
    zrand: jnp.ndarray
    pzrand: jnp.ndarray
    z0: jnp.ndarray
    msim: jnp.ndarray
    ssim: jnp.ndarray
