from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .parameters import DecisionRule


@partial(jax.jit, static_argnums=(2, 6))
def continuation_value(v, P, mp, sg, cp, j, index_m):
    """Expected continuation value over spending, productivity, and menu-cost shocks."""
    def body_fun(carry, m):
        chiprime = P - (mp.μ + sg.sgrid[m])
        P_prime = cp.gam0 + chiprime * cp.gam1
        chiind = jnp.argmin(jnp.abs(chiprime - sg.chigrid))

        p_prime = jnp.clip(j - sg.mgrid[m] - 1, 0, mp.pn - 1)
        v_cont = v[chiind, p_prime, :, 0:2]

        fill_cont = (
            jnp.sum(mp.pz * sg.Fz[:, :, None] * v_cont[None, :, :], axis=1)
            + (1 - mp.pz) * v_cont
        )
        carry = carry + sg.sdist[m] * jnp.exp(P_prime) / jnp.exp(P) * (
            mp.α * fill_cont[:, 0] + (1 - mp.α) * fill_cont[:, 1]
        )
        return carry, None

    initial_cont = jnp.zeros(sg.zgrid.shape[0])
    cont, _ = jax.lax.scan(body_fun, initial_cont, jnp.arange(index_m))
    return cont


@partial(jax.jit, static_argnums=(1,))
def bellman(new_v, mp, sg, cp, i, j):
    """Values of keeping price j and choosing it after paying the menu cost."""
    index_m = sg.mgrid.shape[0]
    P = cp.gam0 + sg.chigrid[i] * cp.gam1
    fixed_real_cost = -mp.f * mp.ω / jnp.exp(P)

    cont = continuation_value(new_v, P, mp, sg, cp, j, index_m)

    profit = (
        (jnp.exp(sg.pgrid[j]) - mp.ω / jnp.exp(sg.zgrid).T)
        * jnp.exp(sg.pgrid[j]) ** (-mp.θ)
        * jnp.exp(P) ** (mp.θ - 2)
    )
    non_v = profit + mp.β * cont
    adj_v = non_v + fixed_real_cost
    return non_v, adj_v


def bellman_operator(v, mp, sg, cp):
    """Last axis of v: free adjustment, costly adjustment, no adjustment."""
    range_i = jnp.arange(sg.chigrid.shape[0])
    range_j = jnp.arange(sg.pgrid.shape[0])

    max_adjv = jnp.maximum(v[..., 1], v[..., 2])
    max_adj0v = jnp.maximum(v[..., 0], v[..., 2])
    new_v = jnp.stack([max_adj0v, max_adjv, v[..., 2]], axis=-1)

    B = jax.vmap(bellman, in_axes=(None, None, None, None, None, 0))
    B = jax.vmap(B, in_axes=(None, None, None, None, 0, None))
    non_v, adj_v = B(new_v, mp, sg, cp, range_i, range_j)

    new_adj_v = jnp.repeat(
        jnp.expand_dims(jnp.max(adj_v, axis=1), axis=1),
        sg.pgrid.shape[0],
        axis=1,
    )
    adj0_v = jnp.repeat(
        jnp.expand_dims(jnp.max(non_v, axis=1), axis=1),
        sg.pgrid.shape[0],
        axis=1,
    )
    return jnp.stack([adj0_v, new_adj_v, non_v], axis=-1)


@partial(jax.jit, static_argnums=(1,))
def _vfi_chunk(v, mp, sg, cp, tol, limit):
    def cond_fun(state):
        i, _, error = state
        return jnp.logical_and(error > tol, i < limit)

    def body_fun(state):
        i, v_old, _ = state
        v_new = bellman_operator(v_old, mp, sg, cp)
        max_diff = jnp.max(jnp.abs((v_new - v_old) / (0.5 * v_new + 0.5 * v_old)))
        return i + 1, v_new, max_diff

    init = (jnp.asarray(0), v, jnp.asarray(jnp.inf, dtype=v.dtype))
    n_iter, v_out, error = jax.lax.while_loop(cond_fun, body_fun, init)
    return n_iter, v_out, error


class VFIResult(NamedTuple):
    v: jnp.ndarray
    n_iter: int
    error: float
    converged: bool

def solve_vfi(v0, mp, vp, sg, cp):
    v, done, error = v0, 0, float("inf")

    while done < vp.vfi_maxiter and error > vp.vfi_tol:
        limit = min(100, vp.vfi_maxiter - done)
        n, v, err = _vfi_chunk(v, mp, sg, cp, vp.vfi_tol, limit)
        done += int(n)
        error = float(err)

        print(f"VFI iteration {done}: error = {error:.3e}")

    return VFIResult(
        v=v,
        n_iter=done,
        error=error,
        converged=error <= vp.vfi_tol,
    )


@jax.jit
def policy_function(v, sg):
    """Price indices after the decision for positive and zero menu costs."""
    max_index = jnp.expand_dims(v[..., 2].argmax(axis=1), axis=1)
    adj_sigma = jnp.where(v[..., 1] >= v[..., 2], 1, 0)
    adj0_sigma = jnp.where(v[..., 0] >= v[..., 2], 1, 0)

    adj_price = max_index * adj_sigma
    adj0_price = max_index * adj0_sigma

    dim_2 = jnp.arange(sg.pgrid.shape[0]).reshape(1, -1, 1)
    p_shape = (sg.chigrid.shape[0], 1, sg.zgrid.shape[0])
    p_grid_index = jnp.tile(dim_2, p_shape)

    non_adj_price = (1 - adj_sigma) * p_grid_index
    non_adj0_price = (1 - adj0_sigma) * p_grid_index

    optimal_price = adj_price + non_adj_price
    optimal0_price = adj0_price + non_adj0_price
    return DecisionRule(
        optimal_price=optimal_price,
        optimal0_price=optimal0_price[:, 0, :],
    )
