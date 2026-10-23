from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .parameters import DecisionRule, ShockGrids, SimDraws


def draw_shocks(kp, mp, sg: ShockGrids, key) -> SimDraws:
    """Draw shocks once for reuse across KS iterations."""
    total = kp.nburn + kp.ntime
    _, f_key, z_key, pz_key, m_key = jax.random.split(key, 5)

    frand = jax.random.uniform(f_key, shape=(kp.nfirm, total))
    zrand = jax.random.uniform(z_key, shape=(kp.nfirm, total))
    pzrand = jax.random.uniform(pz_key, shape=(kp.nfirm, total))
    mrand = jax.random.uniform(m_key, shape=(1, total))[0]

    # Initial productivity reuses the first column of zrand.
    z0 = jnp.minimum(
        jnp.searchsorted(jnp.cumsum(sg.zdist), zrand[:, 0], side="right"),
        mp.zn - 1,
    )
    m_idx = jnp.minimum(
        jnp.searchsorted(jnp.cumsum(sg.sdist), mrand, side="right"),
        mp.mn - 1,
    )

    return SimDraws(
        free_adjust=frand <= mp.α,
        zrand=zrand,
        pzrand=pzrand,
        z0=z0,
        msim=sg.mgrid[m_idx],
        ssim=sg.sgrid[m_idx],
    )


def log_price_index(price_idx, pgrid, θ):
    """Log CES index of normalized firm prices."""
    n_firms = price_idx.shape[0]
    return jnp.log(
        jnp.sum(jnp.exp(pgrid[price_idx]) ** (1 - θ) / n_firms, axis=0)
        ** (1 / (1 - θ))
    )


class PanelSimulation(NamedTuple):
    chi: jnp.ndarray        # (T,)
    log_P: jnp.ndarray      # (T,)
    price_idx: jnp.ndarray  # (T, N)
    start_idx: jnp.ndarray  # (T, N)
    adj_freq: jnp.ndarray   # (T,)


@partial(jax.jit, static_argnums=(0,))
def simulate_panel(mp, sg: ShockGrids, draws: SimDraws, rule: DecisionRule) -> PanelSimulation:
    """Simulate firm prices and the aggregate price index."""
    n_firms, n_periods = draws.zrand.shape
    n_prices = sg.pgrid.shape[0]
    n_z = sg.zgrid.shape[0]
    searchsorted_rows = jax.vmap(
        lambda cdf, u: jnp.searchsorted(cdf, u),
        in_axes=(0, 0),
    )

    def step(carry, t):
        z_prev, p_prev, P_prev = carry

        #NOTE: careful with pz
        cdf = jnp.cumsum(sg.Fz[z_prev, :], axis=-1)
        z_draw = jnp.minimum(searchsorted_rows(cdf, draws.zrand[:, t]), n_z - 1)
        z = jnp.where(
            draws.pzrand[:, t] <= mp.pz, z_draw, z_prev
        ).astype(z_prev.dtype)

        chi = P_prev - mp.μ - draws.ssim[t]
        chi_idx = jnp.argmin(jnp.abs(chi - sg.chigrid))

        p_start = jnp.clip(p_prev - 1 - draws.msim[t], 0, n_prices - 1)
        p_adjust = rule.optimal_price[chi_idx, p_start, z]
        p_adjust0 = rule.optimal0_price[chi_idx, z]
        p_new = jnp.where(
            draws.free_adjust[:, t], p_adjust0, p_adjust
        ).astype(p_prev.dtype)

        P = log_price_index(p_new, sg.pgrid, mp.θ)
        adj_freq = jnp.mean(p_new != p_start)
        return (z, p_new, P), (chi, P, p_new, p_start, adj_freq)

    p0 = jnp.full((n_firms,), jnp.argmin(jnp.abs(sg.pgrid)))
    P0 = log_price_index(p0, sg.pgrid, mp.θ)

    _, (chi, log_P, price_idx, start_idx, adj_freq) = jax.lax.scan(
        step, (draws.z0, p0, P0), jnp.arange(n_periods)
    )
    return PanelSimulation(
        chi=chi,
        log_P=log_P,
        price_idx=price_idx,
        start_idx=start_idx,
        adj_freq=adj_freq,
    )
