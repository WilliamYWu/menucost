"""
Tauchen (1986) discretization of an AR(1) process, in JAX.
Based on a version found on QuantEcon which was in numba and now converted to JAX.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import erfc

@jax.jit
def std_norm_cdf(y):
    return 0.5 * erfc(-y / jnp.sqrt(2))

@jax.jit
def fill_transition_matrix(y, n, rho, sigma, mu, step):
    def compute_row(yi):
        z = y - rho * yi
        p_first = std_norm_cdf((y[0] - rho * yi + step - mu) / sigma)
        p_last = 1 - std_norm_cdf((y[n - 1] - rho * yi - step - mu) / sigma)
        p_middle = std_norm_cdf((z + step - mu) / sigma) - std_norm_cdf((z - step - mu) / sigma)

        row = jnp.zeros(p_middle.shape[0])
        row = row.at[0].set(p_first)
        row = row.at[1:-1].set(p_middle[1:-1])
        row = row.at[-1].set(p_last)
        return row
    return jax.vmap(compute_row)(y)


def tauchen(n, rho, sigma, mu, n_std, preset_grid=None, tol=1e-8, maxiter=10000):
    if preset_grid is None:
        std = n_std * jnp.sqrt(sigma**2 / (1 - rho**2))
        center = mu / (1 - rho)
        y = jnp.linspace(start=center - std, stop=center + std, num=n, endpoint=True)
        step = std / (n - 1)
    else:
        y = preset_grid
        step = (y[1] - y[0]) / 2

    P = fill_transition_matrix(y, n, rho, sigma, mu, step)

    def body_fun(state):
        i, _, pie = state
        pie_next = pie @ P
        return i + 1, jnp.max(jnp.abs(pie_next - pie)), pie_next

    def cond_fun(state):
        i, diff, _ = state
        return jnp.logical_and(diff > tol, i < maxiter)

    pie0 = jnp.ones(n) / n
    _, _, pie = jax.lax.while_loop(cond_fun, body_fun, (0, 1.0, pie0))
    return y, P, pie
