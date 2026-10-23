import time
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from .parameters import CoefParameters, DecisionRule, ShockGrids
from .regression import Regression, law_of_motion
from .simulation import PanelSimulation, simulate_panel
from .tauchen import tauchen
from .vfi import VFIResult, policy_function, solve_vfi


def create_shock_grids(mp) -> ShockGrids:
    """Build productivity, spending-shock, price, and aggregate-state grids."""
    zgrid, Fz, zdist = tauchen(
        n=mp.zn,
        rho=mp.rhoz,
        sigma=mp.sigz,
        mu=0.0,
        n_std=mp.tauchen_spread,
    )

    smin = -(mp.mn - 1) / 2 * mp.μ
    sgrid = jnp.linspace(start=smin, stop=-smin, num=mp.mn, endpoint=True)
    _, _, sdist = tauchen(
        n=mp.mn,
        rho=0.0,
        sigma=mp.sigs,
        mu=0.0,
        n_std=mp.tauchen_spread,
        preset_grid=sgrid,
    )

    pend = mp.μ * (mp.pn - 1) / 2
    pgrid = jnp.linspace(start=-pend, stop=pend, num=mp.pn, endpoint=True)
    chigrid = jnp.linspace(start=mp.chi_min, stop=mp.chi_max, num=mp.chin, endpoint=True)
    mgrid = jnp.arange(mp.mn) - mp.mn // 2

    return ShockGrids(
        zgrid=zgrid,
        Fz=Fz,
        zdist=zdist,
        sgrid=sgrid,
        sdist=sdist,
        pgrid=pgrid,
        chigrid=chigrid,
        mgrid=mgrid,
    )


class KSPass(NamedTuple):
    index: int
    coef_used: CoefParameters
    coef_next: CoefParameters
    error: float
    vfi: VFIResult
    rule: DecisionRule
    sim: PanelSimulation
    fit: Regression
    seconds: float

def ks_pass(it, v0, coef, mp, vp, kp, sg, draws) -> KSPass:
    """Solve, simulate, regress, and update the forecasting rule once."""
    t0 = time.perf_counter()

    vfi = solve_vfi(v0, mp, vp, sg, coef)
    rule = policy_function(vfi.v, sg)
    sim = simulate_panel(mp, sg, draws, rule)

    chi = np.asarray(sim.chi)[kp.nburn:]
    log_P = np.asarray(sim.log_P)[kp.nburn:]
    fit = law_of_motion(chi, log_P)

    error = float(max(
        abs(fit.coef[0] - coef.gam0),
        abs((fit.coef[1] - coef.gam1) / coef.gam1),
    ))

    new = kp.damp * fit.coef + (1 - kp.damp) * coef.as_array()
    coef_next = CoefParameters(gam0=float(new[0]), gam1=float(new[1]))

    step = KSPass(
        index=it,
        coef_used=coef,
        coef_next=coef_next,
        error=error,
        vfi=vfi,
        rule=rule,
        sim=sim,
        fit=fit,
        seconds=time.perf_counter() - t0,
    )
    print(
        f"Rule used: ({step.coef_used.gam0:.5f}, {step.coef_used.gam1:.5f}) | "
        f"OLS: ({step.fit.coef[0]:.5f}, {step.fit.coef[1]:.5f}) | "
        f"R2 {step.fit.r2:.4f} | KS error {step.error:.2e} | "
        f"{step.seconds:.1f}s"
    )
    return step


class KSResult(NamedTuple):
    passes: list[KSPass]
    converged: bool

    @property
    def final(self) -> KSPass:
        return self.passes[-1]

def solve_ks(mp, vp, kp, sg, draws, coef0: CoefParameters) -> KSResult:
    """Iterate the forecasting rule until convergence or ks_maxiter."""
    initial_states = jnp.zeros((mp.chin, mp.pn, mp.zn, 3))
    coef, v = coef0, initial_states
    passes = []
    converged = False

    for it in range(1, kp.ks_maxiter + 1):
        step = ks_pass(it, v, coef, mp, vp, kp, sg, draws)
        passes.append(step)

        if step.error <= kp.ks_tol:
            converged = True
            break

        coef, v = step.coef_next, step.vfi.v
    print(f"Converged: {converged} | Passes: {len(passes)} | KS error: {passes[-1].error:.2e}")
    return KSResult(passes=passes, converged=converged)
