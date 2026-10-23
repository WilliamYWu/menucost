"""
Menu-cost model with Krusell-Smith aggregation in JAX.
"""
import jax

jax.config.update("jax_enable_x64", True)

from .parameters import (
    CoefParameters,
    DecisionRule,
    KSParameters,
    ModelParameters,
    ShockGrids,
    SimDraws,
    VFIParameters,
)
from .regression import Regression, law_of_motion
from .simulation import PanelSimulation, draw_shocks, simulate_panel
from .solver import (
    KSPass,
    KSResult,
    create_shock_grids,
    ks_pass,
    solve_ks,
)
from .vfi import VFIResult, policy_function, solve_vfi

__version__ = "0.1.0"
