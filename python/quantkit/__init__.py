"""Options pricing, Greeks, implied volatility, Monte Carlo, vol surfaces and risk metrics.

Thin Python bindings over the Rust `quantkit` crate. All functions validate their
inputs and raise `quantkit.QuantError` (a `ValueError`) on bad data.
"""

from ._quantkit import *  # noqa: F401,F403
from ._quantkit import QuantError, VolSurface, __version__  # noqa: F401
