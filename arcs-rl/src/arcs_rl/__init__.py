"""Python package for ARCS: config, observation, training, and replay storage."""

from arcs_rl.observation import (
    FEATURE_NAMES,
    STATE_DIM,
    PolicyContext,
    StateAggregator,
    aggregator_from_config,
    build_state_vector,
)

__version__ = "0.1.0"

__all__ = [
    "FEATURE_NAMES",
    "STATE_DIM",
    "PolicyContext",
    "StateAggregator",
    "aggregator_from_config",
    "build_state_vector",
    "__version__",
]
