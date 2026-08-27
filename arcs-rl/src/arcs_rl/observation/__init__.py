"""
Turn raw metrics and policy settings into the fixed 12-number state for the learner.

Public objects are re-exported here so other packages can do
`from arcs_rl.observation import StateAggregator`.
"""

from arcs_rl.observation.aggregator import (
    PolicyContext,
    StateAggregator,
    aggregator_from_config,
    build_state_vector,
)
from arcs_rl.observation.features import FEATURE_NAMES, STATE_DIM

__all__ = [
    "FEATURE_NAMES",
    "STATE_DIM",
    "PolicyContext",
    "StateAggregator",
    "aggregator_from_config",
    "build_state_vector",
]
