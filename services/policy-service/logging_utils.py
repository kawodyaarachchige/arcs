"""Structured, human-readable lines for audit trails (easy to grep later)."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("arcs.policy")


def log_decision(payload: dict[str, Any]) -> None:
    """
    Emit one JSON line per decision.
    """
    logger.info(json.dumps(payload, separators=(",", ":"), sort_keys=True))
