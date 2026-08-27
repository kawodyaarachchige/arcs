"""JSON shapes for the HTTP API (what clients send and what we send back)."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class SuggestedActionBody(BaseModel):
    """What the model (or a static controller) wants to try next."""

    retry: int = Field(ge=0, description="How many retries before giving up.")
    backoff_multiplier: float = Field(ge=0.0, description="Multiplier applied between retry waits.")
    timeout_ms: float = Field(ge=0.0, description="Per-attempt timeout in milliseconds.")


class DecideRequest(BaseModel):
    """One decision call: which route, how unhealthy traffic looks, and a suggested policy."""

    route: str = Field(
        min_length=1,
        description="Stable id for the path or service (per-route limits).",
    )
    error_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Recent error fraction, 0.0 = perfect, 1.0 = all errors.",
    )
    suggested: SuggestedActionBody | None = Field(
        default=None,
        description="Required unless you send a 12-float observation while inference is enabled.",
    )
    observation: list[float] | None = Field(
        default=None,
        description=(
            "Twelve normalized features in training order; used with a loaded TorchScript model."
        ),
    )
    trace_id: str | None = Field(
        default=None,
        description="Optional id from tracing headers for log correlation.",
    )

    @model_validator(mode="after")
    def _need_source(self) -> DecideRequest:
        obs = self.observation
        if obs is not None and len(obs) not in (0, 12):
            msg = "observation must be omitted, empty, or have exactly 12 numbers"
            raise ValueError(msg)
        has_obs = obs is not None and len(obs) == 12
        if self.suggested is None and not has_obs:
            msg = "Send `suggested`, or a 12-number `observation` when inference is enabled."
            raise ValueError(msg)
        return self


class DecideResponse(BaseModel):
    """Final numbers after clamps / freeze / rate limits, plus reasons for operators."""

    retry: int
    backoff_multiplier: float
    timeout_ms: float
    override_reasons: list[str]
    frozen_active: bool
