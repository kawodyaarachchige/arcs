from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SuggestedAction(_message.Message):
    __slots__ = ("retry", "backoff_multiplier", "timeout_ms")
    RETRY_FIELD_NUMBER: _ClassVar[int]
    BACKOFF_MULTIPLIER_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    retry: int
    backoff_multiplier: float
    timeout_ms: float
    def __init__(self, retry: _Optional[int] = ..., backoff_multiplier: _Optional[float] = ..., timeout_ms: _Optional[float] = ...) -> None: ...

class DecideRequest(_message.Message):
    __slots__ = ("route", "error_rate", "suggested", "trace_id", "observation")
    ROUTE_FIELD_NUMBER: _ClassVar[int]
    ERROR_RATE_FIELD_NUMBER: _ClassVar[int]
    SUGGESTED_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_FIELD_NUMBER: _ClassVar[int]
    route: str
    error_rate: float
    suggested: SuggestedAction
    trace_id: str
    observation: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, route: _Optional[str] = ..., error_rate: _Optional[float] = ..., suggested: _Optional[_Union[SuggestedAction, _Mapping]] = ..., trace_id: _Optional[str] = ..., observation: _Optional[_Iterable[float]] = ...) -> None: ...

class DecideResponse(_message.Message):
    __slots__ = ("retry", "backoff_multiplier", "timeout_ms", "override_reasons", "frozen_active")
    RETRY_FIELD_NUMBER: _ClassVar[int]
    BACKOFF_MULTIPLIER_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    OVERRIDE_REASONS_FIELD_NUMBER: _ClassVar[int]
    FROZEN_ACTIVE_FIELD_NUMBER: _ClassVar[int]
    retry: int
    backoff_multiplier: float
    timeout_ms: float
    override_reasons: _containers.RepeatedScalarFieldContainer[str]
    frozen_active: bool
    def __init__(self, retry: _Optional[int] = ..., backoff_multiplier: _Optional[float] = ..., timeout_ms: _Optional[float] = ..., override_reasons: _Optional[_Iterable[str]] = ..., frozen_active: bool = ...) -> None: ...
