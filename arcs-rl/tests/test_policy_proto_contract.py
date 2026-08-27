"""
Round-trip protobuf messages so field numbers stay aligned with the .proto file.

CI also runs ``scripts/gen_proto.sh`` and fails if generated stubs drift; this test catches
logic mistakes that still compile.
"""

from __future__ import annotations

from arcs.policy.v1 import policy_pb2


def test_decide_request_round_trip() -> None:
    req = policy_pb2.DecideRequest(
        route="/api/orders",
        error_rate=0.12,
        trace_id="trace-abc",
        observation=[0.1] * 12,
    )
    req.suggested.retry = 2
    req.suggested.backoff_multiplier = 1.5
    req.suggested.timeout_ms = 800.0

    blob = req.SerializeToString()
    req2 = policy_pb2.DecideRequest()
    req2.ParseFromString(blob)

    assert req2.route == req.route
    assert abs(req2.error_rate - 0.12) < 1e-9
    assert req2.trace_id == "trace-abc"
    assert list(req2.observation) == list(req.observation)
    assert req2.suggested.retry == 2


def test_decide_response_round_trip() -> None:
    res = policy_pb2.DecideResponse(
        retry=1,
        backoff_multiplier=2.0,
        timeout_ms=1500.0,
        override_reasons=["max_retries"],
        frozen_active=False,
    )
    blob = res.SerializeToString()
    res2 = policy_pb2.DecideResponse()
    res2.ParseFromString(blob)
    assert res2.retry == 1
    assert res2.override_reasons == ["max_retries"]
    assert res2.frozen_active is False


def test_suggested_action_standalone() -> None:
    s = policy_pb2.SuggestedAction(retry=3, backoff_multiplier=1.0, timeout_ms=500.0)
    b = s.SerializeToString()
    s2 = policy_pb2.SuggestedAction()
    s2.ParseFromString(b)
    assert s2.retry == 3
