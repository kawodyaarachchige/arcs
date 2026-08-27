"""gRPC Decide round-trip against an in-process server (no Docker)."""

from __future__ import annotations

import os
import socket
from concurrent import futures
from pathlib import Path

import grpc
import pytest

from arcs.policy.v1 import policy_pb2, policy_pb2_grpc
from arcs_rl.config import load_config, validate_config_keys
from arcs_rl.inference.runtime import load_torch_inference_or_none
from arcs_rl.safeguards import SafeguardConfig, SafeguardEngine

REPO_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = int(s.getsockname()[1])
    s.close()
    return p


@pytest.fixture()
def grpc_stub():
    # Import policy service implementation from the sibling package (pytest pythonpath).
    from grpc_servicer import PolicyServicer  # noqa: E402

    os.environ["ARCS_CONFIG"] = str(REPO_ROOT / "configs" / "arcs.default.yaml")
    root = load_config(os.environ["ARCS_CONFIG"])
    validate_config_keys(root)
    cfg = SafeguardConfig.from_arcs_config(root)
    inference = load_torch_inference_or_none(root)
    runtime = type(
        "R",
        (),
        {
            "engine": SafeguardEngine(cfg),
            "inference": inference,
            "root": root,
        },
    )()
    port = _free_port()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    policy_pb2_grpc.add_PolicyServicer_to_server(
        PolicyServicer(root=root, runtime=runtime),
        server,
    )
    server.add_insecure_port(f"127.0.0.1:{port}")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = policy_pb2_grpc.PolicyStub(channel)
    try:
        yield stub
    finally:
        channel.close()
        server.stop(0)


def test_grpc_decide_ok(grpc_stub: policy_pb2_grpc.PolicyStub) -> None:
    req = policy_pb2.DecideRequest(
        route="/grpc-demo",
        error_rate=0.0,
        trace_id="t-grpc-1",
        suggested=policy_pb2.SuggestedAction(
            retry=2,
            backoff_multiplier=1.0,
            timeout_ms=2000.0,
        ),
    )
    resp = grpc_stub.Decide(req)
    assert resp.retry == 2
    assert resp.frozen_active is False


def test_grpc_decide_invalid_raises(grpc_stub: policy_pb2_grpc.PolicyStub) -> None:
    req = policy_pb2.DecideRequest(route="/bad", error_rate=0.0)
    with pytest.raises(grpc.RpcError) as ei:
        grpc_stub.Decide(req)
    assert ei.value.code() == grpc.StatusCode.INVALID_ARGUMENT
