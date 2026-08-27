// Sends HTTP requests through Envoy so every call hits ext_authz and the policy service.
// All knobs come from the environment — the Python runner sets them from benchmarks/config/default.yaml.

import http from "k6/http";
import { check } from "k6";

// Turn a JSON string like [{"duration":"10s","target":5}] into k6 stage objects.
function stagesFromEnv() {
  const raw = __ENV.K6_STAGES_JSON;
  if (!raw || raw.trim() === "") {
    // Sensible default if someone runs k6 by hand without the runner.
    return [{ duration: "30s", target: 10 }];
  }
  return JSON.parse(raw);
}

// Build the option block: either ramping virtual users or a fixed request rate.
function buildOptions() {
  const executor = (__ENV.K6_EXECUTOR || "ramping-vus").trim();

  if (executor === "constant-arrival-rate") {
    const rate = parseInt(__ENV.K6_RATE || "100", 10);
    const timeUnit = __ENV.K6_TIME_UNIT || "1s";
    const duration = __ENV.K6_DURATION || "30s";
    const pre = parseInt(__ENV.K6_PRE_ALLOCATED_VUS || "50", 10);
    const max = parseInt(__ENV.K6_MAX_VUS || "100", 10);
    return {
      scenarios: {
        steady_rps: {
          executor: "constant-arrival-rate",
          rate: rate,
          timeUnit: timeUnit,
          duration: duration,
          preAllocatedVUs: pre,
          maxVUs: max,
        },
      },
    };
  }

  // Classic ramp: load increases step by step so we can stress gradually.
  return {
    discardResponseBodies: true,
    stages: stagesFromEnv(),
  };
}

// constant-arrival-rate uses scenarios only; ramping uses top-level stages (see buildOptions).

export const options = buildOptions();

const targetUrl = __ENV.TARGET_URL || "http://127.0.0.1:10000/health/live";
const routeHeader = __ENV.ARC_ROUTE || "/bench";
const errorRateHeader = __ENV.ARC_ERROR_RATE || "0.0";
// Twelve numbers separated by commas — must match training feature count when testing adaptive mode.
const obsHeader = __ENV.ARC_OBS || "";

export default function () {
  const headers = {
    "x-arcs-route": routeHeader,
    "x-arcs-error-rate": errorRateHeader,
  };
  if (obsHeader.length > 0) {
    headers["x-arcs-obs"] = obsHeader;
  }

  const res = http.get(targetUrl, { headers });
  check(res, {
    "status is 200": (r) => r.status === 200,
  });
}
