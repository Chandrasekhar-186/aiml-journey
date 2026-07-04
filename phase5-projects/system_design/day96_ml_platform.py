# Phase 5 Day 1 — System Design
# Date: July 4, 2026
# ML Platform at Scale — Databricks style!

print("="*60)
print("System Design: ML Platform at Scale")
print("="*60)

"""
DATABRICKS SYSTEM DESIGN INTEL:

From 2026 interview report:
→ System Design: adding delete + trash
  to database server
→ Write RUNNING PSEUDOCODE (not just diagrams!)
→ Low-level design: like a CS mini-project
→ Production optimization strategies

Two types asked:
1. High-level: architecture diagrams
2. Low-level: write actual working code!

Today: practice BOTH types on ML problems

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM DESIGN 1: ML EXPERIMENT TRACKER
(similar to MLflow — design it from scratch!)

Requirements:
- Log experiments (params, metrics, artifacts)
- Compare experiments
- Tag runs
- Retrieve by ID or name
- Scale to 1M+ runs

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM DESIGN 2: FEATURE STORE
(like Databricks Feature Store)

Requirements:
- Register feature tables
- Point-in-time lookups (no data leakage!)
- Online serving (low latency)
- Offline serving (batch)
- Feature lineage tracking

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SYSTEM DESIGN 3: MODEL SERVING SYSTEM
(like Databricks Model Serving)

Requirements:
- Register + version models
- A/B traffic splitting
- Auto-scaling
- Health checks
- Rollback capability
- SLA monitoring
"""

# ─────────────────────────────────────────
# LOW-LEVEL DESIGN 1: ML Experiment Tracker
# Write RUNNING CODE (Databricks style!)
# ─────────────────────────────────────────
print("\n=== LOW-LEVEL: ML Experiment Tracker ===")

import uuid
import time
import json
from collections import defaultdict
from typing import Any, Dict, List, Optional

class ExperimentRun:
    def __init__(self, run_id: str,
                  experiment_id: str,
                  run_name: str):
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.run_name = run_name
        self.status = "RUNNING"
        self.params: Dict[str, Any] = {}
        self.metrics: Dict[str, List] = {}
        self.tags: Dict[str, str] = {}
        self.artifacts: List[str] = []
        self.start_time = time.time()
        self.end_time: Optional[float] = None

    def log_param(self, key: str, value: Any):
        self.params[key] = value

    def log_metric(self, key: str,
                    value: float,
                    step: int = 0):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append({
            "value": value,
            "step": step,
            "timestamp": time.time()
        })

    def log_artifact(self, path: str):
        self.artifacts.append(path)

    def set_tag(self, key: str, value: str):
        self.tags[key] = value

    def finish(self, status: str = "FINISHED"):
        self.status = status
        self.end_time = time.time()

    def get_latest_metric(self,
                           key: str
                           ) -> Optional[float]:
        if key not in self.metrics or \
           not self.metrics[key]:
            return None
        return self.metrics[key][-1]["value"]

    def duration_seconds(self) -> float:
        end = self.end_time or time.time()
        return end - self.start_time


class ExperimentTracker:
    """
    Low-level ML experiment tracker.
    Write running code in Databricks interview!
    """
    def __init__(self):
        # experiment_name → experiment_id
        self._experiments: Dict[str, str] = {}
        # experiment_id → list of run_ids
        self._exp_runs: Dict[
            str, List[str]
        ] = defaultdict(list)
        # run_id → ExperimentRun
        self._runs: Dict[
            str, ExperimentRun
        ] = {}
        self._active_run: Optional[
            ExperimentRun
        ] = None

    def create_experiment(self,
                           name: str) -> str:
        if name in self._experiments:
            return self._experiments[name]
        exp_id = str(uuid.uuid4())[:8]
        self._experiments[name] = exp_id
        return exp_id

    def start_run(self,
                   experiment_name: str,
                   run_name: str = ""
                   ) -> ExperimentRun:
        exp_id = self.create_experiment(
            experiment_name
        )
        run_id = str(uuid.uuid4())[:12]
        run = ExperimentRun(
            run_id, exp_id,
            run_name or f"run_{run_id[:6]}"
        )
        self._runs[run_id] = run
        self._exp_runs[exp_id].append(run_id)
        self._active_run = run
        return run

    def end_run(self,
                 status: str = "FINISHED"):
        if self._active_run:
            self._active_run.finish(status)
            self._active_run = None

    def get_run(self,
                 run_id: str
                 ) -> Optional[ExperimentRun]:
        return self._runs.get(run_id)

    def search_runs(
        self,
        experiment_name: str,
        filter_str: str = "",
        order_by: str = "start_time"
    ) -> List[ExperimentRun]:
        exp_id = self._experiments.get(
            experiment_name
        )
        if not exp_id:
            return []

        runs = [
            self._runs[rid]
            for rid in self._exp_runs[exp_id]
            if rid in self._runs
        ]

        # Apply filter (simplified)
        if filter_str:
            # e.g. "accuracy > 0.9"
            parts = filter_str.split()
            if len(parts) == 3:
                metric, op, val = parts
                threshold = float(val)
                filtered = []
                for r in runs:
                    m = r.get_latest_metric(
                        metric
                    )
                    if m is None:
                        continue
                    if op == ">" and \
                       m > threshold:
                        filtered.append(r)
                    elif op == ">=" and \
                         m >= threshold:
                        filtered.append(r)
                    elif op == "<" and \
                         m < threshold:
                        filtered.append(r)
                runs = filtered

        return sorted(
            runs,
            key=lambda r: getattr(
                r, order_by, r.start_time
            ),
            reverse=True
        )

    def compare_runs(self,
                      run_ids: List[str],
                      metric: str) -> List[dict]:
        results = []
        for rid in run_ids:
            run = self._runs.get(rid)
            if run:
                results.append({
                    "run_id": rid,
                    "run_name": run.run_name,
                    metric: run.get_latest_metric(
                        metric
                    ),
                    "params": run.params
                })
        return sorted(
            results,
            key=lambda x: x.get(metric) or 0,
            reverse=True
        )

# Demo the tracker!
tracker = ExperimentTracker()

# Run 1: GBT model
run1 = tracker.start_run(
    "classification_experiment",
    "GBT_baseline"
)
run1.log_param("model", "GBT")
run1.log_param("n_estimators", 100)
run1.log_param("learning_rate", 0.1)
for step, acc in enumerate(
    [0.75, 0.82, 0.87, 0.91]
):
    run1.log_metric("accuracy", acc, step)
run1.log_metric("f1", 0.89)
run1.set_tag("framework", "sklearn")
run1.log_artifact("models/gbt_v1.pkl")
tracker.end_run("FINISHED")
run1_id = run1.run_id

# Run 2: RF model
run2 = tracker.start_run(
    "classification_experiment",
    "RF_comparison"
)
run2.log_param("model", "RF")
run2.log_param("n_estimators", 50)
for step, acc in enumerate(
    [0.72, 0.79, 0.83, 0.86]
):
    run2.log_metric("accuracy", acc, step)
run2.log_metric("f1", 0.85)
tracker.end_run("FINISHED")
run2_id = run2.run_id

# Run 3: failed run
run3 = tracker.start_run(
    "classification_experiment",
    "XGB_failed"
)
run3.log_param("model", "XGB")
run3.log_metric("accuracy", 0.45)
tracker.end_run("FAILED")

# Test all operations
print("All runs in experiment:")
runs = tracker.search_runs(
    "classification_experiment"
)
for r in runs:
    acc = r.get_latest_metric("accuracy")
    print(f"  {r.run_name}: "
          f"acc={acc:.3f} "
          f"status={r.status} "
          f"({r.duration_seconds():.3f}s)")

print("\nRuns with accuracy > 0.85:")
filtered = tracker.search_runs(
    "classification_experiment",
    "accuracy > 0.85"
)
for r in filtered:
    print(f"  {r.run_name}: "
          f"{r.get_latest_metric('accuracy'):.3f}")

print("\nCompare runs by accuracy:")
comparison = tracker.compare_runs(
    [run1_id, run2_id], "accuracy"
)
for c in comparison:
    print(f"  {c['run_name']}: "
          f"acc={c['accuracy']:.3f} "
          f"params={c['params']}")

# ─────────────────────────────────────────
# LOW-LEVEL DESIGN 2: Rate Limiter
# (Asked in system design rounds!)
# ─────────────────────────────────────────
print("\n=== LOW-LEVEL: Rate Limiter ===")

import time
from collections import deque

class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.
    Use for: ML API endpoints!
    Prevents: overloading model serving
    """
    def __init__(self, max_requests: int,
                  window_seconds: float):
        self.max_requests = max_requests
        self.window = window_seconds
        # user_id → deque of timestamps
        self.requests: Dict[
            str, deque
        ] = defaultdict(deque)

    def is_allowed(self, user_id: str) -> bool:
        now = time.time()
        user_requests = self.requests[user_id]

        # Remove expired timestamps
        while user_requests and \
              user_requests[0] < now - self.window:
            user_requests.popleft()

        if len(user_requests) < self.max_requests:
            user_requests.append(now)
            return True
        return False

    def get_wait_time(self,
                       user_id: str) -> float:
        """How long until next request allowed"""
        user_requests = self.requests[user_id]
        if not user_requests:
            return 0
        oldest = user_requests[0]
        return max(
            0,
            oldest + self.window - time.time()
        )

# Test rate limiter
limiter = SlidingWindowRateLimiter(
    max_requests=3,
    window_seconds=1.0
)
user = "user_123"
print(f"Rate limiter (3 req/sec):")
for i in range(5):
    allowed = limiter.is_allowed(user)
    status = "✅ ALLOWED" if allowed \
             else "❌ BLOCKED"
    print(f"  Request {i+1}: {status}")

print("\nExperiment Tracker + Rate Limiter built!")
print("Low-level system design = MASTERED! 🎯")
