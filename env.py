from typing import Optional, Dict, Any, List, Tuple
from models import SREObservation, SREAction, SREReward


TASKS = {
    "cpu_spike": {
        "name": "CPU Spike Investigation",
        "difficulty": "easy",
        "description": (
            "An api-server is throwing high-CPU alerts. Investigate the symptoms, "
            "identify the root cause, and recommend the correct remediation action."
        ),
        "severity_answer": "high",
        "root_cause_keywords": [
            "traffic", "cpu", "load", "spike", "rate limit", "scale", "overload",
            "request", "rps", "throttl",
        ],
        "action_keywords": [
            "scale", "replica", "rate limit", "throttl", "autoscal", "horizontal",
            "rate-limit", "limit",
        ],
        "initial_alerts": [
            "HIGH CPU ALERT: api-server CPU usage at 95% for 5 minutes",
            "WARNING: api-server response time P99 increased to 4.2s (threshold: 1s)",
            "INFO: api-server error rate at 2.3% (threshold: 5%)",
        ],
        "initial_logs": [
            "2024-01-15 14:23:01 INFO  api-server: Request received GET /products",
            "2024-01-15 14:23:01 INFO  api-server: Request received GET /products",
            "2024-01-15 14:23:01 INFO  api-server: Request received POST /cart",
            "2024-01-15 14:23:02 WARN  api-server: High request queue depth: 847 pending",
            "2024-01-15 14:23:03 WARN  api-server: Worker threads at capacity (32/32)",
            "2024-01-15 14:23:05 ERROR api-server: Request timeout after 5000ms for GET /search",
            "2024-01-15 14:23:05 INFO  load-balancer: Incoming RPS: 12,450 (baseline: 2,100)",
        ],
        "initial_metrics": {
            "api-server": {
                "cpu_percent": 95,
                "memory_percent": 62,
                "error_rate": 2.3,
                "rps": 12450,
                "p99_latency_ms": 4200,
            },
            "database": {
                "cpu_percent": 45,
                "memory_percent": 71,
                "connections": 89,
                "query_time_ms": 12,
            },
            "cache": {"cpu_percent": 22, "memory_percent": 55, "hit_rate": 0.89},
        },
        "investigation_data": {
            "api-server": [
                "2024-01-15 14:20:00 INFO  nginx: Traffic spike detected from /api/products endpoint",
                "2024-01-15 14:20:05 INFO  api-server: No rate limiting configured for /api/products",
                "2024-01-15 14:20:10 WARN  api-server: Single IP 203.45.67.89 making 8,000 req/min",
                "2024-01-15 14:22:00 INFO  auto-scaler: Scaling triggered but max replicas (3) already reached",
                "2024-01-15 14:22:30 ERROR api-server: OOM risk - heap at 91%",
            ],
            "database": [
                "2024-01-15 14:23:01 INFO  postgres: Connection pool healthy (89/200)",
                "2024-01-15 14:23:01 INFO  postgres: Query cache hit rate: 94%",
                "2024-01-15 14:23:01 INFO  postgres: No slow queries detected",
            ],
            "load-balancer": [
                "2024-01-15 14:19:55 INFO  lb: RPS jumped from 2100 to 12450 in 60 seconds",
                "2024-01-15 14:20:00 INFO  lb: All traffic destined to /api/products (scraper pattern)",
                "2024-01-15 14:20:01 WARN  lb: No rate-limiting rules active for this endpoint",
            ],
        },
    },

    "cascading_failure": {
        "name": "Cascading Service Failure",
        "difficulty": "medium",
        "description": (
            "Multiple services are reporting errors simultaneously. "
            "Identify the root service that triggered the cascade and recommend a rollback strategy."
        ),
        "severity_answer": "critical",
        "root_cause_keywords": [
            "payment", "database", "db", "connection", "pool", "deploy",
            "rollback", "exhausted", "config", "v2.3.1",
        ],
        "action_keywords": [
            "rollback", "revert", "payment-service", "restart", "connection pool",
            "v2.3.0", "previous version",
        ],
        "initial_alerts": [
            "CRITICAL: payment-service health check failing (0% healthy pods)",
            "HIGH: checkout-service error rate at 78% — dependency on payment-service",
            "HIGH: order-service queue backing up — 12,000 unprocessed orders",
            "MEDIUM: user-service latency increased to 890ms (normal: 120ms)",
        ],
        "initial_logs": [
            "2024-01-15 16:45:01 ERROR payment-service: Failed to acquire DB connection (pool exhausted)",
            "2024-01-15 16:45:01 ERROR payment-service: Failed to acquire DB connection (pool exhausted)",
            "2024-01-15 16:45:02 ERROR checkout-service: payment-service call timed out after 3000ms",
            "2024-01-15 16:45:02 ERROR checkout-service: No circuit breaker configured, returning 503",
            "2024-01-15 16:45:03 WARN  order-service: Payment unavailable, queueing order #98234",
            "2024-01-15 16:44:30 INFO  deployment: payment-service v2.3.1 rolled out successfully",
            "2024-01-15 16:44:31 WARN  payment-service: Connection pool warming up with 500 connections",
        ],
        "initial_metrics": {
            "payment-service": {
                "cpu_percent": 12,
                "memory_percent": 88,
                "error_rate": 100,
                "healthy_pods": 0,
                "db_connections_open": 500,
            },
            "checkout-service": {
                "cpu_percent": 34,
                "memory_percent": 61,
                "error_rate": 78,
                "rps": 450,
            },
            "order-service": {
                "cpu_percent": 41,
                "memory_percent": 55,
                "queue_depth": 12000,
            },
            "user-service": {
                "cpu_percent": 28,
                "memory_percent": 52,
                "error_rate": 8,
                "p99_latency_ms": 890,
            },
            "payment-db": {
                "cpu_percent": 98,
                "active_connections": 500,
                "max_connections": 100,
                "wait_queue_depth": 400,
            },
        },
        "investigation_data": {
            "payment-service": [
                "2024-01-15 16:44:31 ERROR payment-service: DB pool config in v2.3.1: pool_size=500 (was 10)",
                "2024-01-15 16:44:31 ERROR payment-db: Max connections (100) exceeded, rejecting new connections",
                "2024-01-15 16:44:32 ERROR payment-service: All 500 connection attempts failed — DB at capacity",
                "2024-01-15 16:44:32 INFO  git: payment-service v2.3.1 changed DB_POOL_SIZE env var from 10 to 500",
                "2024-01-15 16:44:33 CRIT  payment-service: Cannot process any transactions",
            ],
            "checkout-service": [
                "2024-01-15 16:45:01 INFO  checkout-service: Calling payment-service for order #98234",
                "2024-01-15 16:45:01 ERROR checkout-service: payment-service returned 503",
                "2024-01-15 16:45:01 INFO  checkout-service: No circuit breaker or fallback configured",
            ],
            "payment-db": [
                "2024-01-15 16:44:31 ERROR postgres: connection limit 100 reached, refusing connections",
                "2024-01-15 16:44:31 ERROR postgres: 400 clients waiting in connection queue",
                "2024-01-15 16:44:35 INFO  postgres: CPU 98% due to connection overhead, not query load",
            ],
        },
    },

    "memory_leak": {
        "name": "Memory Leak Detection",
        "difficulty": "hard",
        "description": (
            "Multiple unrelated services are experiencing OOM kills over a 2-hour window. "
            "Find the shared root cause across all affected services."
        ),
        "severity_answer": "high",
        "root_cause_keywords": [
            "memory", "leak", "middleware", "logging", "shared", "library",
            "oom", "gradual", "buffer", "v2.1.0", "async",
        ],
        "action_keywords": [
            "rollback", "revert", "middleware", "logging", "shared library",
            "v2.0", "previous version", "logging-middleware",
        ],
        "initial_alerts": [
            "WARNING: api-server pod OOMKilled and restarted 3 times in last 2 hours",
            "WARNING: worker-service pod OOMKilled and restarted 2 times in last 2 hours",
            "WARNING: notification-service pod OOMKilled 4 times in last 2 hours",
            "INFO: auth-service memory trending up (71% now vs 45% two hours ago)",
        ],
        "initial_logs": [
            "2024-01-15 14:00:01 INFO  k8s: api-server OOMKilled — restarting pod",
            "2024-01-15 14:31:22 INFO  k8s: worker-service OOMKilled — restarting pod",
            "2024-01-15 15:02:44 INFO  k8s: notification-service OOMKilled — restarting pod",
            "2024-01-15 15:33:11 INFO  k8s: api-server OOMKilled — restarting pod",
            "2024-01-15 12:00:00 INFO  deployment: logging-middleware v2.1.0 deployed to all services",
            "2024-01-15 14:00:00 WARN  api-server: Memory at 95%, GC pressure increasing",
            "2024-01-15 14:30:00 WARN  worker-service: Memory at 95%, GC pressure increasing",
        ],
        "initial_metrics": {
            "api-server": {
                "cpu_percent": 34,
                "memory_percent": 71,
                "restarts_2h": 3,
                "memory_trend": "+2.1%/10min",
            },
            "worker-service": {
                "cpu_percent": 28,
                "memory_percent": 65,
                "restarts_2h": 2,
                "memory_trend": "+1.8%/10min",
            },
            "notification-service": {
                "cpu_percent": 19,
                "memory_percent": 78,
                "restarts_2h": 4,
                "memory_trend": "+2.4%/10min",
            },
            "auth-service": {
                "cpu_percent": 22,
                "memory_percent": 71,
                "restarts_2h": 0,
                "memory_trend": "+1.5%/10min",
            },
            "payment-service": {
                "cpu_percent": 31,
                "memory_percent": 42,
                "restarts_2h": 0,
                "memory_trend": "+0.1%/10min",
            },
        },
        "investigation_data": {
            "api-server": [
                "2024-01-15 13:55:00 DEBUG logging-middleware: Buffer pool size: 245MB (expected: <10MB)",
                "2024-01-15 13:55:01 DEBUG logging-middleware: Log entries buffered but flush() never called",
                "2024-01-15 13:55:02 WARN  api-server: Heap dump — logging-middleware holds 89% of live objects",
            ],
            "worker-service": [
                "2024-01-15 14:26:00 DEBUG logging-middleware: Buffer pool size: 198MB (expected: <10MB)",
                "2024-01-15 14:26:01 WARN  worker-service: Heap dominated by com.logging.BufferPool objects",
            ],
            "notification-service": [
                "2024-01-15 14:58:00 DEBUG logging-middleware: Buffer pool size: 267MB (expected: <10MB)",
                "2024-01-15 12:00:10 INFO  deployment: logging-middleware v2.1.0 introduced async buffer (never released)",
            ],
            "logging-middleware": [
                "2024-01-15 12:00:00 INFO  logging-middleware v2.1.0: async buffer enabled",
                "2024-01-15 12:00:00 WARN  logging-middleware v2.1.0: flush interval set to 0 (disabled)",
                "2024-01-15 12:00:00 INFO  changelog: v2.1.0 added async buffering, regression: flush not wired up",
            ],
        },
    },
}


class SRETriageEnv:
    def __init__(self, task_name: str = "cpu_spike"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASKS.keys())}")
        self.task_name = task_name
        self.task = TASKS[task_name]
        self.step_count = 0
        self.max_steps = 10
        self.done = False
        self.best_score = 0.01
        self.last_reward: Optional[SREReward] = None
        self.reward_history: List[float] = []
        self.investigated: List[str] = []

    def reset(self) -> SREObservation:
        self.step_count = 0
        self.done = False
        self.best_score = 0.01
        self.last_reward = None
        self.reward_history = []
        self.investigated = []
        return self._build_observation()

    def step(self, action: SREAction) -> Tuple[SREObservation, SREReward, bool, Dict]:
        self.step_count += 1

        if action.action_type == "investigate":
            reward = self._handle_investigate(action)
        elif action.action_type == "diagnose":
            reward = self._grade_diagnosis(action)
        elif action.action_type == "resolve":
            reward = self._grade_resolution(action)
        elif action.action_type == "done":
            reward = SREReward(
                score=self.best_score,
                feedback="Episode ended by agent.",
            )
            self.done = True
        else:
            reward = SREReward(
                score=0.01,
                feedback=f"Unknown action_type '{action.action_type}'. Use: investigate, diagnose, resolve, done.",
            )

        if reward.score > self.best_score:
            self.best_score = reward.score
        self.last_reward = reward
        self.reward_history.append(round(reward.score, 4))

        if self.step_count >= self.max_steps:
            self.done = True

        obs = self._build_observation()
        return obs, reward, self.done, {"task": self.task_name, "step": self.step_count}

    def state(self) -> Dict:
        return {
            "task": self.task_name,
            "step": self.step_count,
            "done": self.done,
            "best_score": self.best_score,
            "last_reward": self.last_reward.score if self.last_reward else 0.01,
            "investigated_services": self.investigated,
        }

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_observation(self, extra_info: Optional[Dict] = None) -> SREObservation:
        return SREObservation(
            alerts=self.task["initial_alerts"],
            logs=self.task["initial_logs"],
            metrics=self.task["initial_metrics"],
            task_description=self.task["description"],
            step=self.step_count,
            additional_info=extra_info,
        )

    def _handle_investigate(self, action: SREAction) -> SREReward:
        service = (action.service or "").strip().lower()
        inv_data = self.task["investigation_data"]

        # Normalise service name for matching
        matched_key = None
        for key in inv_data:
            if service in key.lower() or key.lower() in service:
                matched_key = key
                break

        if not matched_key:
            available = list(inv_data.keys())
            return SREReward(
                score=max(0.01, self.best_score),
                feedback=f"No investigation data for '{service}'. Available services: {available}",
            )

        if matched_key in self.investigated:
            return SREReward(
                score=max(0.01, self.best_score),
                feedback=f"Already investigated {matched_key}. Try diagnosing with what you know.",
            )

        self.investigated.append(matched_key)
        logs = inv_data[matched_key]
        # Inject extra logs into observation by updating initial_logs temporarily
        self.task["initial_logs"] = self.task["initial_logs"] + logs

        return SREReward(
            score=min(0.15, max(0.01, self.best_score)),
            feedback=f"Retrieved {len(logs)} additional log lines for {matched_key}.",
        )

    def _grade_diagnosis(self, action: SREAction) -> SREReward:
        task = self.task
        score = 0.01

        # Severity (0.30 weight)
        severity_correct = False
        if action.severity and action.severity.lower() == task["severity_answer"]:
            severity_correct = True
            score += 0.30

        # Root cause keyword match (up to 0.45)
        rc_score = 0.0
        if action.root_cause:
            text = action.root_cause.lower()
            hits = sum(1 for kw in task["root_cause_keywords"] if kw.lower() in text)
            if hits > 0:
                rc_score = min(0.45, 0.12 * hits)
                score += rc_score

        # Affected services listed (0.05 bonus)
        if action.affected_services and len(action.affected_services) > 0:
            score += 0.05

        score = min(0.75, score)  # diagnose caps at 0.75; resolve needed for full
        score = max(0.02, score)

        feedback_parts = []
        if severity_correct:
            feedback_parts.append(f"Severity '{action.severity}' is correct (+0.30).")
        else:
            feedback_parts.append(
                f"Severity '{action.severity}' is wrong (expected '{task['severity_answer']}')."
            )
        if rc_score > 0:
            feedback_parts.append(f"Root cause has relevant keywords (+{rc_score:.2f}).")
        else:
            feedback_parts.append("Root cause missing key terms. Investigate more services.")

        return SREReward(
            score=round(score, 4),
            severity_correct=severity_correct,
            root_cause_score=round(rc_score, 4),
            feedback=" ".join(feedback_parts),
        )

    def _grade_resolution(self, action: SREAction) -> SREReward:
        task = self.task

        # Re-run diagnosis component if provided
        diag_score = 0.01
        severity_correct = False
        rc_score = 0.0

        if action.severity and action.severity.lower() == task["severity_answer"]:
            severity_correct = True
            diag_score += 0.30

        if action.root_cause:
            text = action.root_cause.lower()
            hits = sum(1 for kw in task["root_cause_keywords"] if kw.lower() in text)
            if hits > 0:
                rc_score = min(0.40, 0.12 * hits)
                diag_score += rc_score

        # Action keyword match (up to 0.25)
        action_score = 0.0
        if action.recommended_action:
            text = action.recommended_action.lower()
            hits = sum(1 for kw in task["action_keywords"] if kw.lower() in text)
            if hits > 0:
                action_score = min(0.25, 0.09 * hits)

        total = diag_score + action_score
        total = min(0.99, max(0.02, total))

        feedback_parts = []
        if severity_correct:
            feedback_parts.append(f"Severity correct.")
        if rc_score > 0:
            feedback_parts.append(f"Root cause score: {rc_score:.2f}.")
        if action_score > 0:
            feedback_parts.append(f"Recommended action score: {action_score:.2f}.")
        else:
            feedback_parts.append("Recommended action missing key terms.")

        return SREReward(
            score=round(total, 4),
            severity_correct=severity_correct,
            root_cause_score=round(rc_score, 4),
            action_score=round(action_score, 4),
            feedback=" ".join(feedback_parts),
        )
