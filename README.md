---
title: SRE Incident Triage
emoji: 🔥
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - sre
  - reinforcement-learning
  - incident-response
  - devops
  - agent
---

<div align="center">

# 🚨 SRE Incident Triage

### *Can your AI survive an on-call shift?*

**An OpenEnv reinforcement learning environment where agents triage real production incidents —
reading noisy alerts, tracing cascading failures, and recommending the right fix under pressure.**

---

![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Tasks](https://img.shields.io/badge/Tasks-3%20(Easy%20→%20Hard)-orange?style=for-the-badge)
![Scores](https://img.shields.io/badge/Baseline-0.66%20–%200.96-blue?style=for-the-badge)

</div>

---

## 🧠 What Is This?

Every company has an on-call rotation. At 3am, an engineer gets paged — alerts firing, logs streaming, services cascading. They have minutes to find the root cause and act.

**This environment turns that into an RL benchmark.**

An AI agent receives the same raw signals a human SRE would see: production alerts, service logs, and real-time metrics. It must investigate, diagnose, and resolve — just like a real engineer would.

> Unlike toy environments, this one models a task where **reasoning quality directly determines the score**. LLMs that have absorbed engineering knowledge genuinely outperform those that haven't.

---

## 🎯 Three Incident Scenarios

```
EASY ──────────────────────────────────────────── HARD
  │                    │                            │
  ▼                    ▼                            ▼
cpu_spike       cascading_failure            memory_leak
```

| 🏷️ Task | 🎚️ Difficulty | 💥 Incident | 🔍 What Agent Must Find |
|---------|-------------|------------|------------------------|
| `cpu_spike` | 🟢 Easy | Traffic flood hitting api-server | No rate limiting, IP making 8K req/min |
| `cascading_failure` | 🟡 Medium | 4 services down in a cascade | payment-service v2.3.1 set DB pool to 500 (was 10) |
| `memory_leak` | 🔴 Hard | Multi-service OOM kills over 2h | Shared logging-middleware v2.1.0 never flushes its buffer |

---

## ⚡ Agent Action Space

The agent speaks **JSON** — clean, structured, LLM-native:

```json
// 🔍 Drill into a specific service
{"action_type": "investigate", "service": "payment-service"}

// 🩺 Submit your diagnosis
{
  "action_type": "diagnose",
  "severity": "critical",
  "root_cause": "DB connection pool exhausted after v2.3.1 deploy",
  "affected_services": ["payment-service", "checkout-service"]
}

// 🛠️ Recommend the fix (highest scoring action)
{
  "action_type": "resolve",
  "severity": "critical",
  "root_cause": "payment-service v2.3.1 set DB_POOL_SIZE=500, exceeding DB max_connections=100",
  "affected_services": ["payment-service", "checkout-service", "order-service"],
  "recommended_action": "rollback payment-service to v2.3.0 and restart connection pool"
}

// ✅ End episode
{"action_type": "done"}
```

---

## 👁️ Observation Space

Every step, the agent sees:

```json
{
  "alerts": [
    "CRITICAL: payment-service health check failing (0% healthy pods)",
    "HIGH: checkout-service error rate at 78%"
  ],
  "logs": [
    "16:44:30 INFO  deployment: payment-service v2.3.1 rolled out",
    "16:44:31 ERROR payment-service: Failed to acquire DB connection (pool exhausted)",
    "16:45:02 ERROR checkout-service: payment-service call timed out after 3000ms"
  ],
  "metrics": {
    "payment-service": {"cpu_percent": 12, "error_rate": 100, "healthy_pods": 0},
    "payment-db":      {"cpu_percent": 98, "active_connections": 500, "max_connections": 100}
  },
  "step": 2,
  "additional_info": null  // populated after investigate actions
}
```

---

## 📊 Reward Function

Scores are **partial** — every step provides signal, not just the final answer:

| Component | Max Weight | How It's Earned |
|-----------|-----------|-----------------|
| 🎚️ Severity classification | **+0.30** | Exact match: `critical / high / medium / low` |
| 🔎 Root cause accuracy | **+0.45** | Keyword matching — partial credit per relevant term |
| 🗂️ Affected services | **+0.05** | Any services identified |
| 🛠️ Recommended action | **+0.25** | Keyword match on correct remediation |

> All scores clamped to **(0.01, 0.99)** — never binary, always informative.

**Strategy that works best:** `investigate → investigate → resolve` (2-step exploration before committing)

---

## 📈 Baseline Results

Tested with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router:

```
cpu_spike          ████████████████████░  0.92  ✅ Success
cascading_failure  ████████████████████░  0.96  ✅ Success
memory_leak        █████████████░░░░░░░░  0.66  ✅ Success
```

The hard task (`memory_leak`) genuinely challenges frontier models — the shared component spans 4 services with no obvious single alert.

---

## 🚀 Quick Start

```bash
# Clone and build
git clone https://github.com/samnitmehandiratta/my-openenv
cd my-openenv
docker build -t sre-triage .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=hf_... \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  sre-triage
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reset` | Start new episode — body: `{"task": "cpu_spike"}` |
| `POST` | `/step` | Take action — body: `{"action": {...}}` |
| `GET` | `/state` | Current episode state |

## 🧪 Run Inference Locally

```bash
HF_TOKEN=hf_... \
API_BASE_URL=https://router.huggingface.co/v1 \
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
python inference.py
```

---

<div align="center">

Built for the **Scaler × HuggingFace OpenEnv Hackathon 2025**

*Because the best RL environments are the ones where intelligence actually matters.*

</div>
