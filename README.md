---
title: SRE Incident Triage
emoji: 🚨
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
tags:
  - openenv
---

# SRE Incident Triage Environment

An OpenEnv environment where an AI agent acts as an on-call SRE engineer triaging real production incidents. Given alerts, logs, and metrics, the agent must investigate services, identify the root cause, and recommend the correct remediation.

## Overview

On-call SRE triage is one of the most cognitively demanding real-world tasks — reading noisy signals, finding causal chains across services, and acting under time pressure. This environment models three realistic incident archetypes with increasing complexity.

## Tasks

| Task | Difficulty | Incident Type |
|------|-----------|---------------|
| `cpu_spike` | Easy | Traffic spike overwhelming api-server, no rate limiting |
| `cascading_failure` | Medium | Bad deployment config causing DB exhaustion → cascade |
| `memory_leak` | Hard | Shared logging middleware leak causing multi-service OOM |

## Action Space

Actions are JSON objects with `action_type`:

```json
{"action_type": "investigate", "service": "api-server"}

{"action_type": "diagnose", "severity": "high", "root_cause": "...", "affected_services": ["api-server"]}

{"action_type": "resolve", "severity": "critical", "root_cause": "...", "affected_services": ["payment-service"], "recommended_action": "rollback payment-service to v2.3.0"}

{"action_type": "done"}
```

## Observation Space

```json
{
  "alerts": ["HIGH CPU ALERT: api-server at 95%", "..."],
  "logs": ["2024-01-15 14:23:02 WARN api-server: queue depth 847", "..."],
  "metrics": {"api-server": {"cpu_percent": 95, "error_rate": 2.3}, "..."},
  "task_description": "...",
  "step": 1,
  "additional_info": null
}
```

## Reward Function

| Component | Weight | Signal |
|-----------|--------|--------|
| Severity classification | 0.30 | Correct critical/high/medium/low |
| Root cause keywords | up to 0.45 | Partial credit per relevant keyword |
| Affected services listed | 0.05 | Any services identified |
| Recommended action | up to 0.25 | Keywords matching correct fix |

Scores are always in `(0.01, 0.99)`. Investigate actions return small rewards (0.05-0.15) to encourage exploration without over-rewarding.

## Setup

```bash
docker build -t sre-triage .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  sre-triage
```

## API Endpoints

- `POST /reset` — Start new episode (optionally pass `{"task": "cpu_spike"}`)
- `POST /step` — Take an action (`{"action": {"action_type": "investigate", "service": "api-server"}}`)
- `GET /state` — Current environment state

## Baseline Scores

| Task | Score | Notes |
|------|-------|-------|
| cpu_spike | ~0.70 | Easy — clear traffic + rate limit signals |
| cascading_failure | ~0.55 | Medium — requires tracing deployment diff |
| memory_leak | ~0.40 | Hard — shared component across services |

## Running Inference

```bash
HF_TOKEN=hf_... python inference.py
```
