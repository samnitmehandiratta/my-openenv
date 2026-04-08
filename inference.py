"""
SRE Incident Triage - Baseline Inference Script
================================================
Mandatory env vars:
  API_BASE_URL  - LLM endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME    - Model identifier
  HF_TOKEN      - API key
"""
import json
import os
import re

from openai import OpenAI

from env import SRETriageEnv
from models import SREAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "sre_incident_triage"
MAX_STEPS = 8
TEMPERATURE = 0.3

TASKS = ["cpu_spike", "cascading_failure", "memory_leak"]

SYSTEM_PROMPT = """You are an expert SRE (Site Reliability Engineer) triaging a production incident.

You will receive alerts, logs, and metrics. Your job is to investigate, diagnose the root cause, and recommend a fix.

Available actions (respond with ONLY a valid JSON object):

1. Investigate a service for more logs:
   {"action_type": "investigate", "service": "<service_name>"}

2. Submit your diagnosis:
   {"action_type": "diagnose", "severity": "critical|high|medium|low", "root_cause": "<description>", "affected_services": ["<svc1>", ...]}

3. Submit diagnosis + recommended fix (best for high score):
   {"action_type": "resolve", "severity": "critical|high|medium|low", "root_cause": "<description>", "affected_services": ["<svc1>", ...], "recommended_action": "<specific remediation step>"}

4. Finish:
   {"action_type": "done"}

Strategy: investigate 1-2 suspicious services first, then resolve with a specific recommended_action.
Respond with ONLY a JSON object - no explanation, no markdown."""


def build_user_prompt(obs) -> str:
    metrics_str = json.dumps(obs.metrics, indent=2)
    alerts_str = "\n".join(f"  - {a}" for a in obs.alerts)
    logs_str = "\n".join(f"  {l}" for l in obs.logs[-20:])
    extra = ""
    if obs.additional_info:
        extra = f"\nADDITIONAL INFO:\n{json.dumps(obs.additional_info, indent=2)}"

    return f"""TASK: {obs.task_description}
STEP: {obs.step}

ALERTS:
{alerts_str}

RECENT LOGS:
{logs_str}

METRICS:
{metrics_str}{extra}

What action do you take? Respond with ONLY a JSON object."""


def parse_action(text: str) -> SREAction:
    text = re.sub(r"```(?:json)?", "", text).strip().strip("`").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return SREAction(**data)
        except Exception:
            pass
    return SREAction(action_type="done")


def run_task(client: OpenAI, task_name: str) -> dict:
    env = SRETriageEnv(task_name=task_name)
    obs = env.reset()
    rewards = []

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    for step_num in range(1, MAX_STEPS + 1):
        user_prompt = build_user_prompt(obs)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=300,
            )
            raw = completion.choices[0].message.content or ""
        except Exception as exc:
            raw = '{"action_type": "done"}'
            print(f"[DEBUG] LLM error: {exc}", flush=True)

        action = parse_action(raw)
        obs, reward, done, info = env.step(action)
        rewards.append(round(reward.score, 2))

        error_str = reward.feedback if reward.feedback else "null"
        print(
            f"[STEP] step={step_num} action={action.action_type} reward={reward.score:.2f} "
            f"done={str(done).lower()} error={error_str}",
            flush=True,
        )

        if done:
            break

    env.close()

    final_score = max(rewards) if rewards else 0.01
    success = final_score >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={len(rewards)} "
        f"score={final_score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return {"success": success, "steps": len(rewards), "score": final_score, "rewards": rewards}


def main():
    if not API_KEY:
        print("[END] success=false steps=0 score=0.01 rewards=0.01", flush=True)
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in TASKS:
        run_task(client, task)


if __name__ == "__main__":
    main()
