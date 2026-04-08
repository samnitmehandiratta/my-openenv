import asyncio
import os
import json
import re
from typing import List
from dotenv import load_dotenv

import numpy as np

from openai import OpenAI

from env import PhysicsSurrogateEnv
from models import PhysicsAction

load_dotenv()

API_KEY = (
    os.getenv("HF_TOKEN") or os.getenv("OPENROUTER_API_KEY") or os.getenv("API_KEY")
)
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-2-2b-it:free")
BENCHMARK = "physics_surrogate"
MAX_STEPS = 8
TEMPERATURE = 0.7

TASKS = ["fluid_short_prediction", "fluid_medium_prediction", "turbulence_prediction"]

SYSTEM_PROMPT = """You are a physics surrogate model. Your task is to predict future states of a physical system given an initial condition.

You are given a 2D field (128x384 grid) representing density/pressure values from a physics simulation.
Values are normalized between 0 and 1.

You need to predict what the field will look like at the next timestep. Respond with JSON:
{
    "reasoning": "Brief explanation of what you predict",
    "direction": "left|right|up|down|diagonal|stationary",
    "intensity": float between 0 and 1 (how much change),
    "done": false
}

Example: {"reasoning": "The fluid appears to flow rightward", "direction": "right", "intensity": 0.3, "done": false}

Respond ONLY with valid JSON."""


def parse_llm_response(response: str) -> dict:
    """Parse LLM response into prediction parameters"""
    try:
        # Extract JSON from response
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return {
                "direction": data.get("direction", "stationary"),
                "intensity": data.get("intensity", 0.1),
                "done": data.get("done", False),
                "reasoning": data.get("reasoning", ""),
            }
    except:
        pass

    return {"direction": "stationary", "intensity": 0.1, "done": False, "reasoning": ""}


def make_prediction(
    initial_state: np.ndarray, direction: str, intensity: float, step: int
) -> np.ndarray:
    """Generate prediction based on LLM output"""
    h, w = initial_state.shape
    pred = initial_state.copy()

    # Simple shift based on direction
    shift_x, shift_y = 0, 0

    if direction == "right":
        shift_x = int(5 * intensity)
    elif direction == "left":
        shift_x = -int(5 * intensity)
    elif direction == "up":
        shift_y = -int(5 * intensity)
    elif direction == "down":
        shift_y = int(5 * intensity)
    elif direction == "diagonal":
        shift_x = int(3 * intensity)
        shift_y = int(3 * intensity)

    # Apply shift using numpy roll
    if shift_x != 0:
        pred = np.roll(pred, shift_x, axis=1)
    if shift_y != 0:
        pred = np.roll(pred, shift_y, axis=0)

    # Add some temporal evolution
    time_factor = step * 0.05 * intensity
    x = np.linspace(0, 2 * np.pi, w)
    y = np.linspace(0, 2 * np.pi, h)
    X, Y = np.meshgrid(x, y)

    wave = np.sin(X + time_factor) * np.cos(Y + time_factor * 0.5)
    wave = (wave - wave.min()) / (wave.max() - wave.min() + 1e-8)

    # Blend shifted with wave
    blend = intensity * 0.3
    pred = pred * (1 - blend) + wave * blend

    return pred


def run_task(client: OpenAI, task_name: str) -> dict:
    env = PhysicsSurrogateEnv(task_name=task_name)
    obs = env.reset()
    rewards = []

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    for step in range(1, MAX_STEPS + 1):
        initial_state = np.array(obs.initial_state)

        # Get summary of state for LLM
        state_summary = f"Field shape: {initial_state.shape}, range: [{initial_state.min():.3f}, {initial_state.max():.3f}], mean: {initial_state.mean():.3f}"

        try:
            # Call LLM
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Current timestep: {step}/{MAX_STEPS}. {state_summary}. What do you predict for the next frame?",
                    },
                ],
                temperature=TEMPERATURE,
                max_tokens=200,
            )

            llm_response = response.choices[0].message.content
            parsed = parse_llm_response(llm_response)

            action_str = f"{parsed['direction']}:{parsed['intensity']:.2f}"

            # Generate prediction based on LLM response
            pred = make_prediction(
                initial_state, parsed["direction"], parsed["intensity"], step
            )

            if parsed["done"]:
                action = PhysicsAction(
                    predicted_field=pred.tolist(), num_steps=1, done=True
                )
            else:
                action = PhysicsAction(
                    predicted_field=pred.tolist(), num_steps=1, done=False
                )

        except Exception as e:
            print(f"LLM call failed: {e}, using fallback")
            pred = initial_state.copy()
            action = PhysicsAction(
                predicted_field=pred.tolist(), num_steps=1, done=(step >= MAX_STEPS)
            )
            action_str = "fallback"

        obs, reward, done, info = env.step(action)
        rewards.append(round(reward.score, 2))

        error_str = "null"
        print(
            f"[STEP] step={step} action={action_str} reward={reward.score:.2f} done={str(done).lower()} error={error_str}",
            flush=True,
        )

        if done:
            break

    env.close()

    final_score = max(rewards) if rewards else 0.0
    success = final_score >= 0.5
    rewards_str = ",".join(str(r) for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={len(rewards)} score={final_score:.2f} rewards={rewards_str}",
        flush=True,
    )

    return {
        "success": success,
        "steps": len(rewards),
        "score": final_score,
        "rewards": rewards,
    }


async def main():
    if not API_KEY:
        print(
            "[END] success=false steps=0 score=0.00 rewards=0.00 error=HF_TOKEN not set",
            flush=True,
        )
        return

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    for task in TASKS:
        result = run_task(client, task)
        results.append(result)
        await asyncio.sleep(1)

    print(f"\n=== BASELINE RESULTS ===", flush=True)
    for i, task in enumerate(TASKS):
        r = results[i]
        print(f"{task}: score={r['score']:.2f}, success={r['success']}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
