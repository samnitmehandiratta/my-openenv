import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import numpy as np

from env import PhysicsSurrogateEnv
from models import PhysicsAction

app = FastAPI(title="Physics Surrogate OpenEnv")

env: Optional[PhysicsSurrogateEnv] = None
current_task_name = "fluid_short_prediction"


class ResetInput(BaseModel):
    task: Optional[str] = None


class StepInput(BaseModel):
    action: Optional[Dict[str, Any]] = None


@app.get("/")
async def root():
    return {"status": "ok", "message": "Physics Surrogate Environment - OpenEnv"}


@app.get("/gradio")
async def gradio_redirect():
    from fastapi.responses import RedirectResponse

    return RedirectResponse(url="/")


@app.post("/reset")
async def reset(input_data: ResetInput = ResetInput(task=None)):
    global env, current_task_name
    task = input_data.task if input_data.task else current_task_name
    current_task_name = task

    try:
        env = PhysicsSurrogateEnv(task_name=task)
        obs = env.reset()

        return {
            "observation": {
                "initial_state": obs.initial_state,
                "metadata": obs.metadata.model_dump(),
                "task_description": obs.task_description,
                "hints": obs.hints,
            },
            "reward": None,
            "done": False,
            "info": {"task": task},
        }
    except Exception as e:
        print(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(input_data: StepInput):
    global env
    if env is None:
        env = PhysicsSurrogateEnv(task_name=current_task_name)
        env.reset()

    try:
        action_dict = input_data.action or {}
        action = PhysicsAction(
            predicted_field=action_dict.get("predicted_field"),
            num_steps=action_dict.get("num_steps", 1),
            done=action_dict.get("done", False),
        )

        obs, reward, done, info = env.step(action)

        return {
            "observation": {
                "initial_state": obs.initial_state,
                "metadata": obs.metadata.model_dump(),
                "task_description": obs.task_description,
                "hints": obs.hints,
            },
            "reward": reward.score if reward else None,
            "done": done,
            "info": info,
        }
    except Exception as e:
        print(f"Step error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
async def state():
    global env
    if env is None:
        return {"error": "Environment not initialized"}
    return env.state()


# For local testing
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
