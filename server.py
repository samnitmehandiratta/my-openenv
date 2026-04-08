#!/usr/bin/env python3
"""
SRE Incident Triage Environment - OpenEnv Server
FastAPI server exposing reset / step / state endpoints
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn

from env import SRETriageEnv
from models import SREAction

app = FastAPI(title="SRE Incident Triage - OpenEnv")

env: Optional[SRETriageEnv] = None
current_task_name = "cpu_spike"


class ResetInput(BaseModel):
    task: Optional[str] = None


class StepInput(BaseModel):
    action: Dict[str, Any]


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "SRE Incident Triage Environment",
        "tasks": ["cpu_spike", "cascading_failure", "memory_leak"],
    }


@app.post("/reset")
async def reset(input_data: ResetInput = ResetInput(task=None)):
    global env, current_task_name
    task = input_data.task if input_data.task else current_task_name
    current_task_name = task
    try:
        env = SRETriageEnv(task_name=task)
        obs = env.reset()
        return {
            "observation": obs.model_dump(),
            "reward": None,
            "done": False,
            "info": {"task": task},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(input_data: StepInput):
    global env
    if env is None:
        env = SRETriageEnv(task_name=current_task_name)
        env.reset()
    try:
        action = SREAction(**input_data.action)
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/state")
async def state():
    global env
    if env is None:
        env = SRETriageEnv(task_name=current_task_name)
        env.reset()
    return env.state()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
