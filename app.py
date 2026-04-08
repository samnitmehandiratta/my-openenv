#!/usr/bin/env python3
"""
Unified server for Physics Surrogate Environment
Runs FastAPI for API endpoints and mounts Gradio UI
"""
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn

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


# Mount Gradio UI
try:
    import gradio as gr
    import numpy as np
    import json
    from gradio.routes import mount_gradio_app

    gradio_env = None
    gradio_current_task = "fluid_short_prediction"

    def reset_env(task_name):
        global gradio_env, gradio_current_task
        gradio_current_task = task_name
        gradio_env = PhysicsSurrogateEnv(task_name=task_name)
        obs = gradio_env.reset()
        field = np.array(obs.initial_state)
        return (
            field.tolist(),
            obs.task_description,
            json.dumps(obs.metadata.model_dump()),
            "\n".join(obs.hints),
        )

    def take_action(num_steps=1, done=False):
        global gradio_env
        if gradio_env is None:
            reset_env(gradio_current_task)

        action = PhysicsAction(predicted_field=None, num_steps=num_steps, done=done)
        obs, reward, done, info = gradio_env.step(action)
        field = np.array(obs.initial_state)

        return (
            field.tolist(),
            f"Score: {reward.score:.4f}\nMSE: {reward.mse:.6f}\nCorrelation: {reward.correlation:.4f}",
            f"Step {gradio_env.current_step}/{gradio_env.max_steps}",
            "Done!" if done else f"Episode in progress",
        )

    with gr.Blocks(title="Physics Surrogate Environment") as gradio_app:
        gr.Markdown("# Physics Surrogate Prediction Environment")
        gr.Markdown("Predict future states of physical systems using The Well dataset")

        with gr.Row():
            with gr.Column():
                task_dropdown = gr.Dropdown(
                    choices=[
                        "fluid_short_prediction",
                        "fluid_medium_prediction",
                        "turbulence_prediction",
                    ],
                    value="fluid_short_prediction",
                    label="Select Task",
                )
                reset_btn = gr.Button("Reset Environment")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Current State")
                state_out = gr.JSON(label="State Field (first 10x10)")
            with gr.Column():
                gr.Markdown("### Task Info")
                task_out = gr.Textbox(label="Task Description")
                meta_out = gr.JSON(label="Metadata")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Hints")
                hints_out = gr.Textbox(lines=6, label="Hints")

        with gr.Row():
            num_steps_in = gr.Slider(
                minimum=1, maximum=50, value=1, step=1, label="Prediction Steps"
            )
            done_check = gr.Checkbox(label="Done", value=False)
            action_btn = gr.Button("Take Action")

        with gr.Row():
            score_out = gr.Textbox(label="Score/Metrics")
            step_out = gr.Textbox(label="Step Status")
            status_out = gr.Textbox(label="Episode Status")

        task_dropdown.change(
            reset_env, task_dropdown, [state_out, task_out, meta_out, hints_out]
        )
        reset_btn.click(
            reset_env, task_dropdown, [state_out, task_out, meta_out, hints_out]
        )
        action_btn.click(
            take_action,
            [num_steps_in, done_check],
            [state_out, score_out, step_out, status_out],
        )

        gr.Markdown("### Instructions")
        gr.Markdown("""
        1. Select a task from the dropdown
        2. Click Reset Environment to start
        3. Use Take Action to run predictions
        4. Check Done when satisfied with predictions
        5. Score is based on prediction accuracy vs ground truth
        """)

    app = mount_gradio_app(app, gradio_app, path="/gradio")
    print("Gradio UI mounted at /gradio")
    
except ImportError as e:
    print(f"Gradio not available: {e}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)


def main():
    """Entry point for server script"""
    uvicorn.run(app, host="0.0.0.0", port=7860)
