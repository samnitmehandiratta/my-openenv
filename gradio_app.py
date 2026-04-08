import gradio as gr
import numpy as np
import json

from env import PhysicsSurrogateEnv
from models import PhysicsAction

env = None
current_task = "fluid_short_prediction"


def reset_env(task_name):
    global env, current_task
    current_task = task_name
    env = PhysicsSurrogateEnv(task_name=task_name)
    obs = env.reset()

    # Convert to 2D array for display
    field = np.array(obs.initial_state)

    return (
        field.tolist(),
        obs.task_description,
        json.dumps(obs.metadata.model_dump()),
        "\n".join(obs.hints),
    )


def take_action(num_steps=1, done=False):
    global env
    if env is None:
        reset_env(current_task)

    action = PhysicsAction(predicted_field=None, num_steps=num_steps, done=done)

    obs, reward, done, info = env.step(action)
    field = np.array(obs.initial_state)

    return (
        field.tolist(),
        f"Score: {reward.score:.4f}\nMSE: {reward.mse:.6f}\nCorrelation: {reward.correlation:.4f}",
        f"Step {env.current_step}/{env.max_steps}",
        "Done!" if done else f"Episode in progress",
    )


with gr.Blocks(title="Physics Surrogate Environment") as demo:
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

demo.launch(server_port=7860)
