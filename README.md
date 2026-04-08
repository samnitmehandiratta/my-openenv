---
title: Physics Surrogate Environment
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
---

# Physics Surrogate Environment

An OpenEnv environment for physics surrogate modeling using **real data from The Well** (NeurIPS 2024). Agents learn to predict future states of physical systems from initial conditions.

## Overview

This environment uses **actual physics simulation data** from The Well dataset - a 15TB collection of machine learning datasets for spatiotemporal physical systems. The environment loads real HDF5 files containing:
- **Turbulent Radiative Layer** simulations (128x384 resolution)
- **Density** and **Pressure** fields
- 101 timesteps per trajectory

## Real Data Source

- **Dataset**: [turbulent_radiative_layer_2D](https://github.com/Polymathicai/the_well/tree/main/the_well/datasets/turbulent_radiative_layer_2D)
- **Source**: Polymathic AI, The Well (NeurIPS 2024)
- **Resolution**: 128 × 384 pixels
- **Timesteps**: 101 per trajectory
- **Physics**: Radiative layer simulations with varying cooling parameters

## Tasks

| Task | Difficulty | Prediction Horizon | Field |
|------|------------|-------------------|-------|
| fluid_short_prediction | Easy | 5 timesteps | density |
| fluid_medium_prediction | Medium | 20 timesteps | pressure |
| turbulence_prediction | Hard | 50 timesteps | density |

## Action Space

Agents provide predictions for future states:

- `predicted_field`: The predicted 2D scalar field (128x384)
- `num_steps`: Number of timesteps to predict
- `done`: Whether to finish the episode

## Observation Space

- `initial_state`: Current 2D field (normalized 0-1)
- `metadata`: Dataset name, resolution, field type, time step
- `task_description`: Description of the prediction task
- `hints`: Guidance for the agent

## Reward Function

Score (0.0-1.0) based on:
- MSE between prediction and ground truth (real physics data)
- Pearson correlation coefficient
- Partial credit for reasonable predictions

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- pydantic, openai, numpy, h5py
- the_well package for data handling

## Data Setup

The environment expects the data file at:
```
./the_well_data/data/test/turbulent_radiative_layer_tcool_0.03.hdf5
```

Download from HuggingFace:
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='polymathic-ai/turbulent_radiative_layer_2D',
    filename='data/test/turbulent_radiative_layer_tcool_0.03.hdf5',
    repo_type='dataset',
    local_dir='./the_well_data'
)
```

## Running Locally

```python
from env import PhysicsSurrogateEnv
from models import PhysicsAction
import numpy as np

env = PhysicsSurrogateEnv(task_name="fluid_short_prediction")
obs = env.reset()

# Make prediction
pred = np.random.rand(128, 384).tolist()
action = PhysicsAction(predicted_field=pred, num_steps=1)
obs, reward, done, info = env.step(action)

print(f"Reward: {reward.score}")
env.close()
```

## Running Inference

```bash
# Set API key
export OPENROUTER_API_KEY=your_key

# Run inference
python inference.py
```

Environment variables:
- `OPENROUTER_API_KEY` - API key for LLM
- `API_BASE_URL` - API endpoint (default: https://openrouter.ai/v1)
- `MODEL_NAME` - Model to use (default: google/gemma-2-2b-it:free)

## Baseline Scores

| Task | Score |
|------|-------|
| fluid_short_prediction | 0.8-1.0 |
| fluid_medium_prediction | 0.6-1.0 |
| turbulence_prediction | 0.8-1.0 |

## Docker

```bash
docker build -t physics-surrogate .
docker run -p 7860:7860 physics-surrogate
```

## API Endpoints

When running as a server:
- `GET /` - Health check
- `POST /reset` - Reset environment
- `POST /step` - Take an action
- `GET /state` - Get current state

## References

- [The Well Dataset](https://github.com/polymathicai/the_well) - Polymathic AI
- [Paper](https://arxiv.org/abs/2412.00568) - NeurIPS 2024
- [HuggingFace Dataset](https://huggingface.co/datasets/polymathic-ai/turbulent_radiative_layer_2D)