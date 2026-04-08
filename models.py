from pydantic import BaseModel, Field
from typing import List, Optional, Any, Union
import numpy as np


class PhysicsMetadata(BaseModel):
    dataset: str
    resolution: List[int]
    field_type: str
    time_step: int


class PhysicsObservation(BaseModel):
    initial_state: Union[List[List[float]], List[float]]  # 2D grid or flattened
    metadata: PhysicsMetadata
    task_description: str
    hints: List[str] = []


class PhysicsAction(BaseModel):
    predicted_field: Optional[Union[List[List[float]], List[float]]] = None
    num_steps: int = 1
    done: bool = False


class PhysicsReward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    mse: float
    correlation: float
    error: Optional[str] = None
