from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SREObservation(BaseModel):
    alerts: List[str]
    logs: List[str]
    metrics: Dict[str, Any]
    task_description: str
    step: int
    additional_info: Optional[Dict[str, Any]] = None


class SREAction(BaseModel):
    action_type: str  # "investigate", "diagnose", "resolve", "done"
    service: Optional[str] = None
    severity: Optional[str] = None  # "critical", "high", "medium", "low"
    root_cause: Optional[str] = None
    affected_services: Optional[List[str]] = None
    recommended_action: Optional[str] = None


class SREReward(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    severity_correct: bool = False
    root_cause_score: float = 0.0
    action_score: float = 0.0
    feedback: str = ""
