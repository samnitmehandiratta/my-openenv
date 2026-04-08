import os
import h5py
import numpy as np
from typing import Optional, Dict, Any, List
from models import PhysicsObservation, PhysicsAction, PhysicsReward, PhysicsMetadata


DATA_FILES = {
    "fluid_short_prediction": {
        "name": "Short-term Fluid Prediction",
        "difficulty": "easy",
        "description": "Predict 5 timesteps ahead for turbulent radiative layer simulation",
        "file_path": os.getenv(
            "DATA_FILE_PATH",
            "./the_well_data/data/test/turbulent_radiative_layer_tcool_0.03.hdf5",
        ),
        "field": "density",
        "resolution": [128, 384],
        "num_steps_predict": 5,
        "target_mse": 0.05,
        "hints": [
            "Turbulent radiative layers show complex convection patterns",
            "Initial state contains temperature/density gradients",
            "Prediction error accumulates over time",
        ],
    },
    "fluid_medium_prediction": {
        "name": "Medium-term Fluid Prediction",
        "difficulty": "medium",
        "description": "Predict 20 timesteps ahead for turbulent flow",
        "file_path": os.getenv(
            "DATA_FILE_PATH",
            "./the_well_data/data/test/turbulent_radiative_layer_tcool_0.03.hdf5",
        ),
        "field": "pressure",
        "resolution": [128, 384],
        "num_steps_predict": 20,
        "target_mse": 0.10,
        "hints": [
            "Pressure fields show wave propagation",
            "Longer prediction horizons require understanding of fluid dynamics",
            "Consider temporal coherence",
        ],
    },
    "turbulence_prediction": {
        "name": "Turbulence Long-range Prediction",
        "difficulty": "hard",
        "description": "Predict 50 timesteps ahead for turbulent flow with high accuracy",
        "file_path": os.getenv(
            "DATA_FILE_PATH",
            "./the_well_data/data/test/turbulent_radiative_layer_tcool_0.03.hdf5",
        ),
        "field": "density",
        "resolution": [128, 384],
        "num_steps_predict": 50,
        "target_mse": 0.20,
        "hints": [
            "Turbulent flows have chaotic behavior",
            "Long-term predictions diverge quickly",
            "Focus on maintaining physical constraints",
        ],
    },
}


def compute_metrics(
    prediction: np.ndarray, ground_truth: np.ndarray
) -> Dict[str, float]:
    mse = float(np.mean((prediction - ground_truth) ** 2))
    pred_flat = prediction.flatten()
    gt_flat = ground_truth.flatten()
    correlation = float(np.corrcoef(pred_flat, gt_flat)[0, 1])
    if np.isnan(correlation):
        correlation = 0.0
    return {"mse": mse, "correlation": correlation}


class PhysicsSurrogateEnv:
    def __init__(self, task_name: str = "fluid_short_prediction"):
        self.task_name = task_name
        self.task = DATA_FILES[task_name]
        self.field_name = self.task["field"]
        self.resolution = self.task["resolution"]
        self.num_steps_predict = self.task["num_steps_predict"]
        self.target_mse = self.task["target_mse"]

        self.current_step = 0
        self.max_steps = 8
        self.done = False

        self.initial_state: np.ndarray = None
        self.ground_truth_trajectory: List[np.ndarray] = None
        self.current_prediction: np.ndarray = None
        self.last_reward = None
        self.reward_history = []

        self._load_data()

    def _load_data(self):
        file_path = self.task["file_path"]

        if not os.path.exists(file_path):
            print(
                f"Data file not found at {file_path}, downloading from HuggingFace..."
            )
            try:
                from huggingface_hub import hf_hub_download

                file_path = hf_hub_download(
                    repo_id="polymathic-ai/turbulent_radiative_layer_2D",
                    filename="data/test/turbulent_radiative_layer_tcool_0.03.hdf5",
                    repo_type="dataset",
                    local_dir="./the_well_data",
                )
                print(f"Downloaded to: {file_path}")
            except Exception as e:
                print(f"Download failed: {e}")
                raise FileNotFoundError(f"Data file not found and could not download")

        with h5py.File(file_path, "r") as f:
            t0_fields = f["t0_fields"]
            field_data = t0_fields[self.field_name]

            max_timesteps = min(self.num_steps_predict + 10, field_data.shape[1])
            trajectory = []
            for t in range(max_timesteps):
                frame = field_data[0, t]
                frame = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
                trajectory.append(frame.astype(np.float32))

            self.ground_truth_trajectory = trajectory

    def reset(self) -> PhysicsObservation:
        self.current_step = 0
        self.done = False
        self.last_reward = None
        self.reward_history = []

        self.initial_state = self.ground_truth_trajectory[0].copy()

        obs = PhysicsObservation(
            initial_state=self.initial_state.tolist(),
            metadata=PhysicsMetadata(
                dataset="turbulent_radiative_layer_2D (The Well)",
                resolution=self.resolution,
                field_type=self.field_name,
                time_step=0,
            ),
            task_description=self.task["description"],
            hints=self.task["hints"],
        )
        return obs

    def step(
        self, action: PhysicsAction
    ) -> tuple[PhysicsObservation, PhysicsReward, bool, Dict]:
        self.current_step += 1

        if action.predicted_field is not None:
            pred = np.array(action.predicted_field, dtype=np.float32)
            if pred.ndim == 1:
                pred = pred.reshape(self.resolution)
        else:
            if self.current_step == 1:
                pred = self.initial_state.copy()
            else:
                pred = self.initial_state.copy() * 0.9 + 0.1 * 0.5

        self.current_prediction = pred

        if action.done:
            self.done = True
            reward = self._calculate_reward()
            self.last_reward = reward
            self.reward_history.append(reward.score)
            return self._get_observation(), reward, self.done, {"task": self.task_name}

        target_idx = min(self.current_step, len(self.ground_truth_trajectory) - 1)
        ground_truth = self.ground_truth_trajectory[target_idx]

        metrics = compute_metrics(pred, ground_truth)

        reward = PhysicsReward(
            score=0.0,
            mse=metrics["mse"],
            correlation=metrics["correlation"],
            error=None,
        )

        if metrics["mse"] <= self.target_mse:
            reward.score = 1.0
        elif metrics["mse"] <= self.target_mse * 3:
            reward.score = max(
                0.3, 1.0 - (metrics["mse"] - self.target_mse) / (self.target_mse * 2)
            )
        elif metrics["correlation"] > 0.7:
            reward.score = 0.5
        elif metrics["correlation"] > 0.3:
            reward.score = 0.3
        else:
            reward.score = 0.1

        self.last_reward = reward
        self.reward_history.append(reward.score)

        if self.current_step >= self.max_steps or reward.score >= 0.95:
            self.done = True

        return self._get_observation(), reward, self.done, {"task": self.task_name}

    def _calculate_reward(self) -> PhysicsReward:
        if self.current_prediction is None:
            return PhysicsReward(
                score=0.0, mse=1.0, correlation=0.0, error="No prediction made"
            )

        target_idx = min(self.current_step, len(self.ground_truth_trajectory) - 1)
        ground_truth = self.ground_truth_trajectory[target_idx]
        metrics = compute_metrics(self.current_prediction, ground_truth)

        score = 0.0
        if metrics["mse"] <= self.target_mse:
            score = 1.0
        elif metrics["mse"] <= self.target_mse * 3:
            score = max(
                0.3, 1.0 - (metrics["mse"] - self.target_mse) / (self.target_mse * 2)
            )
        elif metrics["correlation"] > 0.7:
            score = 0.5
        elif metrics["correlation"] > 0.3:
            score = 0.3
        else:
            score = 0.1

        return PhysicsReward(
            score=score, mse=metrics["mse"], correlation=metrics["correlation"]
        )

    def _get_observation(self) -> PhysicsObservation:
        current_field = (
            self.current_prediction
            if self.current_prediction is not None
            else self.initial_state
        )

        return PhysicsObservation(
            initial_state=current_field.tolist(),
            metadata=PhysicsMetadata(
                dataset="turbulent_radiative_layer_2D (The Well)",
                resolution=self.resolution,
                field_type=self.field_name,
                time_step=self.current_step,
            ),
            task_description=self.task["description"],
            hints=self.task["hints"],
        )

    def state(self) -> Dict:
        return {
            "task": self.task_name,
            "step": self.current_step,
            "done": self.done,
            "last_reward": self.last_reward.score if self.last_reward else 0.0,
            "mse": self.last_reward.mse if self.last_reward else None,
        }

    def close(self):
        pass
