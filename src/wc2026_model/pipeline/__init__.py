"""Training and prediction pipelines."""

from .baseline import BaselineTrainingConfig, build_training_frame, train_baseline_model
from .live import predict_world_cup_fixtures, save_world_cup_fixture_predictions

__all__ = [
    "BaselineTrainingConfig",
    "build_training_frame",
    "predict_world_cup_fixtures",
    "save_world_cup_fixture_predictions",
    "train_baseline_model",
]
