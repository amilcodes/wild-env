"""Aeolus-IA research environment."""

from .config import ExperimentConfig, ScenarioConfig, TrainingConfig, load_config

__version__ = "0.7.0"

__all__ = [
    "ExperimentConfig",
    "ScenarioConfig",
    "TrainingConfig",
    "__version__",
    "load_config",
]
