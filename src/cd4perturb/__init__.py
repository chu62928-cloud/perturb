"""Auditable CD4 Perturb-seq planning pipeline.

The package deliberately keeps orchestration and statistical baselines free of
deep-model dependencies.  STATE, STACK, PRiMeFlow and GEARS are called through
separate adapters only after the single-step gates have passed.
"""

__version__ = "0.1.0"

from .config import PipelineConfig, load_config
from .contracts import EvaluationStatus, TargetContract

__all__ = ["PipelineConfig", "load_config", "EvaluationStatus", "TargetContract"]
