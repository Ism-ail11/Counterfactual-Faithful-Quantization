from __future__ import annotations

import copy
from pathlib import Path

from cfq.cli import config_from_mapping
from cfq.config import ExperimentConfig
from cfq.utils import load_yaml


def base_config(path: str | Path = "configs/base.yaml") -> ExperimentConfig:
    return config_from_mapping(load_yaml(path))


def clone_config(config: ExperimentConfig) -> ExperimentConfig:
    return copy.deepcopy(config)
