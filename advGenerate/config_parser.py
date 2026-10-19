# src/advGenerate/config_parser.py

import yaml
import inspect
from advGenerate.rl_types import RLArgs, RLTrainerConfig, RLEnvConfig, RLModelConfig
from advGenerate.rl_types import BERTArgs, BERTTrainerConfig, BERTModelConfig

from typing import Dict, Any, Type


def filter_kwargs(cls: Type[Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Filters dictionary keys to only include those valid for the class constructor."""
    valid_keys = inspect.signature(cls).parameters
    return {k: v for k, v in kwargs.items() if k in valid_keys}


def load_rl_config(yaml_path: str) -> RLArgs:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return RLArgs(
        trainer=RLTrainerConfig(**filter_kwargs(RLTrainerConfig, data["trainer"])),
        env=RLEnvConfig(**filter_kwargs(RLEnvConfig, data["env"])),
        model=RLModelConfig(**filter_kwargs(RLModelConfig, data["model"])),
    )

def load_bert_config(yaml_path: str) -> BERTArgs:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    # Applying the filter here prevents the 'load_save_pth' error
    return BERTArgs(
        trainer=BERTTrainerConfig(**filter_kwargs(BERTTrainerConfig, data.get("trainer", {}))),
        model=BERTModelConfig(**filter_kwargs(BERTModelConfig, data.get("model", {}))),
    )
