# ./pipeline_config/vocal_separator_model_config.py

from typing import Any


class VocalSeparatorModelConfig:
    def __init__(self, config : dict[str, Any]):
        self.use_half_precision = config["vocal_separator_model"]["use_half_precision"]
        self.sample_rate = config["vocal_separator_model"]["sample_rate"]
        self.segment = config["vocal_separator_model"]["segment"]
        self.overlap = config["vocal_separator_model"]["overlap"]
        