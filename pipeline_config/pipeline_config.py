# ./pipeline_config/pipeline_config.py
from typing import Any
import yaml

from pipeline_config.paths import Paths
from pipeline_config.transcriber_model_config import ModelConfig
from pipeline_config.silero_vad_options import SileroVadOptions
from pipeline_config.transcription_config import TranscriptionConfig


class PipelineConfig:

    paths: Paths
    model_config: ModelConfig
    silero_vad_options: SileroVadOptions
    transcription_config: TranscriptionConfig

    def __init__(self, config_path: str):
        config = self.__load_config(config_path)

        self.paths = Paths(config=config)
        self.model_config = ModelConfig(config=config)
        self.silero_vad_options = SileroVadOptions(config=config)
        self.transcription_config = TranscriptionConfig(config=config)

    def __load_config(self, config_path: str) -> dict[str, Any]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config
