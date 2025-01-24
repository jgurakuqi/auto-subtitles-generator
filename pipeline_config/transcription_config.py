# ./pipeline_config/transcription.py


from typing import Any


class TranscriptionConfig:

    max_chars : int
    debug_mode : bool

    def __init__(self, config : dict[str, Any]):
        self.max_chars = config["transcription"]["max_chars"]
        self.debug_mode = config["transcription"]["debug_mode"]