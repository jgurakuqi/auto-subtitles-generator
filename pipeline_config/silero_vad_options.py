# ./pipeline_config/silero_vad_options.py


from typing import Any


class SileroVadOptions:

    use_vad_filter : bool
    settings : dict[str, Any]

    def __init__(self, config : dict[str, Any]):
        self.use_vad_filter = config["use_vad_filter"]
        self.settings = config["settings"]