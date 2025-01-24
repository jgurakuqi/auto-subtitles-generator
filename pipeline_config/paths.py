# ./pipeline_config/paths.py


import os
from typing import Any
from utils.utils import create_folder_if_not_exists


class Paths:
    videos_folder: str
    vocals_only_folder: str

    def __init__(self, config: dict[str, Any]):
        self.videos_folder = config["paths"]["videos_folder"]
        self.vocals_only_folder = config["paths"]["vocals_only_folder"]
