# ./pipeline_config/model_config.py


from typing import Any
from torch import device as torch_device
from torch.cuda import is_available as torch_cuda_is_available


class ModelConfig:

    model_id : str
    device : torch_device
    compute_type : str
    beam_size : int
    patience : int
    language : str
    log_progress : bool
    use_word_timestamps : bool
    use_batched_inference : bool
    batch_size : int | None
    num_workers : int | None


    def __init__(self, config : dict[str, Any]):
        self.model_id = config["transcriber_model"]["model_id"]
        self.device = config["transcriber_model"].get("device", torch_device("cuda" if torch_cuda_is_available() else "cpu"))
        self.compute_type = config["transcriber_model"]["compute_type"]
        self.beam_size = config["transcriber_model"]["beam_size"]
        self.patience = config["transcriber_model"]["patience"]
        self.language = config["transcriber_model"]["language"]
        self.log_progress = config["transcriber_model"]["log_progress"]
        self.use_word_timestamps = config["transcriber_model"]["use_word_timestamps"]
        self.use_batched_inference = config["transcriber_model"]["use_batched_inference"]
        if self.use_batched_inference:
            self.batch_size = config["transcriber_model"]["batched_inference_params"]["batch_size"]
            self.num_workers = config["transcriber_model"]["batched_inference_params"]["num_workers"]
        else:
            self.batch_size = None
            self.num_workers = None
