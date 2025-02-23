class WhisperConfig:

    model_id: str
    compute_type: str
    device: str
    batch_size: int
    download_root: str
    language: str

    def __init__(
        self,
        model_id: str,
        compute_type: str,
        device: str,
        batch_size: int,
        download_root: str,
        language: str,
    ):
        self.model_id = model_id
        self.compute_type = compute_type
        self.device = device
        self.batch_size = batch_size
        self.download_root = download_root
        self.language = language
