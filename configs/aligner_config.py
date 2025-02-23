class AlignerConfing:

    model_dir: str
    device: str
    language: str
    print_progress: bool

    def __init__(
        self, model_dir: str, device: str, language: str, print_progress: bool
    ):
        self.model_dir = model_dir
        self.device = device
        self.language = language
        self.print_progress = print_progress
