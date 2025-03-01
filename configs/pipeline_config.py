from configs.aligner_config import AlignerConfing
from configs.vocal_separator_config import VocalSeparatorConfig
from configs.whisper_config import WhisperConfig


class PipelineConfig:

    vocal_separator_config: VocalSeparatorConfig
    whisper_config: WhisperConfig
    aligner_config: AlignerConfing
    tmp_numpy_audio_folder: str
    tmp_intermediate_result_folder: str
    supported_video_extensions: list[str]

    def __init__(
        self,
        vocal_separator_config: VocalSeparatorConfig,
        whisper_config: WhisperConfig,
        aligner_config: AlignerConfing,
        tmp_numpy_audio_folder: str = "./tmp_numpy_audio/",
        tmp_intermediate_result_folder: str = "./tmp_intermediate_result/",
        supported_video_extensions: list[str] = ["mp4", "avi", "mov", "wmv", "mkv"],
    ) -> None:
        self.vocal_separator_config = vocal_separator_config
        self.whisper_config = whisper_config
        self.aligner_config = aligner_config
        self.tmp_numpy_audio_folder = tmp_numpy_audio_folder
        self.tmp_intermediate_result_folder = tmp_intermediate_result_folder
        self.supported_video_extensions = supported_video_extensions
