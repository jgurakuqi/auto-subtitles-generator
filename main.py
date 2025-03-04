from configs.aligner_config import AlignerConfing
from configs.pipeline_config import PipelineConfig
from configs.vocal_separator_config import VocalSeparatorConfig
from configs.whisper_config import WhisperConfig
from pipeline import pipeline


if __name__ == "__main__":
    device = "cuda"  # or "cpu"
    language = "en"  # or any language supported by Faster-Whisper
    input_folder_path = r"/mnt/c/Users/jgura/Desktop/NARUTO/16_"
    files_to_process = []  # ["Kakashi vs. Itachi"]
    # files_to_process = ["The Ultimate Art"]

    whisper_config = WhisperConfig(
        model_id="deepdml/faster-whisper-large-v3-turbo-ct2",
        compute_type="float16",  # or "int8"
        device=device,
        batch_size=23,  # 23 takes up to ~7.9 gb of VRAM with float16
        download_root="./model/",
        language=language,
    )

    aligner_config = AlignerConfing(
        device=device,
        model_dir="./align_model/",
        language=language,
        print_progress=True,
    )

    vocal_separator_config = VocalSeparatorConfig(overlap=0.257, segment=11)

    pipeline_config = PipelineConfig(
        vocal_separator_config=vocal_separator_config,
        whisper_config=whisper_config,
        aligner_config=aligner_config,
    )

    pipeline(
        input_folder_path=input_folder_path,
        pipeline_config=pipeline_config,
        enable_reverse_sorting=True,
        files_to_process=files_to_process,
        enable_debug_prints=True,
    )
