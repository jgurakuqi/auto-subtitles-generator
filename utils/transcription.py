# Std libs
from tqdm import tqdm
from os.path import basename as os_path_basename, join as os_path_join
from numpy import load as np_load


# 3rd party libs
from configs.pipeline_config import PipelineConfig
from whisperx import load_model as whisperx_load_model

# Local libs
from utils.utils import (
    store_json,
    cuda_force_collect,
    get_filename_with_new_extension,
    build_path,
)


def transcribe_audios(
    audio_paths: list[str],
    pipeline_config: PipelineConfig,
    enable_debug_prints: bool,
) -> None:
    """
    Transcribe a list of audio files using Whisperx.

    Args:
        audio_paths (list[str]): A list of audio file paths.
        pipeline_config (PipelineConfig): The configuration for the pipeline.
        enable_debug_prints (bool): Enable debug prints.
    """

    print("--- Loading whisper model ---")
    model = whisperx_load_model(
        whisper_arch=pipeline_config.whisper_config.model_id,
        device=pipeline_config.whisper_config.device,
        compute_type=pipeline_config.whisper_config.compute_type,
        download_root=pipeline_config.whisper_config.download_root,
        language=pipeline_config.whisper_config.language,
    )

    print("--- Transcribing audios ---")

    for audio_path in tqdm(
        audio_paths, total=len(audio_paths), desc="Transcribing audios..."
    ):
        if enable_debug_prints:
            print(
                "utils.transcriptor.transcribe_audios:: Processing audio: ",
                os_path_basename(audio_path),
            )

        numpy_path = build_path(
            folder_path=pipeline_config.tmp_numpy_audio_folder,
            file_path=audio_path,
            extension_replacement=".npy",
        )

        try:
            # audio = whisperx.load_audio(audio_path)
            audio = np_load(numpy_path)
        except:
            print(
                "utils.transcriptor.transcribe_audios:: ERROR: Failed to load audio: ",
                os_path_basename(audio_path),
            )
            continue

        result = model.transcribe(
            audio,
            batch_size=pipeline_config.whisper_config.batch_size,
            language=pipeline_config.whisper_config.language,
        )

        audio = None

        # Saving the audio and the transcript
        # numpy_path = os_path_join( tmp_np_audio_folder, get_filename_with_new_extension(audio_path, "npy"), )
        intermediate_result_path = os_path_join(
            pipeline_config.tmp_intermediate_result_folder,
            get_filename_with_new_extension(audio_path, "json"),
        )

        # np_save( file=numpy_path, arr=audio, )
        store_json(data=result, output_path=intermediate_result_path)

        result = None
        cuda_force_collect(iterations=1)

    model = None
    cuda_force_collect(iterations=3, sleep_time=0.2)
