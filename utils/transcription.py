# Std libs
from tqdm import tqdm
import os

from numpy import load as np_load


# 3rd party libs
from whisperx import load_model as whisperx_load_model

# Local libs
from configs.whisper_config import WhisperConfig
from utils.utils import (
    store_json,
    cuda_force_collect,
    get_filename_with_new_extension,
    build_path,
)


def transcribe_audios(
    whisper_config: WhisperConfig,
    audio_paths: list[str],
    tmp_transcripts_folder: str,
    tmp_np_audio_folder: str,
    enable_debug_prints: bool,
) -> list[str]:

    print("--- Loading whisper model ---")
    model = whisperx_load_model(
        whisper_arch=whisper_config.model_id,
        device=whisper_config.device,
        compute_type=whisper_config.compute_type,
        download_root=whisper_config.download_root,
        language=whisper_config.language,
    )

    print("--- Transcribing audios ---")

    for audio_path in tqdm(
        audio_paths, total=len(audio_paths), desc="Transcribing audios..."
    ):
        if enable_debug_prints:
            print(
                "utils.transcriptor.transcribe_audios:: Processing audio: ",
                os.path.basename(audio_path),
            )

        numpy_path = build_path(
            folder_path=tmp_np_audio_folder,
            file_path=audio_path,
            extension_replacement=".npy",
        )

        try:
            # audio = whisperx.load_audio(audio_path)
            audio = np_load(numpy_path)
        except:
            print(
                "utils.transcriptor.transcribe_audios:: ERROR: Failed to load audio: ",
                os.path.basename(audio_path),
            )
            continue

        result = model.transcribe(
            audio,
            batch_size=whisper_config.batch_size,
            language=whisper_config.language,
        )

        audio = None

        # Saving the audio and the transcript
        # numpy_path = os.path.join( tmp_np_audio_folder, get_filename_with_new_extension(audio_path, "npy"), )
        intermediate_result_path = os.path.join(
            tmp_transcripts_folder,
            get_filename_with_new_extension(audio_path, "json"),
        )

        # np_save( file=numpy_path, arr=audio, )
        store_json(data=result, output_path=intermediate_result_path)

        result = None
        cuda_force_collect()

    model = None
    cuda_force_collect()
