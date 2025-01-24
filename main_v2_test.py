# Std library imports
import logging, torch, os
from typing import Any
import numpy as np
import librosa

# Local repository imports
from utils.audio_extraction import extract_audio
from utils.transcription import load_model
from utils.srt_generation import generate_srt
from utils.utils import (
    recursively_read_video_paths,
    initialize_logging,
)
from utils.vocals_separator import extract_vocals_only_audio, load_audio
from utils.energy_vad import perform_energy_vad, load_vad_timestamps

# Third-party library imports
from faster_whisper.transcribe import Segment, TranscriptionInfo

initialize_logging(
    logs_folder_path="./logs", log_level=logging.DEBUG, log_to_console=True
)
logger = logging.getLogger("auto-sub-gen")


import torch


# Yaml content:
# paths:
#   videos_folder: "/mnt/c/Users/jgura/Downloads/NaruCannon/Dub/2_Chunin Exams/"
#   vocals_only_folder : "./vocals_only/"

# model:
#   use_batched_inference: false
#   model_id: "deepdml/faster-whisper-large-v3-turbo-ct2"
#   # device: "cuda" # This will be dynamically determined in code if not enabled here.
#   compute_type: "float16"
#   beam_size: 8
#   patience: 1.25
#   language: "en"
#   log_progress: true
#   use_word_timestamps: true
#   batched_inference_params:
#     batch_size: 22
#     num_workers: 1

# vad:
#   use_vad_filter: false
#   settings: null # Using Demucs+Custom VAD instead --> otherwise set use_vad_filter to True and initize VAD settings as indicated on Faster Whisper repo

# transcription:
#   max_chars: 60
#   atomic_transcription: false

# srt:
#   debug_mode: true

# audio_extraction:
#   force_extract: false
#   atomic_extraction: true
#   ffmpeg_threads: 6


def main():
    config_path = "./config.yaml"
    config = load_config(config_path)
    video_folder = config["paths"]["videos_folder"]
    vocals_only_folder = config["paths"]["vocals_only_folder"]

    orig_video_paths = recursively_read_video_paths(FOLDER_VIDEOS_PATH)

    logger.debug(f"Found video_paths:\n")
    for video_path in orig_video_paths:
        logger.debug(f"{video_path}\n")

    video_paths_to_extract = [
        video_path
        for video_path in orig_video_paths
        if FORCE_EXTRACT
        or not os.path.exists(
            os.path.join(
                AUDIO_FOLDER,
                os.path.basename(video_path).replace(
                    os.path.splitext(video_path)[-1], ".wav"
                ),
            )
        )
    ]

    extract_audio(
        audio_extraction_paths=video_paths_to_extract,
        force_extract=FORCE_EXTRACT,
        num_workers=FFMPEG_THREADS_NUM,
        atomic_operation=ATOMIC_AUDIO_EXTRACTION,
        audio_output_folder=AUDIO_FOLDER,
    )

    logger.info(f"Loading model...")

    model = load_model(
        USE_BATCHED_INFERENCE, MODEL_ID, DEVICE, COMPUTE_TYPE, NUM_WORKERS
    )

    logger.info(f"Model loaded.")

    all_audio_paths = [
        os.path.join(
            AUDIO_FOLDER,
            os.path.basename(video_path).replace(
                os.path.splitext(video_path)[-1], ".wav"
            ),
        )
        for video_path in orig_video_paths
    ]

    logger.info(f"Starting transcription of subtitles...")
    for audio_path in all_audio_paths:
        logger.debug(f"Fetchable audio: {audio_path}...")

    segments: list[Segment]
    info: list[TranscriptionInfo]
    for audio_path in all_audio_paths:
        try:
            logger.info(f"Transcribing {audio_path}...")

            # Load VAD timestamps for the current audio
            vad_timestamps = load_vad_timestamps("./timestamps/vad_timestamps.json")
            audio_waveform, _ = load_audio(audio_path, torch.device("cpu"))

            if isinstance(audio_waveform, torch.Tensor):
                audio_waveform = audio_waveform.cpu().numpy()

            if audio_waveform.ndim > 1:
                audio_waveform = np.mean(audio_waveform, axis=0)

            target_sample_rate = 16000
            original_sample_rate = 41000
            audio_waveform = librosa.resample(
                audio_waveform,
                orig_sr=original_sample_rate,
                target_sr=target_sample_rate,
            )

            segments, info = transcribe_speech_segments(
                model=model,
                audio_waveform=audio_waveform,
                sample_rate=16000,
                vad_timestamps=vad_timestamps,
                beam_size=BEAM_SIZE,
                language=LANGUAGE,
                use_vad_filter=USE_VAD_FILTER,
                patience=PATIENCE,
                use_word_timestamps=USE_WORD_TIMESTAMPS,
                vad_settings=VAD_SETTINGS,
                log_progress=LOG_PROGRESS,
                use_batched_inference=USE_BATCHED_INFERENCE,
                batch_size=BATCH_SIZE,
            )

            logger.info(f"Generating SRT for {audio_path}...")
            generate_srt(
                segments=segments,
                audio_path=audio_path,
                max_chars=MAX_CHARS,
                srt_debug_mode=SRT_DEBUG_MODE,
            )

        except Exception as e:
            logger.error(f"main::EXCEPTION::Error while transcribing {audio_path}: {e}")
            if ATOMIC_TRANSCRIPTION:
                return

    logger.info(f"Done.")


if __name__ == "__main__":
    main()
