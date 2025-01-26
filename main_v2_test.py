# Std library imports
import logging, torch, os
from typing import Any
import numpy as np
import librosa
from tqdm import tqdm

# Local repository imports
from utils.audio_extraction import extract_audio
from utils.transcription import load_model, transcribe_speech_segments
from utils.srt_generation import generate_srt
from utils.utils import (
    recursively_read_video_paths,
    initialize_logging,
)
from utils.vocals_separator import extract_vocals_only_audio, load_audio
from utils.energy_vad import perform_energy_vad, load_vad_timestamps
from pipeline_config.pipeline_config import PipelineConfig


# Third-party library imports
from faster_whisper.transcribe import Segment, TranscriptionInfo

initialize_logging(
    logs_folder_path="./logs", log_level=logging.DEBUG, log_to_console=True
)
logger = logging.getLogger("auto-sub-gen")


def main():
    pipeline_config = PipelineConfig(config_path="./config.yaml")

    orig_video_paths = recursively_read_video_paths(pipeline_config.paths.videos_folder)

    # 1. Extract vocals only audios
    # 2. Apply custom VAD on vocals only audios
    # 3. Transcribe vocals only audios
    # 4. Generate SRT files

    sources_to_split = [
        video_path
        for video_path in orig_video_paths
        if not os.path.exists(
            os.path.join(
                pipeline_config.paths.vocals_only_folder,
                os.path.basename(video_path).replace(
                    os.path.splitext(video_path)[-1], "_vocals.wav"
                ),
            )
        )
    ]

    logger.info(f"Sources to extract vocals from: {len(sources_to_split)}")

    extract_vocals_only_audio(
        input_audio_paths=sources_to_split,
        # segment=pipeline_config..segment,
        # overlap=pipeline_config.vad.overlap,
        vocals_only_folder=pipeline_config.paths.vocals_only_folder,
    )

    # get all file paths in the folder of vocals only audios
    vocals_only_audio_paths = [
        os.path.join(pipeline_config.paths.vocals_only_folder, file_path)
        for file_path in os.listdir(pipeline_config.paths.vocals_only_folder)
    ]

    # 1. Extract vocals only audios
    # 2. Apply custom VAD on vocals only audios
    # 3. Transcribe vocals only audios
    # 4. Generate SRT files

    sources_for_timestamps = []
    FORCE_VAD = True

    for audio_path in vocals_only_audio_paths:
        timestamp_path = os.path.join(
            pipeline_config.paths.timestamps_folder,
            os.path.basename(audio_path).replace(
                os.path.splitext(audio_path)[-1], "_timestamps.json"
            ),
        )
        if not os.path.exists(timestamp_path) or FORCE_VAD:
            sources_for_timestamps.append(audio_path)
    # print(sources_for_timestamps)
    logger.info(f"Sources to perform VAD on: {len(sources_for_timestamps)}")

    perform_energy_vad(
        audio_paths=sources_for_timestamps,
        # frame_length=
        # hop_length=
        # energy_threshold=
        # seconds_per_chunk=2000,
        timestamps_folder=pipeline_config.paths.timestamps_folder,
    )

    logger.info(f"Loading transcriber model...")

    return

    model = load_model(
        pipeline_config.model_config.use_batched_inference,
        pipeline_config.model_config.model_id,
        pipeline_config.model_config.device.type,
        pipeline_config.model_config.compute_type,
        pipeline_config.model_config.num_workers,
    )

    logger.info(f"Model loaded.")

    # map in a tuple the audio paths and the original full-video paths, to check if srts already exist during the loop.
    # Match them intelligently, as the vocals_only folder might contain audio paths from different videos
    sources_to_transcribe = []
    for video_path in orig_video_paths:
        related_audio_path = os.path.join(
            pipeline_config.paths.vocals_only_folder,
            os.path.splitext(os.path.basename(video_path))[0] + "_vocals.wav",
        )
        related_timestamps_path = os.path.join(
            pipeline_config.paths.timestamps_folder,
            os.path.splitext(os.path.basename(related_audio_path))[0]
            + "_timestamps.json",
        )
        logger.debug(
            f"Related audio path: {related_audio_path}, related timestamps path: {related_timestamps_path}"
        )
        if os.path.exists(related_audio_path) and os.path.exists(
            related_timestamps_path
        ):
            sources_to_transcribe.append(
                (video_path, related_audio_path, related_timestamps_path)
            )
        else:
            logger.warning(f"No audio or timestamps found for {video_path}")

    logger.info(f"Transcribing {len(sources_to_transcribe)} audio files...")

    segments: list[Segment]
    info: list[TranscriptionInfo]
    for original_video_path, audio_path, timestamp_path in tqdm(
        sources_to_transcribe,
        total=len(sources_to_transcribe),
        desc="Transcribing audio files...",
    ):
        try:
            logger.info(f"Transcribing {audio_path}...")

            # if os.path.exists(

            # Load VAD timestamps for the current audio
            vad_timestamps = load_vad_timestamps(timestamp_path)
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
                beam_size=pipeline_config.model_config.beam_size,
                language=pipeline_config.model_config.language,
                use_vad_filter=pipeline_config.silero_vad_options.use_vad_filter,
                patience=pipeline_config.model_config.patience,
                use_word_timestamps=pipeline_config.model_config.use_word_timestamps,
                vad_settings=pipeline_config.silero_vad_options.settings,
                log_progress=pipeline_config.model_config.log_progress,
                use_batched_inference=pipeline_config.model_config.use_batched_inference,
                batch_size=pipeline_config.model_config.batch_size,
            )

            logger.info(f"Generating SRT for {audio_path}...")
            generate_srt(
                segments=segments,
                video_path=original_video_path,
                max_chars=pipeline_config.transcription_config.max_chars,
                srt_debug_mode=pipeline_config.transcription_config.debug_mode,
            )

        except Exception as e:
            logger.error(f"main::EXCEPTION::Error while transcribing {audio_path}: {e}")

    logger.info(f"All transcriptions completed.")


if __name__ == "__main__":
    main()
