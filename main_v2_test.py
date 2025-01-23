# Std library imports
import gc
import logging, torch, os
from typing import Iterable, List, Tuple
import json
import numpy as np
import librosa
from tqdm import tqdm

# Local repository imports
from utils.audio_extraction import extract_audio
from utils.transcription import load_model, transcribe_audio
from utils.srt_generation import generate_srt
from utils.utils import (
    recursively_read_video_paths,
    initialize_logging,
    create_folder_if_not_exists,
)

from preprocessing.vocals_separator import (
    save_vocals,
    separate_vocals,
    load_audio,
    load_vocals_model,
)

# Third-party library imports
from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.transcribe import Segment, TranscriptionInfo

initialize_logging(
    logs_folder_path="./logs", log_level=logging.DEBUG, log_to_console=True
)
logger = logging.getLogger("auto-sub-gen")


def load_vad_timestamps(vad_json_path: str) -> List[Tuple[float, float]]:
    """Load VAD timestamps from a JSON file."""
    with open(vad_json_path, "r") as file:
        vad_data = json.load(file)
    return [(segment["start"], segment["end"]) for segment in vad_data]


def slice_audio(
    audio_waveform: np.ndarray, sample_rate: int, timestamps: List[Tuple[float, float]]
) -> List[np.ndarray]:
    """Slice audio waveform into segments based on timestamps."""
    segments = []
    for start, end in timestamps:
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)
        segments.append(audio_waveform[start_sample:end_sample])
    return segments


def transcribe_speech_segments(
    model: BatchedInferencePipeline | WhisperModel,
    audio_waveform: np.ndarray,
    sample_rate: int,
    vad_timestamps: List[Tuple[float, float]],
    beam_size: int,
    language: str,
    use_vad_filter: bool,
    patience: float,
    use_word_timestamps: bool,
    vad_settings: dict | None,
    log_progress: bool,
    use_batched_inference: bool,
    batch_size: int,
):
    """Transcribe only the speech segments.

    Args:
        model (BatchedInferencePipeline | WhisperModel): The Whisper model.
        audio_waveform (np.ndarray): The audio waveform.
        sample_rate (int): The sample rate of the audio.
        vad_timestamps (List[Tuple[float, float]]): The VAD timestamps.
        beam_size (int): The beam size for the transcriptions.
        language (str): The language for the transcriptions.
        use_vad_filter (bool): Whether to use the VAD filter.
        patience (float): The patience for the transcriptions.
        use_word_timestamps (bool): Whether to use word timestamps.
        vad_settings (dict | None): The VAD settings.
        log_progress (bool): Whether to show progress bar.
        use_batched_inference (bool): Whether to use batched inference.
        batch_size (int): The batch size for the transcriptions. Useful for batched inference.

    Returns:
        tuple[List[Segment], List[TranscriptionInfo]]: The transcribed segments and their info.
    """
    speech_segments = slice_audio(audio_waveform, sample_rate, vad_timestamps)
    all_segments = []
    all_info = []

    segments: Iterable[Segment]
    info: TranscriptionInfo
    speech_segment: np.ndarray

    for speech_segment in tqdm(
        speech_segments,
        total=len(speech_segments),
        desc="Transcribing speech segments: ",
    ):

        segments, info = transcribe_audio(
            model=model,
            audio=speech_segment,
            beam_size=beam_size,
            language=language,
            use_vad_filter=use_vad_filter,
            batch_size=batch_size,
            patience=patience,
            use_word_timestamps=use_word_timestamps,
            vad_settings=vad_settings,
            use_batched_inference=use_batched_inference,
            log_progress=log_progress,
        )

        for segment in segments:
            all_segments.append(segment)

        all_info.append(info)

    return all_segments, all_info


def main():
    FOLDER_VIDEOS_PATH = r"/mnt/c/Users/jgura/Downloads/NaruCannon/Dub/2_Chunin Exams/"

    # Model parameters
    USE_BATCHED_INFERENCE = False
    MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16"  # "int8_float16"
    NUM_WORKERS = 1
    BEAM_SIZE = 8
    PATIENCE = 1.25
    LANGUAGE = "en"
    LOG_PROGRESS = True  # print progress bar during transcription
    USE_WORD_TIMESTAMPS = True
    BATCH_SIZE = 22

    # VAD settings:
    USE_VAD_FILTER = False
    # VAD_SETTINGS = { "threshold": 0.2, "neg_threshold": 0.15, "min_silence_duration_ms": 1000, "speech_pad_ms": 100 }
    VAD_SETTINGS = None  # ! --> Using Demucs+Custom VAD instead

    # Transcription parameters
    MAX_CHARS = 60
    ATOMIC_TRANSCRIPTION = False

    # SRT generation parameters
    SRT_DEBUG_MODE = True

    # Audio extraction parameters
    FORCE_EXTRACT = False
    ATOMIC_AUDIO_EXTRACTION = True
    AUDIO_FOLDER = "./audio_output/"
    FFMPEG_THREADS_NUM = 6

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

    _ = extract_audio(
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
