# ./utils/transcription.py

import logging
from logging import Logger
from typing import BinaryIO, Iterable, cast
from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.transcribe import TranscriptionInfo, Segment
import numpy as np
from tqdm import tqdm

logger = logging.getLogger("auto-sub-gen")


def load_batched_whisper(
    model_id: str, device: str, compute_type: str, num_workers: int
) -> BatchedInferencePipeline:
    """
    Load a batched Whisper model for inference with the given parameters.

    Args:
        model_id (str): Id of the Whisper model.
        device (str): device to use. "cpu" or "cuda".
        compute_type (str): Precision to use. Possible values: "float16", "float32", "int8_float16".
        num_workers (int): Number of workers to use.

    Returns:
        BatchedInferencePipeline: A batched Whisper model for inference.
    """
    return BatchedInferencePipeline(
        WhisperModel(
            model_id, device=device, compute_type=compute_type, num_workers=num_workers
        )
    )


def load_linear_whisper(
    model_id: str, device: str, compute_type: str, num_workers: int
) -> WhisperModel:
    """
    Load a Whisper model for inference with the given parameters.

    Args:
        model_id (str): Id of the Whisper model.
        device (str): device to use. "cpu" or "cuda".
        compute_type (str): Precision to use. Possible values: "float16", "float32", "int8_float16".
        num_workers (int): Number of workers to use.

    Returns:
        WhisperModel: A Whisper model for inference.
    """
    return WhisperModel(
        model_id, device=device, compute_type=compute_type, num_workers=num_workers
    )


def load_model(
    use_batched_inference: bool,
    model_id: str,
    device: str,
    compute_type: str,
    num_workers: int,
) -> WhisperModel | BatchedInferencePipeline:
    """
    Load a Whisper model for inference with the given parameters.

    Args:
        model_id (str): Id of the Whisper model.
        device (str): device to use. "cpu" or "cuda".
        compute_type (str): Precision to use. Possible values: "float16", "float32", "int8_float16".
        num_workers (int): Number of workers to use.

    Returns:
        WhisperModel | BatchedInferencePipeline: A Whisper model for inference.
    """
    if use_batched_inference:
        return load_batched_whisper(
            model_id=model_id,
            device=device,
            compute_type=compute_type,
            num_workers=num_workers,
        )
    else:
        return load_linear_whisper(
            model_id=model_id,
            device="cpu",
            compute_type=compute_type,
            num_workers=1,
        )


def transcribe_audio(
    model: WhisperModel | BatchedInferencePipeline,
    audio: str | BinaryIO | np.ndarray,
    beam_size: int,
    language: str,
    use_vad_filter: bool,
    batch_size: int,
    patience: float,
    use_word_timestamps: bool,
    vad_settings: dict | None,
    use_batched_inference: bool,
    log_progress: bool,
) -> tuple[Iterable[Segment], TranscriptionInfo]:
    """
    Transcribe audio using a Whisper model.

    Args:
        model (WhisperModel | BatchedInferencePipeline): The Whisper model to use for transcription.
        audio (str | BinaryIO | np.ndarray): The audio data to transcribe. Can be a path to a file, a file-like object, or a numpy array.
        beam_size (int): The beam size to use for transcription.
        language (str): The language to use for transcription.
        use_vad_filter (bool): Whether to use voice activity detection.
        batch_size (int): The batch size to use for transcription.
        use_word_timestamps (bool): Whether to use word timestamps.
        vad_settings (dict | None): The voice activity detection settings.
        use_batched_inference (bool): Whether to use batched inference.
        log_progress (bool): Whether to show progress bar.

    Raises:
        Exception: If an error occurs during transcription.

    Returns:
        tuple[list[Segment], TranscriptionInfo]: The segments and the transcription info.
    """
    try:
        if use_batched_inference:

            return cast(BatchedInferencePipeline, model).transcribe(
                audio=audio,
                beam_size=beam_size,
                language=language,
                vad_filter=use_vad_filter,
                patience=patience,
                batch_size=batch_size,
                word_timestamps=use_word_timestamps,
                vad_parameters=vad_settings,
                log_progress=log_progress,
            )
        else:
            return model.transcribe(
                audio=audio,
                beam_size=beam_size,
                language=language,
                vad_filter=use_vad_filter,
                word_timestamps=use_word_timestamps,
                vad_parameters=vad_settings,
                patience=patience,
                log_progress=log_progress,
            )
    except Exception as e:
        logger.error(
            f"utils.transcription.transcribe_audio::EXCEPTION::{e}", exc_info=True
        )
        raise e


def slice_audio(
    audio_waveform: np.ndarray, sample_rate: int, timestamps: list[tuple[float, float]]
) -> list[np.ndarray]:
    """Slice audio waveform into segments based on timestamps.

    Args:
        audio_waveform (np.ndarray): The audio waveform.
        sample_rate (int): The sample rate of the audio.
        timestamps (list[tuple[float, float]]): The timestamps of the segments.

    Returns:
        list[np.ndarray]: The segments of the audio waveform.
    """
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
    vad_timestamps: list[tuple[float, float]],
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
