from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.transcribe import TranscriptionInfo, Segment


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
            device=device,
            compute_type=compute_type,
            num_workers=1,
        )


def transcribe_audio(
    model: WhisperModel | BatchedInferencePipeline,
    input_audio_path: str,
    beam_size: int,
    language: str,
    use_vad_filter: bool,
    batch_size: int,
    patience: int,
    use_word_timestamps: bool,
    vad_settings: dict | None,
    use_batched_inference: bool,
) -> tuple[list[Segment], TranscriptionInfo]:
    """
    Transcribe audio using a Whisper model.

    Args:
        model (WhisperModel | BatchedInferencePipeline): The Whisper model to use for transcription.
        input_audio_path (str): The path to the input audio file.
        beam_size (int): The beam size to use for transcription.
        language (str): The language to use for transcription.
        use_vad_filter (bool): Whether to use voice activity detection.
        batch_size (int): The batch size to use for transcription.
        use_word_timestamps (bool): Whether to use word timestamps.
        vad_settings (dict | None): The voice activity detection settings.
        use_batched_inference (bool): Whether to use batched inference.

    Returns:
        tuple[list[Segment], TranscriptionInfo]: The segments and the transcription info.
    """
    if use_batched_inference:
        return model.transcribe(
            audio=input_audio_path,
            beam_size=beam_size,
            language=language,
            vad_filter=use_vad_filter,
            patience=patience,
            batch_size=batch_size,
            word_timestamps=use_word_timestamps,
            vad_parameters=vad_settings,
        )
    else:
        return model.transcribe(
            audio=input_audio_path,
            beam_size=beam_size,
            language=language,
            vad_filter=use_vad_filter,
            word_timestamps=use_word_timestamps,
            vad_parameters=vad_settings,
            patience=patience,
        )
