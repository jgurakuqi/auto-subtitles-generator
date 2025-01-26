import os
from subprocess import run as subprocess_run
import torch, torchaudio
from preprocessing.vocals_separator import load_audio
import gc
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade
from tqdm import tqdm

from utils.utils import create_folder_if_not_exists, build_path

import logging

logger = logging.getLogger("auto-sub-gen")


def load_vocals_model(device: torch.device) -> tuple[torch.nn.Module, int]:
    """Load the pre-trained HDEMUCS_HIGH_MUSDB_PLUS model.

    Args:
        device (torch.device): The device to use for processing.

    Returns:
        tuple[torch.nn.Module, int]: The model and the sample rate.
    """
    logger.debug(
        f"utils.vocals_separator.load_vocals_model:: Loading model on: {device}"
    )
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    model.to(device)
    return model, bundle.sample_rate


def load_audio(
    file_path: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and normalize the audio file.

    Args:
        file_path (str): The path to the audio file.
        device (torch.device): The device to use for processing.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The audio waveform and the reference signal.
    """
    logger.debug(
        f"utils.vocals_separator.load_audio:: Loading audio: {file_path} on {device}"
    )
    waveform, _ = torchaudio.load(file_path)
    waveform = waveform.to(device)
    if waveform.shape[0] == 1:
        waveform = torch.cat(
            [waveform, waveform], dim=0
        )  # Duplicate the channel if mono
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()
    return waveform, ref


def separate_vocals(
    model: torch.nn.Module,
    mix: torch.Tensor,
    segment: float,
    overlap: float,
    device: torch.device | None = None,
    sample_rate: int = 44100,
) -> torch.Tensor:
    """Separate vocals from the rest of the audio using the half precision Hybrid Demucs model.

    Args:
        model (torch.nn.Module): The model to use for separation.
        mix (torch.Tensor): The mixed audio signal.
        segment (float, optional): The segment length in seconds.
        overlap (float, optional): The overlap between segments in seconds.
        device (torch.device | None, optional): The device to use for processing. Defaults to None.
        sample_rate (int, optional): The sample rate of the audio. Defaults to 44100.

    Returns:
        torch.Tensor: The separated vocals.
    """
    if device is None:
        device = mix.device
    else:
        device = torch.device(device)

    batch, channels, length = mix.shape
    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, len(model.sources), channels, length, device=device)

    num_iterations = (length - overlap_frames) // (chunk_len - overlap_frames) + 1
    torch.cuda.empty_cache()
    gc.collect()

    for _ in tqdm(range(int(num_iterations)), desc="Separating audio chunk..."):
        chunk = mix[:, :, start:end]

        if chunk.shape[-1] > 0:
            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    out = model.forward(chunk)
            out = fade(out)
            final[:, :, :, start:end] += out

        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0
    return final


def save_vocals(
    vocals: torch.Tensor, ref: torch.Tensor, output_path: str, sample_rate: int
) -> None:
    """Save the extracted vocals to a file.

    Args:
        vocals (torch.Tensor): The vocals to save.
        ref (torch.Tensor): The reference signal.
        output_path (str): The path to save the vocals.
        sample_rate (int): The sample rate of the audio.

    Returns:
        None
    """
    vocals = vocals * ref.std() + ref.mean()
    torchaudio.save(output_path, vocals.cpu(), sample_rate)


# def extract_vocals_only_audio(
#     input_audio_paths: list[str],
#     segment: int = 11,
#     overlap: float = 0.257,
#     vocals_only_folder: str = "./vocals_only/",
# ):
#     """Extract vocals only audio from the input audio files.

#     Args:
#         input_audio_paths (list[str]): List of input audio file paths.
#         segment (int, optional): Segment length in seconds. Defaults to 11.
#         overlap (float, optional): Overlap between segments. Defaults to 0.257.
#         vocals_only_folder (str, optional): Output folder for vocals only audio. Defaults to "./vocals_only".

#     Returns:
#         None
#     """
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model, sample_rate = load_vocals_model(device)

#     create_folder_if_not_exists(vocals_only_folder)

#     for input_path in input_audio_paths:
#         output_audio_path = build_path(
#             folder_path=vocals_only_folder,
#             file_path=input_path,
#             extension_replacement="_vocals.wav",
#         )
#         waveform, ref = load_audio(input_path, device)

#         sources = separate_vocals(
#             model, waveform[None], device=device, segment=segment, overlap=overlap
#         )[0]

#         del waveform
#         torch.cuda.empty_cache()
#         gc.collect()

#         vocals = sources[model.sources.index("vocals")]
#         save_vocals(vocals, ref, output_audio_path, sample_rate)


def extract_audio_with_ffmpeg(input_path: str, output_path: str) -> str:
    """
    Extract audio from the input file using ffmpeg.

    Args:
        input_path (str): Path to the input media file
        output_path (str): Path to save the extracted audio

    Returns:
        str: Path to the extracted audio file
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        ffmpeg_command = [
            "ffmpeg",
            "-i",
            input_path,
            "-vn",  # Disable video output
            "-acodec",
            "pcm_s16le",  # Set audio codec to PCM (for WAV)
            "-ar",
            "41000",  # Set audio sample rate to 41kHz
            output_path,
        ]
        result = subprocess_run(ffmpeg_command, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error(f"FFmpeg error: {result.stderr}")
            raise Exception(f"Failed to extract audio: {result.stderr}")

        return output_path

    except Exception as e:
        logger.error(f"Error in extract_audio_with_ffmpeg: {e}")
        raise


def extract_vocals_only_audio(
    input_audio_paths: list[str],
    segment: int = 11,
    overlap: float = 0.257,
    vocals_only_folder: str = "./vocals_only/",
    temp_audio_folder: str = "./temp_audio/",
):
    """Extract vocals only audio from the input audio files.

    Args:
        input_audio_paths (list[str]): List of input audio file paths.
        segment (int, optional): Segment length in seconds. Defaults to 11.
        overlap (float, optional): Overlap between segments. Defaults to 0.257.
        vocals_only_folder (str, optional): Output folder for vocals only audio. Defaults to "./vocals_only".
        temp_audio_folder (str, optional): Temporary folder for extracted audio. Defaults to "./temp_audio/".

    Returns:
        None
    """
    if input_audio_paths is None or len(input_audio_paths) == 0:
        logger.warning("No input audio files provided.")
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, sample_rate = load_vocals_model(device)

    create_folder_if_not_exists(vocals_only_folder)
    create_folder_if_not_exists(temp_audio_folder)

    for input_path in input_audio_paths:
        # Create paths
        temp_audio_path = build_path(
            folder_path=temp_audio_folder,
            file_path=input_path,
            extension_replacement=".wav",
        )
        output_audio_path = build_path(
            folder_path=vocals_only_folder,
            file_path=input_path,
            extension_replacement="_vocals.wav",
        )

        try:
            # Extract audio with FFmpeg
            extracted_audio_path = extract_audio_with_ffmpeg(
                input_path, temp_audio_path
            )

            # Load extracted audio
            waveform, ref = load_audio(extracted_audio_path, device)

            sources = separate_vocals(
                model, waveform[None], device=device, segment=segment, overlap=overlap
            )[0]

            del waveform
            if "cuda" in device.type:
                torch.cuda.empty_cache()
            gc.collect()

            vocals = sources[model.sources.index("vocals")]
            save_vocals(vocals, ref, output_audio_path, sample_rate)

        except Exception as e:
            logger.error(f"Failed to process {input_path}: {e}")

        finally:
            # Clean up temporary audio file
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
