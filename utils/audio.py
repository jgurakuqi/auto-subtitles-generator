from numpy import (
    ndarray as np_ndarray,
    frombuffer as np_frombuffer,
    float32 as np_float32,
    int16 as np_int16,
    save as np_save,
)
from subprocess import (
    run as subprocess_run,
    CalledProcessError as subprocess_CalledProcessError,
)
import torch, torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade
from tqdm import tqdm
from configs.pipeline_config import PipelineConfig
from utils.utils import create_folder_if_not_exists, build_path, cuda_force_collect

# import logging
# logger = logging.getLogger("auto-sub-gen")


def load_audio(file: str, sr: int, volume_factor: float = 1.6) -> np_ndarray:
    """
    Open an audio file and read as mono waveform, with an option to increase volume.

    Args:
        file (str): The path to the audio file.
        sr (int): The sample rate to resample the audio if necessary.
        volume_factor (float, optional): The factor by which to increase the volume. Defaults to 1.0 (no change).

    Returns:
        np_ndarray: A NumPy array containing the audio waveform (i.e., rescaled in interval [-1, 1] by dividing by 32768.0).
    """
    try:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-filter:a",
            f"volume={volume_factor}",
            "-",
        ]
        out = subprocess_run(cmd, capture_output=True, check=True).stdout
    except subprocess_CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np_frombuffer(out, np_int16).flatten().astype(np_float32) / 32768.0


def load_vocals_model(device: torch.device) -> tuple[torch.nn.Module, int]:
    """Load the pre-trained HDEMUCS_HIGH_MUSDB_PLUS model.

    Args:
        device (torch.device): The device to use for processing.

    Returns:
        tuple[torch.nn.Module, int]: The model and the sample rate.
    """
    # logger.debug(
    #     f"utils.vocals_separator.load_vocals_model:: Loading model on: {device}"
    # )
    bundle = HDEMUCS_HIGH_MUSDB_PLUS
    model = bundle.get_model()
    model.to(device)
    return model, bundle.sample_rate


def __convert_waveform_from_np_to_torch(
    waveform_numpy: np_ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a NumPy array to a torch Tensor and rescale it to the device.

    Args:
        waveform_numpy (np_ndarray): The waveform to convert.
        device (torch.device): The device to use for processing.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The waveform and the reference.
    """
    waveform = torch.from_numpy(waveform_numpy)
    waveform = waveform.to(device)
    if waveform.shape[0] == 1:
        waveform = torch.cat([waveform, waveform], dim=0)
    ref = waveform.mean(0)
    waveform = (waveform - ref.mean()) / ref.std()
    return waveform, ref


def __separate_vocals(
    model: torch.nn.Module,
    mix: torch.Tensor,
    segment: float,
    overlap: float,
    device: torch.device | None = None,
    sample_rate: int = 44_100,
) -> torch.Tensor:
    """Separate vocals from the rest of the audio using the half precision Hybrid Demucs model.

    Args:
        model (torch.nn.Module): The model to use for separation.
        mix (torch.Tensor): The mixed audio signal.
        segment (float, optional): The segment length in seconds.
        overlap (float, optional): The overlap between segments in seconds.
        device (torch.device | None, optional): The device to use for processing. Defaults to None.
        sample_rate (int, optional): The sample rate of the audio. Defaults to 44_100.

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

    cuda_force_collect(iterations=1)

    for _ in tqdm(
        range(int(num_iterations)),
        desc="[utils.audio.__separate_vocals]:: Separating audio chunk...",
        leave=False,
    ):
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

    fade = None
    return final


def extract_vocals_only_audio(
    videos_paths: list[str],
    pipeline_config: PipelineConfig,
):
    """Extract vocals only audio from the input audio files.

    Args:
        videos_paths (list[str]): The paths to the videos.
        pipeline_config (PipelineConfig): The configuration for the pipeline.

    Returns:
        None
    """
    waveform_numpy: np_ndarray
    waveform: torch.Tensor
    sources: torch.Tensor
    vocals: torch.Tensor
    vocals_resampled: torch.Tensor
    vocals_resampled_numpy: np_ndarray

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, sample_rate = load_vocals_model(device)

    create_folder_if_not_exists(pipeline_config.tmp_numpy_audio_folder)

    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=16_000
    ).to(device)

    for video_path in tqdm(
        videos_paths,
        desc="[utils.audio.extract_vocals_only_audio]:: Extracting vocals only audio...",
        total=len(videos_paths),
    ):
        try:
            waveform_numpy = load_audio(file=video_path, sr=sample_rate)
            waveform_numpy = waveform_numpy.reshape(1, -1)
            waveform, _ = __convert_waveform_from_np_to_torch(waveform_numpy, device)

            sources = __separate_vocals(
                model,
                waveform[None],
                device=device,
                segment=pipeline_config.vocal_separator_config.segment,
                overlap=pipeline_config.vocal_separator_config.overlap,
            )[0]

            del waveform, waveform_numpy
            cuda_force_collect(iterations=1)

            vocals = sources[model.sources.index("vocals")]

            vocals_resampled = resampler(vocals)

            # Ensure we mix stereo to mono by averaging both channels
            vocals_mono = vocals_resampled.mean(dim=0)

            # Convert to NumPy and flatten to ensure it's 1D
            vocals_resampled_numpy = vocals_mono.cpu().numpy().flatten()

            vocals_only_path = build_path(
                folder_path=pipeline_config.tmp_numpy_audio_folder,
                file_path=video_path,
                extension_replacement=".npy",
            )

            np_save(file=vocals_only_path, arr=vocals_resampled_numpy)

            (
                vocals_resampled,
                vocals_resampled_numpy,
                vocals,
                vocals_mono,
                sources,
                waveform,
                waveform_numpy,
                vocals_only_path,
            ) = (
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

            cuda_force_collect()

        except Exception as e:
            # logger.error(f"Failed to process {video_path}: {e}")
            print(f"Failed to process {video_path}: {e}")

    model, device, resampler = None, None, None
    print("VOCALS EXTRACTION COMPLETED")
    cuda_force_collect()
