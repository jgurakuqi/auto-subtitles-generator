from functools import partial
import gc
import torch
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade
from tqdm import tqdm


def load_vocals_model(device: torch.device) -> tuple[torch.nn.Module, int]:
    """Load the pre-trained HDEMUCS_HIGH_MUSDB_PLUS model.

    Args:
        device (torch.device): The device to use for processing.

    Returns:
        tuple[torch.nn.Module, int]: The model and the sample rate.
    """
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
    segment: float = 10.0,
    overlap: float = 0.1,
    device: torch.device | None = None,
    sample_rate: int = 44100,
) -> torch.Tensor:
    """Separate vocals from the rest of the audio using the half precision Hybrid Demucs model.

    Args:
        model (torch.nn.Module): The model to use for separation.
        mix (torch.Tensor): The mixed audio signal.
        segment (float, optional): The segment length in seconds. Defaults to 10.0.
        overlap (float, optional): The overlap between segments in seconds. Defaults to 0.1.
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

    for _ in tqdm(range(int(num_iterations)), desc="Processing"):
        chunk = mix[:, :, start:end]

        if chunk.shape[-1] > 0:
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.float16):
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
