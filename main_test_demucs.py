import gc
import torch, torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS
from torchaudio.transforms import Fade
from tqdm import tqdm


def separate_vocals(
    model: torch.nn.Module,
    mix: torch.Tensor,
    segment=10.0,
    overlap=0.1,
    device=None,
    sample_rate=44100,
):
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

    # Calculate the number of iterations
    num_iterations = (length - overlap_frames) // (chunk_len - overlap_frames) + 1
    torch.cuda.empty_cache()
    gc.collect()

    # Use tqdm to create a progress bar
    for _ in tqdm(range(int(num_iterations)), desc="Processing"):
        chunk = mix[:, :, start:end]

        # Check if the chunk is non-empty
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


input_path = "./forest_test.mp4"
output_path = r"./raw_audio.wav"


# Load the pre-trained model
bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
sample_rate = bundle.sample_rate

# Load your audio file
SAMPLE_SONG = output_path
waveform, _ = torchaudio.load(SAMPLE_SONG)
waveform = waveform.to(device)
if waveform.shape[0] == 1:
    waveform = torch.cat([waveform, waveform], dim=0)  # Duplicate the channel if mono

# Normalize the waveform
ref = waveform.mean(0)
waveform = (waveform - ref.mean()) / ref.std()

# Separate the sources
sources = separate_vocals(
    model, waveform[None], device=device, segment=11, overlap=0.257
)[0]
# Free cuda memory
del waveform
torch.cuda.empty_cache()
gc.collect()

sources = sources * ref.std() + ref.mean()

# Extract the vocals
vocals = sources[model.sources.index("vocals")]

# Save the vocals to a file
torchaudio.save("./vocals_only.wav", vocals.cpu(), sample_rate)
