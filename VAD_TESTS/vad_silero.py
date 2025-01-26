import os
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from utils.utils import create_folder_if_not_exists


def format_timestamps_in_hh_mm_ss(line: dict[str, float]):
    start = line["start"]
    end = line["end"]
    start_hh_mm_ss = f"{int(start // 3600):02d}:{int((start % 3600) // 60):02d}:{int(start % 60):02d}"
    end_hh_mm_ss = (
        f"{int(end // 3600):02d}:{int((end % 3600) // 60):02d}:{int(end % 60):02d}"
    )
    return f"{start_hh_mm_ss} - {end_hh_mm_ss}"


vocals_only_path = "./vocals_only/test_1_vocals.wav"

timestamps_folder = "./silero_test/"
create_folder_if_not_exists(timestamps_folder)
timestamps_full_path = os.path.join(timestamps_folder, "silero.txt")

model = load_silero_vad()
wav = read_audio(vocals_only_path, sampling_rate=26000)
speech_timestamps = get_speech_timestamps(
    wav,
    model,
    return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    progress_tracking_callback=lambda x: print(f"Progress: {x:.2f}%", end="\r"),
    threshold=0.15,
    min_silence_duration_ms=500,
    speech_pad_ms=100,
)


with open(timestamps_full_path, "w") as f:
    for line in speech_timestamps:
        f.write(f"{format_timestamps_in_hh_mm_ss(line)}\n")
