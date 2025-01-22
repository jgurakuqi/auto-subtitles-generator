import os
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from utils.utils import create_folder_if_not_exists

# from simple_audio_extract import extract_audio


# Converts float seconds to mm:ss format
def format_timestamps(timestamp: dict):
    return f"{int(timestamp['start'] // 60):02d}:{int(timestamp['start'] % 60):02d} - {int(timestamp['end'] // 60):02d}:{int(timestamp['end'] % 60):02d}"


# input_path = "./forest_test.mp4"
# extracted_audio = r"./raw_audio.wav"

# if not os.path.exists(extracted_audio):
#     extract_audio(input_path, extracted_audio)

vocals_only_path = "./vocals_only.wav"

timestamps_folder = "./timestamps/"
create_folder_if_not_exists(timestamps_folder)
timestamps_full_path = os.path.join(timestamps_folder, "silero.txt")

model = load_silero_vad()
wav = read_audio(vocals_only_path, sampling_rate=44100)
speech_timestamps = get_speech_timestamps(
    wav,
    model,
    return_seconds=True,  # Return speech timestamps in seconds (default is samples)
    progress_tracking_callback=lambda x: print(
        f"Progress: {x:.2f}%", end="\r"
    ),  # rounded
    threshold=0.1,
    min_silence_duration_ms=500,
    speech_pad_ms=100,
)


with open(timestamps_full_path, "w") as f:
    for line in speech_timestamps:
        f.write(f"{format_timestamps(line)}\n")
