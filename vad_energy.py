# import os
# from time import time
# import librosa
# from datetime import datetime

# import numpy as np
# from utils.utils import create_folder_if_not_exists


# # Load the audio file
# audio_path = "./vocals_only.wav"
# y, sr = librosa.load(audio_path, sr=None)

# # # Parameters for voice activity detection
# # Parameters for voice activity detection
# frame_length = 1780  # 2048
# hop_length = 126  # 512
# energy_threshold = 0.1

# time1 = datetime.now()


# # Calculate short-time energy using vectorized operations
# energy = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length] ** 2
# energy = np.sum(energy, axis=1)

# # Normalize energy
# energy /= np.max(energy)


# # Detect voice activity
# voice_activity = energy > energy_threshold

# # Find start and end times of voice activity using boolean indexing
# indices = np.flatnonzero(np.diff(voice_activity.astype(int)))
# timestamps = []

# # Pre-calculate hop_length / sr
# hop_duration = hop_length / sr


# for i in range(0, len(indices), 2):
#     start = indices[i] * hop_duration
#     end = indices[i + 1] * hop_duration if i + 1 < len(indices) else len(y) / sr
#     timestamps.append({"start": start, "end": end})

# # Handle case where audio ends with voice activity
# if voice_activity[-1]:
#     end = len(y) / sr
#     timestamps.append({"start": indices[-1] * hop_duration, "end": end})


# def format_timestamps(timestamp: dict):
#     return f"{int(timestamp['start'] / 60):02d}:{int(timestamp['start'] % 60):02d} - {int(timestamp['end'] / 60):02d}:{int(timestamp['end'] % 60):02d}"


# # Merges contiguous timestamps. E.g., 01:11-01:15 and 01:16-01:21 become 01:11-01:21
# def merge_contiguous_timestamps(timestamps):
#     merged_timestamps = []
#     current_start = int(timestamps[0]["start"])
#     current_end = int(timestamps[0]["end"])

#     for timestamp in timestamps[1:]:
#         if int(timestamp["start"]) in [current_end, current_end + 1]:
#             current_end = int(timestamp["end"])
#         else:
#             merged_timestamps.append({"start": current_start, "end": current_end})
#             current_start = int(timestamp["start"])
#             current_end = int(timestamp["end"])

#     merged_timestamps.append({"start": current_start, "end": current_end})

#     return merged_timestamps


# new_timestamps = merge_contiguous_timestamps(timestamps)


# timestamps_folder = "./timestamps/"
# create_folder_if_not_exists(timestamps_folder)
# timestamps_energy_path = "energy.txt"
# timestamps_concat_path = "energy_concat.txt"

# full_timestamps_energy_path = os.path.join(timestamps_folder, timestamps_energy_path)
# full_timestamps_concat_path = os.path.join(timestamps_folder, timestamps_concat_path)


# # Save merged timestamps
# with open(full_timestamps_concat_path, "w") as file:
#     for timestamp in new_timestamps:
#         file.write(f"{format_timestamps(timestamp)}\n")

# # Save original timestamps with energy
# with open(full_timestamps_energy_path, "w") as file:
#     for i, timestamp in enumerate(timestamps):
#         start_frame = int(timestamp["start"] / hop_duration)
#         end_frame = int(timestamp["end"] / hop_duration)
#         if start_frame < end_frame:  # Ensure the slice is not empty
#             avg_energy = np.mean(energy[start_frame:end_frame])
#         else:
#             avg_energy = 0  # Handle empty slice case
#         file.write(f"{format_timestamps(timestamp)} - Energy: {avg_energy:.4f}\n")

# time2 = datetime.now()

# # Calculate the difference in seconds
# time_difference = (time2 - time1).total_seconds()

# # Convert the difference to minutes and seconds
# minutes, seconds = divmod(abs(time_difference), 60)

# # Print the difference in mm:ss format
# print(f"Difference: {int(minutes):02d}:{int(seconds):02d}")


import os
from datetime import datetime
import librosa
import numpy as np
from utils.utils import create_folder_if_not_exists


def load_audio(
    audio_path: str, sample_rate: int | None = None
) -> tuple[np.ndarray, int]:
    """Load the audio file.

    Args:
        audio_path (str): Path to the audio file.
        sample_rate (int, optional): Sample rate. Defaults to None.

    Returns:
        tuple[np.ndarray, int]: Audio data and sample rate.
    """
    return librosa.load(audio_path, sr=sample_rate)


def calculate_energy(
    y: np.ndarray, frame_length: int, hop_length: int, chunk_size: int
) -> np.ndarray:
    """Calculate short-time energy using chunk processing.

    Args:
        y (np.ndarray): Audio data.
        frame_length (int): Frame length.
        hop_length (int): Hop length.
        chunk_size (int): Size of each chunk to process.

    Returns:
        np.ndarray: Short-time energy.
    """
    energy = []
    num_chunks = len(y) // chunk_size + (1 if len(y) % chunk_size != 0 else 0)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size + frame_length, len(y))
        chunk = y[start:end]

        # Calculate energy for the current chunk
        chunk_energy = (
            np.lib.stride_tricks.sliding_window_view(chunk, frame_length)[::hop_length]
            ** 2
        )
        chunk_energy = np.sum(chunk_energy, axis=1)
        energy.append(chunk_energy)

    # Concatenate all chunk energies
    energy = np.concatenate(energy)

    # Normalize energy
    return energy / np.max(energy)


def calculate_energy_in_chunks(
    y: np.ndarray,
    frame_length: int,
    hop_length: int,
    seconds_per_chunk: int,
    sample_rate: int,
) -> np.ndarray:
    """Calculate short-time energy using chunk processing with overlap.

    Args:
        y (np.ndarray): Audio data.
        frame_length (int): Frame length.
        hop_length (int): Hop length.
        seconds_per_chunk (int): Size of each chunk to process in seconds.
        sample_rate (int): Sample rate. Useful to compute chunk size.

    Returns:
        np.ndarray: Short-time energy.
    """
    energy = []

    chunk_size = int(seconds_per_chunk * sample_rate)

    print("Selected chunk size: ", chunk_size)
    num_chunks = len(y) // chunk_size + (1 if len(y) % chunk_size != 0 else 0)

    for i in range(num_chunks):
        start = max(i * chunk_size - frame_length, 0)  # Start with overlap
        end = min((i + 1) * chunk_size + frame_length, len(y))  # End with overlap
        chunk = y[start:end]

        # Calculate energy for the current chunk
        chunk_energy = (
            np.lib.stride_tricks.sliding_window_view(chunk, frame_length)[::hop_length]
            ** 2
        )
        chunk_energy = np.sum(chunk_energy, axis=1)
        energy.append(chunk_energy)

    # Concatenate all chunk energies
    energy = np.concatenate(energy)

    # Normalize energy
    return energy / np.max(energy)


def calculate_energy(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Calculate short-time energy using vectorized operations.

    Args:
        y (np.ndarray): Audio data.
        frame_length (int): Frame length.
        hop_length (int): Hop length.

    Returns:
        np.ndarray: Short-time energy.
    """
    print(
        "Ram Usage INSIDE calculate_energy 1: ",
        os.popen("free -m").readlines()[1].split()[2],
    )
    energy = (
        np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length] ** 2
    )

    print(
        "Ram Usage INSIDE calculate_energy 2: ",
        os.popen("free -m").readlines()[1].split()[2],
    )
    energy = np.sum(energy, axis=1)
    print(
        "Ram Usage INSIDE calculate_energy 3: ",
        os.popen("free -m").readlines()[1].split()[2],
    )
    return energy / np.max(energy)


def detect_voice_activity(energy: np.ndarray, energy_threshold: float) -> np.ndarray:
    """Detect voice activity based on energy threshold.

    Args:
        energy (np.ndarray): Short-time energy.
        energy_threshold (float): Energy threshold.

    Returns:
        np.ndarray: Voice activity.
    """
    return energy > energy_threshold


def extract_timestamps(
    voice_activity: np.ndarray, hop_duration: float, y_length: float, sr: int
) -> list[dict[str, float]]:
    """Extract start and end times of voice activity.

    Args:
        voice_activity (np.ndarray): Voice activity.
        hop_duration (float): Hop duration.
        y_length (float): Length of the audio.
        sr (int): Sample rate.

    Returns:
        list[dict[str, float]]: List of start and end times of voice activity.
    """
    indices = np.flatnonzero(np.diff(voice_activity.astype(int)))
    timestamps = []
    for i in range(0, len(indices), 2):
        start = indices[i] * hop_duration
        end = indices[i + 1] * hop_duration if i + 1 < len(indices) else y_length / sr
        timestamps.append({"start": start, "end": end})
    if voice_activity[-1]:
        end = y_length / sr
        timestamps.append({"start": indices[-1] * hop_duration, "end": end})
    return timestamps


def format_timestamps(timestamp: dict[str, float]) -> str:
    """Format timestamps into mm:ss format.

    Args:
        timestamp (dict[str, float]): Timestamp to format.

    Returns:
        str: Formatted timestamp.
    """
    return f"{int(timestamp['start'] / 60):02d}:{int(timestamp['start'] % 60):02d} - {int(timestamp['end'] / 60):02d}:{int(timestamp['end'] % 60):02d}"


def merge_contiguous_timestamps(
    timestamps: list[dict[str, float]]
) -> list[dict[str, float]]:
    """Merge contiguous timestamps. E.g., 01:11-01:15 and 01:16-01:21 become 01:11-01:21

    Args:
        timestamps (list[dict[str, float]]): List of timestamps.

    Returns:
        list[dict[str, float]]: Merged list of timestamps.
    """
    merged_timestamps = []
    current_start = int(timestamps[0]["start"])
    current_end = int(timestamps[0]["end"])
    for timestamp in timestamps[1:]:
        if int(timestamp["start"]) in [current_end, current_end + 1]:
            current_end = int(timestamp["end"])
        else:
            merged_timestamps.append({"start": current_start, "end": current_end})
            current_start = int(timestamp["start"])
            current_end = int(timestamp["end"])
    merged_timestamps.append({"start": current_start, "end": current_end})
    return merged_timestamps


def save_timestamps(
    timestamps: list[dict[str, float]],
    hop_duration: float,
    file_path: str,
    energy: np.ndarray = None,
) -> None:
    """Save timestamps to a file.

    Args:
        timestamps (list[dict[str, float]]): List of timestamps.
        hop_duration (float): Hop duration.
        file_path (str): Path to the output file.
        energy (np.ndarray, optional): Short-time energy. Defaults to None.

    Returns:
        None
    """
    with open(file_path, "w") as file:
        for i, timestamp in enumerate(timestamps):
            if energy is not None:
                start_frame = int(timestamp["start"] / hop_duration)
                end_frame = int(timestamp["end"] / hop_duration)
                avg_energy = (
                    np.mean(energy[start_frame:end_frame])
                    if start_frame < end_frame
                    else 0
                )
                file.write(
                    f"{format_timestamps(timestamp)} - Energy: {avg_energy:.4f}\n"
                )
            else:
                file.write(f"{format_timestamps(timestamp)}\n")


# def get_available_ram():
#     """Get available RAM in GB."""
#     mem_info = os.popen("free -b").readlines()[1].split()
#     available_ram = int(mem_info[6])  # Available RAM in bytes
#     return available_ram / (1_073_741_824)  # = 1024 * 1024 * 1024


# def find_max_seconds_per_chunk(
#     max_free_memory_ratio: float = 0.8,
#     seconds_per_chunk: int = 60,
#     chunk_overlap_ratio: float = 0.75,
# ):
#     """Find the maximum seconds per chunk based on available RAM.

#     Args:
#         max_free_memory_ratio (float, optional): Maximum free memory ratio. Defaults to 0.8.
#         seconds_per_chunk (int, optional): Initial seconds per chunk. Defaults to 60 seconds. ~0.25gb used per 60 seconds.
#         chunk_overlap_ratio (float, optional): Chunk's ratio used for left and right overlap (i.e., total memory
#             will be chunk_size + (chunk_size * chunk_overlap_ratio * 2)). Defaults to 0.75.

#     Returns:
#         int: Maximum seconds per chunk.
#     """
#     # ~0.25gb used per 60 seconds.
#     used_gb = seconds_per_chunk / 60 * 0.25
#     available_ram = get_available_ram() * max_free_memory_ratio
#     total_chunk_overlap_ratio = chunk_overlap_ratio * 2 + 1
#     while True:
#         if (used_gb * 2) * total_chunk_overlap_ratio < available_ram:
#             seconds_per_chunk *= 2
#             used_gb *= 2
#         elif (used_gb * 1.5) * total_chunk_overlap_ratio < available_ram:
#             seconds_per_chunk *= 1.5
#             used_gb *= 1.5
#         elif (used_gb * 1.25) * total_chunk_overlap_ratio < available_ram:
#             seconds_per_chunk *= 1.25
#             used_gb *= 1.25
#         elif (used_gb * 1.1) * total_chunk_overlap_ratio < available_ram:
#             seconds_per_chunk *= 1.1
#             used_gb *= 1.1
#         else:
#             return seconds_per_chunk


def main():
    audio_path = "./vocals_only.wav"
    frame_length = 1780  # Original : 2048
    hop_length = 126  # Original : 512
    energy_threshold = 0.1

    print(f"Audio path: {audio_path}")

    y: np.ndarray
    sr: int
    y, sr = load_audio(audio_path)
    y_len = len(y)
    hop_duration = hop_length / sr

    time1 = datetime.now()

    audio_length_in_minutes = y_len / (sr * 60)
    print(f"Audio length: {audio_length_in_minutes:.2f} minutes")

    # max_seconds_per_chunk = find_max_seconds_per_chunk()

    # energy = calculate_energy(y=y, frame_length=frame_length, hop_length=hop_length)
    energy = calculate_energy_in_chunks(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        sample_rate=sr,
        seconds_per_chunk=2000,  # 1024 = ~17 minutes, 1562 = ~21 minutes, 2000 = ~30 minutes
    )

    voice_activity = detect_voice_activity(
        energy=energy, energy_threshold=energy_threshold
    )

    timestamps = extract_timestamps(
        voice_activity=voice_activity, hop_duration=hop_duration, y_length=y_len, sr=sr
    )
    new_timestamps = merge_contiguous_timestamps(timestamps=timestamps)

    timestamps_folder = "./timestamps/"
    create_folder_if_not_exists(folder_path=timestamps_folder)
    save_timestamps(
        timestamps=new_timestamps,
        file_path=os.path.join(timestamps_folder, "energy_concat.txt"),
        hop_duration=hop_duration,
        energy=None,
    )
    save_timestamps(
        timestamps=timestamps,
        file_path=os.path.join(timestamps_folder, "energy.txt"),
        hop_duration=hop_duration,
        energy=energy,
    )

    time2 = datetime.now()
    time_difference = (time2 - time1).total_seconds()
    minutes, seconds = divmod(abs(time_difference), 60)
    print(f"Difference: {int(minutes):02d}:{int(seconds):02d}")


if __name__ == "__main__":
    main()
