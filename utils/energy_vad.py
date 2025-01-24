import json
import os
from datetime import datetime
from typing import cast
import librosa
import numpy as np
from utils.utils import create_folder_if_not_exists, build_path

import logging

logger = logging.getLogger("auto-sub-gen")


def load_audio(
    audio_path: str, sample_rate: int | None = None
) -> tuple[np.ndarray, int | float]:
    """Load the audio file.

    Args:
        audio_path (str): Path to the audio file.
        sample_rate (int, optional): Sample rate. Defaults to None.

    Returns:
        tuple[np.ndarray, int]: Audio data and sample rate.
    """
    return librosa.load(audio_path, sr=sample_rate)


def load_vad_timestamps(vad_json_path: str) -> list[tuple[float, float]]:
    """Load VAD timestamps from a JSON file.

    Args:
        vad_json_path (str): Path to the VAD JSON file.

    Returns:
        list[tuple[float, float]]: List of tuples (start, end) for each segment.
    """
    with open(vad_json_path, "r") as file:
        vad_data = json.load(file)
    return [(segment["start"], segment["end"]) for segment in vad_data]


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


# More accurate, but takes to much RAM with long videos. E.g., ~13 gb for a 90 minutes video with 44.1kHz
# def calculate_energy(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
#     """Calculate short-time energy using vectorized operations.

#     Args:
#         y (np.ndarray): Audio data.
#         frame_length (int): Frame length.
#         hop_length (int): Hop length.

#     Returns:
#         np.ndarray: Short-time energy.
#     """
#     energy = (
#         np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length] ** 2
#     )
#     energy = np.sum(energy, axis=1)
#     return energy / np.max(energy)


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
    energy: np.ndarray | None = None,
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


def save_timestamps_as_json(
    timestamps: list[dict[str, float]],
    file_path: str,
    energy: np.ndarray | None = None,
    hop_duration: float | None = None,
) -> None:
    if energy is not None and hop_duration is not None:
        with open(file_path, "w") as file:
            for i, timestamp in enumerate(timestamps):
                start_frame = int(timestamp["start"] / hop_duration)
                end_frame = int(timestamp["end"] / hop_duration)
                avg_energy = (
                    np.mean(energy[start_frame:end_frame])
                    if start_frame < end_frame
                    else 0
                )
                timestamp["energy"] = float(avg_energy)
                file.write(json.dumps(timestamp) + "\n")
    else:
        with open(file_path, "w") as file:
            json.dump(timestamps, file, indent=4)


def perform_energy_vad(
    audio_path: str,
    frame_length: int = 1780,
    hop_length: int = 126,
    energy_threshold: float = 0.1,
    seconds_per_chunk: int = 2000,  # ~30 minutes
    timestamps_folder: str = "./timestamps/",
):
    # audio_path = "./vocals_only.wav"

    logger.debug(f"utils.energy_vad.perform_energy_vad:: Audio path: {audio_path}")

    y: np.ndarray
    sr: int
    y, sr = cast(tuple[np.ndarray, int], load_audio(audio_path))
    y_len = len(y)
    hop_duration = hop_length / sr

    audio_length_in_minutes = y_len / (sr * 60)
    logger.debug(
        f"utils.energy_vad.perform_energy_vad:: Audio length: {audio_length_in_minutes:.2f} minutes"
    )

    energy = calculate_energy_in_chunks(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        sample_rate=sr,
        seconds_per_chunk=seconds_per_chunk,
    )

    voice_activity = detect_voice_activity(
        energy=energy, energy_threshold=energy_threshold
    )

    timestamps = extract_timestamps(
        voice_activity=voice_activity, hop_duration=hop_duration, y_length=y_len, sr=sr
    )
    new_timestamps = merge_contiguous_timestamps(timestamps=timestamps)

    create_folder_if_not_exists(folder_path=timestamps_folder)

    timestap_full_path = build_path(
        folder_path=timestamps_folder,
        file_path=audio_path,
        extension_replacement="_timestamps.json",
    )

    save_timestamps_as_json(
        timestamps=new_timestamps,
        file_path=timestap_full_path,
    )
