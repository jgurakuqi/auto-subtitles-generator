from datetime import datetime
import json
import os
from pydub import AudioSegment
import numpy as np
from utils.utils import create_folder_if_not_exists


def calculate_decibel_levels_with_timestamps(
    audio_file_path: str,
    segment_length_ms: int = 50,  # 100 is good
    decibels_threshold: float = -1,
) -> list[dict[str, float]]:
    """Calculate decibel levels and timestamps for the audio file. The steps are:
    1. Load the audio file.
    2. Split the audio into segments, and for each one:
        2.1. Calculate the RMS (root mean square).
        2.2. Convert the RMS to decibel levels.
        2.3. Calculate the timestamps.

    Args:
        audio_file_path (str): Path to the audio file.
        segment_length_ms (int, optional): Length of each segment in milliseconds. Defaults to 150.
        decibels_threshold (float, optional): Decibel level threshold .Defaults to -1 to disable filter.
    Returns:
        list[dict[str, float]]: List of decibel levels and timestamps.
    """
    audio = AudioSegment.from_file(audio_file_path)

    num_segments = len(audio) // segment_length_ms

    decibels_by_timestamps = []
    neg_inf = -1.0

    computation_start = datetime.now()

    for i in range(num_segments):

        rms = audio[i * segment_length_ms : (i + 1) * segment_length_ms].rms

        decibel = 20 * np.log10(rms) if rms > 0 else neg_inf

        if decibel > decibels_threshold:
            decibels_by_timestamps.append(
                {
                    "timestamp": (i * segment_length_ms) / 1000.0,
                    "decibel": decibel,
                }
            )
        else:
            decibels_by_timestamps.append(
                {
                    "timestamp": (i * segment_length_ms) / 1000.0,
                    "decibel": decibel,
                    "is_silence": "True",
                }
            )

    print(
        "Elapsed time: ",
        (datetime.now() - computation_start).total_seconds(),
    )

    return decibels_by_timestamps


def format_as_hh_mm_ss(timestamp_in_seconds: float) -> str:
    """Format seconds as HH:MM:SS

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time
    """
    hours: int = int(timestamp_in_seconds // 3600)
    minutes: int = int((timestamp_in_seconds % 3600) // 60)
    seconds: float = timestamp_in_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:.3f}"


def filter_timestamps_and_decibels_by_threshold(
    decibels_by_timestamps: list[dict[str, float]], decibel_threshold: float
) -> list[dict[str, float]]:
    return [
        entry
        for entry in decibels_by_timestamps
        if entry["decibel"] > decibel_threshold
    ]


def write_decibel_and_timestamps_to_file(
    decibels_by_timestamps: list[dict[str, float]],
    file_path: str,
    decibel_rounding: int = 3,
) -> None:
    """Write decibel levels and timestamps to a file

    Args:
        timestamps (list[float]): List of timestamps.
        decibel_levels (list[float]): List of decibel levels.
        file_path (str): Path to the output file.

    Returns:
        None
    """
    with open(file_path, "w") as f:
        for line in decibels_by_timestamps:
            entry = {
                "timestamp": format_as_hh_mm_ss(line["timestamp"]),
                "decibel": round(line["decibel"], decibel_rounding),
            }
            if line.get("is_silence", None):
                entry["is_silence"] = line["is_silence"]

            f.write(json.dumps(entry) + "\n")


def discard_short_silences(
    decibels_by_timestamps: list[dict[str, float]],
    min_silence_duration_in_seconds: float,
) -> None:
    end_silence = None
    # Inverse range:
    for i in range(len(decibels_by_timestamps) - 1, -1, -1):
        if decibels_by_timestamps[i].get("is_silence", None):
            # new silence met
            if end_silence == None:
                # This is the first silence
                end_silence = i
        else:
            if end_silence != None:
                # This is the first non-silence after silence
                timestamp_of_end = decibels_by_timestamps[end_silence]["timestamp"]
                timestamp_of_start = decibels_by_timestamps[i + 1]["timestamp"]

                if (
                    timestamp_of_end - timestamp_of_start
                    < min_silence_duration_in_seconds
                ):
                    del decibels_by_timestamps[i + 1 : end_silence + 1]

                end_silence = None
            else:
                # Before of this there was no silence
                continue

    if end_silence:
        del decibels_by_timestamps[0 : end_silence + 1]


def find_intervals_without_silence(
    decibels_by_timestamps: list[dict[str, float]]
) -> list[dict[str, str]]:
    intervals = []
    start = None

    for i, entry in enumerate(decibels_by_timestamps):
        if "is_silence" not in entry:
            if start is None:
                start = entry["timestamp"]
        else:
            if start is not None:
                intervals.append({"start": start, "end": entry["timestamp"]})
                start = None

    # Check if the last segment was non-silence and add it
    if start is not None:
        intervals.append(
            {"start": start, "end": decibels_by_timestamps[-1]["timestamp"]}
        )

    return intervals


def write_timestamp_intervals_to_file(
    intervals: list[dict[str, str]], file_path: str
) -> None:
    with open(file_path, "w") as f:
        json.dump(intervals, f, indent=4)


def perform_decibel_vad(
    audio_paths: list[str],
    store_debug_files: bool = False,
    decibels_threshold: float = 47.5,
    segment_length_ms: int = 25,
    timestamps_folder: str = "./timestamps/",
):

    for audio_file_path in audio_paths:

        create_folder_if_not_exists(timestamps_folder)

        decibels_by_timestamps = calculate_decibel_levels_with_timestamps(
            audio_file_path=audio_file_path,
            decibels_threshold=decibels_threshold,
            segment_length_ms=segment_length_ms,
        )

        if store_debug_files:
            debug_pre_discard = os.path.join(
                timestamps_folder,
                audio_file_path.replace(".wav", "_debug_pre_discard.txt"),
            )
            write_decibel_and_timestamps_to_file(
                decibels_by_timestamps=decibels_by_timestamps,
                file_path=debug_pre_discard,
            )

        discard_short_silences(
            decibels_by_timestamps=decibels_by_timestamps,
            min_silence_duration_in_seconds=0.7,
        )

        if store_debug_files:
            debug_post_discard = os.path.join(
                timestamps_folder,
                audio_file_path.replace(".wav", "_debug_post_discard.txt"),
            )
            write_decibel_and_timestamps_to_file(
                decibels_by_timestamps=decibels_by_timestamps,
                file_path=debug_post_discard,
            )

        output_timestamps_path = os.path.join(
            timestamps_folder,
            audio_file_path.replace(".wav", "_timestamps.json"),
        )
        intervals = find_intervals_without_silence(decibels_by_timestamps)
        write_timestamp_intervals_to_file(intervals, output_timestamps_path)
