# ./utils/srt_generation.py

import logging
import os
from faster_whisper.transcribe import Segment, Word
from utils.utils import format_time

logger = logging.getLogger("auto-sub-gen")


def adjust_outlier_times(words: list[Word]) -> list[Word]:
    if len(words) > 1:
        if words[1].start - words[0].end > 3:
            words[0].end = words[1].start - 0.5
            words[0].start = words[1].end - 0.4

    return words


def generate_srt(
    segments: list[Segment],
    audio_path: str,
    max_chars: int,
    srt_debug_mode: bool,
    srt_debug_path: str = "./raw_result.txt",
) -> None:
    """
    Generate SRT file from segments

    Args:
        segments (list[Segment]): List of segments.
        audio_path (str): Path to the audio file.
        max_chars (int): Maximum number of characters per line.
        srt_debug_mode (bool): Debug mode.
        srt_debug_path (str, optional): Debug file path. Defaults to "./raw_result.txt".

    Returns:
        None
    """
    srt_filename = os.path.splitext(audio_path)[0] + ".srt"
    with open(srt_filename, "w") as srt_file:
        index = 1
        words: list[Word]
        word: Word

        if srt_debug_mode:
            with open(srt_debug_path, "a") as raw_file:
                raw_file.write(
                    "--------------------------------------------------------\n"
                )

        for segment in segments:
            words = segment.words
            current_text = ""
            start_time = None

            if srt_debug_mode:
                with open(srt_debug_path, "a") as raw_file:
                    raw_file.write(f"RAW:\n")
                    for word in words:
                        raw_file.write(
                            f"Word: {word.word}, Start: {word.start}, End: {word.end}, Prob: {word.probability}\n"
                        )

            words = adjust_outlier_times(words)

            if srt_debug_mode:
                with open(srt_debug_path, "a") as raw_file:
                    raw_file.write(f"NORMALIZED:\n")
                    for word in words:
                        raw_file.write(
                            f"Word: {word.word}, Start: {word.start}, End: {word.end}\n"
                        )
                    raw_file.write("\n")

            for word in words:
                if start_time is None:
                    start_time = word.start

                if len(current_text) + len(word.word) + 1 > max_chars:
                    end_time = word.start
                    srt_file.write(
                        f"{index}\n{format_time(start_time)} --> {format_time(end_time)}\n{current_text.strip()}\n\n"
                    )
                    index += 1
                    current_text = ""
                    start_time = word.start

                current_text += " " + word.word

            if current_text:
                end_time = words[-1].end
                srt_file.write(
                    f"{index}\n{format_time(start_time)} --> {format_time(end_time)}\n{current_text.strip()}\n\n"
                )
                index += 1
