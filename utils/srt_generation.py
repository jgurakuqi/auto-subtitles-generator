import os
from faster_whisper.transcribe import Segment, Word
from utils.utils import format_time


def adjust_outlier_times(words: list[Word]) -> list[Word]:
    # Calculate average time difference between consecutive words
    # time_diffs = [words[i + 1].start - words[i].end for i in range(len(words) - 1)]
    # avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0
    # avg_time_diff += 0.0001  # add 0.0001 to avoid zero multiplication

    # avg_single_word_duration = 0
    # for word in words:
    #     avg_single_word_duration += word.end - word.start
    # avg_single_word_duration /= len(words)

    # avg_time_diff_without_outliers = 0
    # # now compute the average time difference between consecutive words without outliers
    # for i in range(len(words) - 1):
    #     if (words[i + 1].start - words[i].end) < 2 * avg_time_diff:
    #         avg_time_diff_without_outliers += words[i + 1].start - words[i].end
    # avg_time_diff_without_outliers /= len(words)
    # avg_time_diff_without_outliers += 0.0001  # add 0.0001 to avoid zero multiplication

    # # Adjust outlier words
    # outliers_detected = False
    # adjusted_words = []
    # for i, word in enumerate(words):
    #     if (
    #         i + 1 < len(words)
    #         and (words[i + 1].start - word.end) > 2 * avg_time_diff_without_outliers
    #     ):
    #         # If the time difference is more than twice the average, adjust the start and end times
    #         word.end = words[i + 1].start - avg_time_diff_without_outliers
    #         word.start = words[i + 1].end - avg_single_word_duration
    #         outliers_detected = True
    #     adjusted_words.append(word)

    # if outliers_detected:
    #     return adjust_outlier_times(adjusted_words)
    # else:
    #     return adjusted_words

    # if len(words) < 2:
    #     return words

    # average_time_diff = 0
    # for i in range(len(words) - 1):
    #     average_time_diff += words[i + 1].start - words[i].end
    # average_time_diff /= len(words)
    # average_time_diff += 0.0001  # add 0.0001 to avoid zero multiplication

    # smallest_time_diff_in_words = 100
    # for i in range(len(words) - 1):
    #     smallest_time_diff_in_words = (
    #         words[i + 1].start - words[i].end
    #         if words[i + 1].start - words[i].end < smallest_time_diff_in_words
    #         else smallest_time_diff_in_words
    #     )

    # found_outliers = False
    # for i in range(len(words) - 1):
    #     curr_diff_tollerance = (
    #         average_time_diff * 1.5 if "..." not in words[i].word else 2.0
    #     )
    #     if words[i + 1].start - words[i].end > curr_diff_tollerance:
    #         words[i].end = words[i + 1].start - smallest_time_diff_in_words
    #         words[i].start = words[i + 1].end - smallest_time_diff_in_words
    #         found_outliers = True

    # if found_outliers:
    #     return adjust_outlier_times(words)
    # else:
    #     return words
    if len(words) > 1:
        if words[1].start - words[0].end > 3:
            words[0].end = words[1].start - 0.5
            words[0].start = words[1].end - 0.4

    return words


def generate_srt(segments: list[Segment], video_path: str, max_chars: int) -> None:
    """
    Generate SRT file from segments

    Args:
        segments (list[Segment]): List of segments
        video_path (str): Path to the video file
        max_chars (int): Maximum number of characters per line

    Returns:
        None
    """
    srt_filename = os.path.splitext(video_path)[0] + ".srt"
    with open(srt_filename, "w") as srt_file:
        index = 1
        words: list[Word]
        word: Word

        with open("raw_result.txt", "a") as raw_file:
            raw_file.write("--------------------------------------------------------\n")

        for segment in segments:
            words = segment.words
            current_text = ""
            start_time = None

            with open("raw_result.txt", "a") as raw_file:
                for word in words:
                    raw_file.write(
                        f"Word: {word.word}, Start: {word.start}, End: {word.end}, Prob: {word.probability}\n"
                    )

            words = adjust_outlier_times(words)

            with open("raw_result.txt", "a") as raw_file:
                for word in words:
                    raw_file.write(f"NORMALIZED Start: {word.start}, End: {word.end}\n")
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
