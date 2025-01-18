import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
import concurrent
from tqdm import tqdm


def extract_audio(
    audios_to_extract_and_video_paths: list[tuple[str, str]], force_extract: bool
) -> None:
    """
    Extracts audio from video files using ffmpeg.

    Args:
        audios_to_extract_and_video_paths (list[tuple[str, str]]): List of tuples containing audio and video paths.

    Returns:
        None
    """
    if len(audios_to_extract_and_video_paths) > 0 or force_extract:
        print("\nAudios to extract:")
        for audio_path, video_path in audios_to_extract_and_video_paths:
            print(audio_path, "\n from video: ", video_path, "\n")

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(
                    subprocess.run,
                    args=[
                        "ffmpeg",
                        "-y",
                        "-i",
                        video_path,
                        "-vn",
                        "-af",
                        (
                            "equalizer=f=60:t=q:w=1:g=-10:f=310:t=q:w=1:g=5:f=1000:t=q:w=1:g=5:f=3000:t=q:w=1:g=5:f=6000:t=q:w=1:g=-10,"
                            "highpass=f=200,"
                            "lowpass=f=3000,"
                            "volume=3.5"
                        ),
                        "-acodec",
                        "pcm_s16le",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        audio_path,
                    ],
                    check=True,
                )
                for audio_path, video_path in audios_to_extract_and_video_paths
            ]

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Extracting audio:...",
            ):
                future.result()

            executor.shutdown()
            print("All audios' extractions completed.")
    else:
        print("All audios are already extracted.")
