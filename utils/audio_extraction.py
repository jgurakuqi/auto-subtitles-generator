# # utils/audio_extraction.py

import logging, os, subprocess
import concurrent.futures as concurrent_futures
from tqdm import tqdm

from utils.utils import create_folder_if_not_exists

logger = logging.getLogger("auto-sub-gen")


def extract_audio(
    audio_extraction_paths: list[str],
    force_extract: bool,
    num_workers: int,
    atomic_operation: bool = False,
    audio_output_folder: str = "./audio_output/",
) -> None:
    """
    Extracts audio from video files using ffmpeg.

    Args:
        audio_extraction_paths (list[str]): List of the video paths from which the audio will be extracted.
        force_extract (bool): Whether to force the extraction of audio even if it already exists.
        num_workers (int): Number of threads to use for parallel processing.
        atomic_operation (bool, optional): Whether the process should be completed even if there are errors. Defaults to False.
        audio_output_folder (str, optional): Output audio folder path. Defaults to "./audio_output/".

    Returns:
        None
    """
    if len(audio_extraction_paths) > 0 or force_extract:
        logger.info("Starting audio extraction from video files...")
        for video_path in audio_extraction_paths:
            logger.debug(f"From: {video_path}")

        logger.info("Creating audio output folder...")
        create_folder_if_not_exists(audio_output_folder)

        failed_paths = []
        failures = 0

        with concurrent_futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    run_subprocess,
                    video_path,
                    atomic_operation,
                    audio_output_folder,
                )
                for video_path in audio_extraction_paths
            ]

            for future in tqdm(
                concurrent_futures.as_completed(futures),
                total=len(futures),
                desc="Extracting audio:...",
            ):
                try:
                    future.result()
                except Exception as e:
                    failures += 1
                    failed_paths.append(str(e))
                    if atomic_operation:
                        logger.error("Atomic operation failed. Terminating process.")
                        break

            executor.shutdown()

        if not atomic_operation and failures > 0:
            logger.error(
                f"Audio extraction failed for {failures} files: {failed_paths}"
            )

        logger.info("Audio extraction completed.")
    else:
        logger.info("No files to extract audio from.")


def run_subprocess(
    video_path: str,
    atomic_operation: bool,
    audio_output_folder: str,
) -> None:
    """
    Runs the ffmpeg command to extract audio from a video file.

    Args:
        video_path (str): Input video file path.
        atomic_operation (bool): Whether the process should be completed even if there are errors.
        audio_output_folder (str): Output audio folder path.

    Raises:
        RuntimeError: Thrown if the ffmpeg command fails.

    Returns:
        None
    """
    try:
        # Paths is = audio_output_folder + original video name + .wav
        output_path = os.path.join(
            audio_output_folder,
            os.path.basename(video_path).replace(
                os.path.splitext(video_path)[-1], ".wav"
            ),
        )
        subprocess.run(
            args=[
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-vn",
                "-af",
                (
                    "equalizer=f=60:t=q:w=1:g=-10:f=310:t=q:w=1:g=5:f=1000:t=q:w=1:g=5:f=3000:t=q:w=1:g=5:f=6000:t=q:w=1:g=-10,"
                    "highpass=f=100,"  # 200
                    "lowpass=f=4000,"  # 3000
                    "volume=3.5"
                ),
                "-acodec",
                "pcm_s16le",
                "-ar",
                "16000",
                "-ac",
                "1",
                output_path,
            ],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        error_message = f"Failed to extract audio from {video_path}: {e}"
        logger.error(error_message)
        if atomic_operation:
            raise RuntimeError(error_message)
