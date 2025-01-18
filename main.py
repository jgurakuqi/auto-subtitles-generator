# main.py

import os

import torch
from utils.audio_extraction import extract_audio
from utils.transcription import load_model, transcribe_audio
from utils.srt_generation import generate_srt
from utils.utils import recursively_read_video_paths, sort_key
from faster_whisper.transcribe import Segment, TranscriptionInfo


def main():
    # FOLDER_VIDEOS_PATH = r"/mnt/c/Users/jgura/Downloads/The Ocean Cut/"
    FOLDER_VIDEOS_PATH = (
        r"/mnt/c/Users/jgura/Downloads/NaruCannon/Dub/3_Leaf Destruction/"
    )
    USE_BATCHED_INFERENCE = False
    MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
    # dinamically check if GPU is available, otherwise use CPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16"  # "int8_float16"
    NUM_WORKERS = 1
    BEAM_SIZE = 8
    LANGUAGE = "en"
    USE_WORD_TIMESTAMPS = True
    BATCH_SIZE = 22
    MAX_CHARS = 60
    FORCE_EXTRACT = True
    USE_VAD_FILTER = True
    # VAD_SETTINGS = {
    #     "threshold": 0.15,
    #     "min_silence_duration_ms": 2500,
    #     "speech_pad_ms": 500,
    # }
    VAD_SETTINGS = {
        "threshold": 0.11,
        "min_silence_duration_ms": 2500,
        "speech_pad_ms": 500,
    }
    PATIENCE = 1.25  # default 1
    FFMPEG_THREADS_NUM = 6

    video_paths = recursively_read_video_paths(FOLDER_VIDEOS_PATH)

    # Test
    video_paths = [
        video_path
        for video_path in video_paths
        # if "Genin Takedown".lower() in video_path.lower()
    ]

    # video_paths.sort(key=sort_key, reverse=True)

    audios_to_extract_and_video_paths = []
    if not FORCE_EXTRACT:
        # Extract audio from only missing videos
        for video_path in video_paths:
            _, file_extension = os.path.splitext(video_path)
            tmp_audio_path = video_path.replace(file_extension, ".wav")

            if not os.path.exists(tmp_audio_path):
                audios_to_extract_and_video_paths.append((tmp_audio_path, video_path))
    else:
        # Extract audio from all videos
        for video_path in video_paths:
            _, file_extension = os.path.splitext(video_path)
            tmp_audio_path = video_path.replace(file_extension, ".wav")

            audios_to_extract_and_video_paths.append((tmp_audio_path, video_path))

    extract_audio(
        audios_to_extract_and_video_paths=audios_to_extract_and_video_paths,
        force_extract=FORCE_EXTRACT,
        num_workers=FFMPEG_THREADS_NUM,
    )

    # return
    model = load_model(
        USE_BATCHED_INFERENCE, MODEL_ID, DEVICE, COMPUTE_TYPE, NUM_WORKERS
    )

    segments: list[Segment]
    info: TranscriptionInfo
    for video_path in video_paths:
        _, file_extension = os.path.splitext(video_path)
        input_audio_path = video_path.replace(file_extension, ".wav")

        segments, info = transcribe_audio(
            model=model,
            input_audio_path=input_audio_path,
            beam_size=BEAM_SIZE,
            language=LANGUAGE,
            use_vad_filter=USE_VAD_FILTER,
            batch_size=BATCH_SIZE,
            use_word_timestamps=USE_WORD_TIMESTAMPS,
            vad_settings=VAD_SETTINGS,
            use_batched_inference=USE_BATCHED_INFERENCE,
            patience=PATIENCE,
        )

        generate_srt(segments=segments, video_path=video_path, max_chars=MAX_CHARS)


if __name__ == "__main__":
    main()
