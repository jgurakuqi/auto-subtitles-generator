# main.py

# Std library imports
import logging, torch, os

# Local repository imports
from utils.audio_extraction import extract_audio
from utils.transcription import load_model, transcribe_audio
from utils.srt_generation import generate_srt
from utils.utils import recursively_read_video_paths, initialize_logging

# Third-party library imports
from faster_whisper.transcribe import Segment, TranscriptionInfo

initialize_logging(
    logs_folder_path="./logs", log_level=logging.DEBUG, log_to_console=True
)
logger = logging.getLogger("auto-sub-gen")


def main():
    # FOLDER_VIDEOS_PATH = r"/mnt/c/Users/jgura/Downloads/The Ocean Cut/"
    # FOLDER_VIDEOS_PATH = (
    #     r"/mnt/c/Users/jgura/Downloads/NaruCannon/Dub/3_Leaf Destruction/"
    # )
    FOLDER_VIDEOS_PATH = r"/mnt/c/Users/jgura/Downloads/NaruCannon/Dub/2_Chunin Exams/"

    # Model parameters
    USE_BATCHED_INFERENCE = False
    MODEL_ID = "deepdml/faster-whisper-large-v3-turbo-ct2"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = "float16"  # "int8_float16"
    NUM_WORKERS = 1
    BEAM_SIZE = 8
    PATIENCE = 1.25
    LANGUAGE = "en"
    LOG_PROGRESS = True  # print progress bar during transcription
    USE_WORD_TIMESTAMPS = True
    BATCH_SIZE = 22

    # VAD settings:
    USE_VAD_FILTER = True
    VAD_SETTINGS = {
        "threshold": 0.2,
        "neg_threshold": 0.15,
        "min_silence_duration_ms": 1000,
        "speech_pad_ms": 100,
    }

    # Transcription parameters
    MAX_CHARS = 60
    ATOMIC_TRANSCRIPTION = False

    # SRT generation parameters
    SRT_DEBUG_MODE = True

    # Audio extraction parameters
    FORCE_EXTRACT = False
    ATOMIC_AUDIO_EXTRACTION = True
    AUDIO_FOLDER = "./audio_output/"
    FFMPEG_THREADS_NUM = 6

    orig_video_paths = recursively_read_video_paths(FOLDER_VIDEOS_PATH)

    logger.debug(f"Found video_paths:\n")
    for video_path in orig_video_paths:
        logger.debug(f"{video_path}\n")

    # Keeping videos that don't have an extracted audio existing or all the audios if extraction is FORCED
    video_paths_to_extract = [
        video_path
        for video_path in orig_video_paths
        if FORCE_EXTRACT
        or not os.path.exists(
            os.path.join(
                AUDIO_FOLDER,
                os.path.basename(video_path).replace(
                    os.path.splitext(video_path)[-1], ".wav"
                ),
            )
        )
    ]

    _ = extract_audio(
        audio_extraction_paths=video_paths_to_extract,
        force_extract=FORCE_EXTRACT,
        num_workers=FFMPEG_THREADS_NUM,
        atomic_operation=ATOMIC_AUDIO_EXTRACTION,
        audio_output_folder=AUDIO_FOLDER,
    )

    logger.info(f"Loading model...")

    # return
    model = load_model(
        USE_BATCHED_INFERENCE, MODEL_ID, DEVICE, COMPUTE_TYPE, NUM_WORKERS
    )

    logger.info(f"Model loaded.")

    all_audio_paths = [
        os.path.join(
            AUDIO_FOLDER,
            os.path.basename(video_path).replace(
                os.path.splitext(video_path)[-1], ".wav"
            ),
        )
        for video_path in orig_video_paths
    ]

    logger.info(f"Starting transcription of subtitles...")
    for audio_path in all_audio_paths:
        logger.debug(f"Fetchable audio: {audio_path}...")

    segments: list[Segment]
    info: TranscriptionInfo
    for audio_path in all_audio_paths:
        try:
            logger.info(f"Transcribing {audio_path}...")
            segments, info = transcribe_audio(
                model=model,
                input_audio_path=audio_path,
                beam_size=BEAM_SIZE,
                language=LANGUAGE,
                use_vad_filter=USE_VAD_FILTER,
                batch_size=BATCH_SIZE,
                use_word_timestamps=USE_WORD_TIMESTAMPS,
                vad_settings=VAD_SETTINGS,
                use_batched_inference=USE_BATCHED_INFERENCE,
                patience=PATIENCE,
                log_progress=LOG_PROGRESS,
            )

            logger.info(f"Generating SRT for {audio_path}...")
            generate_srt(
                segments=segments,
                audio_path=audio_path,
                max_chars=MAX_CHARS,
                srt_debug_mode=SRT_DEBUG_MODE,
            )

        except Exception as e:
            logger.error(f"main::EXCEPTION::Error while transcribing {audio_path}: {e}")
            if ATOMIC_TRANSCRIPTION:
                return

    logger.info(f"Done.")


if __name__ == "__main__":
    main()
