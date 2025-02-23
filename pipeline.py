from utils.transcription import transcribe_audios
from utils.subtitles import generate_aligned_subtitles
from utils.utils import (
    read_all_paths_recursively,
    create_folder_if_not_exists,
)
from configs.whisper_config import WhisperConfig
from configs.aligner_config import AlignerConfing
from utils.utils import delete_folder_if_exists
from utils.audio import extract_vocals_only_audio


def pipeline(
    input_folder_path: str,
    whisper_config: WhisperConfig,
    aligner_config: AlignerConfing,
    enable_reverse_sorting: bool = True,
    files_to_process: str = [],
    enable_debug_prints: bool = True,
    tmp_numpy_audio_folder: str = "./tmp_numpy_audio/",
    tmp_intermediate_result_folder: str = "./tmp_intermediate_result/",
):
    """
    Generate subtitles for all the videos in the provided folder.

    Args:
        input_folder_path (str): The path to the folder containing the videos.
        whisper_config (WhisperConfig): The configuration for the Whisper model.
        aligner_config (AlignerConfing): The configuration for the alignment model.
        enable_reverse_sorting (bool, optional): Enable reverse sorting of the audio paths. Defaults to True.
        files_to_process (str, optional): The files to process. Defaults to [].
        enable_debug_prints (bool, optional): Enable debug prints. Defaults to True.
        tmp_numpy_audio_folder (str, optional): The temporary folder for numpy audio. Here will be stored
            the audios extracted as numpy arrays, to be used also for the alignment. Defaults to "./tmp_numpy_audio/".
        tmp_intermediate_result_folder (str, optional): The temporary folder for intermediate results. Here will be
            stored the segments produced by the Whisper model. They are needed for the alignment. Defaults to "./tmp_intermediate_result/".
    """

    if enable_debug_prints:
        print("main.main:: input_folder_path: ", input_folder_path)
    sources_paths = read_all_paths_recursively(input_folder_path)

    supported_video_extensions = ["mp4", "avi", "mov", "wmv", "mkv"]

    # Filter the source paths to keep only videos
    sources_paths = [
        source_path
        for source_path in sources_paths
        if any(
            source_path.lower().endswith(extension)
            for extension in supported_video_extensions
        )
    ]

    # filter the audio paths: keep only the ones that contain the files to process.
    if files_to_process:
        sources_paths = [
            source_path
            for source_path in sources_paths
            if any(
                file_to_process.lower() in source_path.lower()
                for file_to_process in files_to_process
            )
        ]

    sources_paths.sort(reverse=enable_reverse_sorting)

    if enable_debug_prints:
        print(
            "main.main:: All sources paths: \n - ", ",\n - ".join(sources_paths), "\n\n"
        )

    # Create the folders for the numpy audio and the intermediate results
    create_folder_if_not_exists(tmp_numpy_audio_folder)
    create_folder_if_not_exists(tmp_intermediate_result_folder)

    # ! Step 1: Extract vocals only audios
    extract_vocals_only_audio(
        videos_paths=sources_paths,
        # segment=whisper_config.segment,
        # overlap=whisper_config.overlap,
        vocals_only_folder=tmp_numpy_audio_folder,
    )

    # ! Step 2: Transcribe the audios, producing the intermediate results/segments.
    transcribe_audios(
        whisper_config=whisper_config,
        audio_paths=sources_paths,
        tmp_transcripts_folder=tmp_intermediate_result_folder,
        tmp_np_audio_folder=tmp_numpy_audio_folder,
        enable_debug_prints=enable_debug_prints,
    )

    print("--- Aligning audios ---")

    # ! Step 3: Align the text-segments.
    generate_aligned_subtitles(
        audio_paths=sources_paths,
        tmp_numpy_audio_folder=tmp_numpy_audio_folder,
        tmp_intermediate_result_folder=tmp_intermediate_result_folder,
        align_model_config=aligner_config,
        debug_full_result_storage=True,
    )

    # clean up
    delete_folder_if_exists(folder_path=tmp_numpy_audio_folder)
    delete_folder_if_exists(folder_path=tmp_intermediate_result_folder)
