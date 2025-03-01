from configs.pipeline_config import PipelineConfig
from utils.transcription import transcribe_audios
from utils.subtitles import generate_aligned_subtitles
from utils.utils import (
    filter_video_sources_by_filenames,
    read_all_paths_recursively,
    create_folder_if_not_exists,
)
from utils.utils import delete_folder_if_exists, filter_video_sources
from utils.audio import extract_vocals_only_audio


def pipeline(
    input_folder_path: str,
    pipeline_config: PipelineConfig,
    enable_reverse_sorting: bool = True,
    files_to_process: str = [],
    enable_debug_prints: bool = True,
):
    """
    Generate subtitles for all the videos in the provided folder.

    Args:
        input_folder_path (str): The path to the folder containing the videos.
        pipeline_config (PipelineConfig): The configuration for the pipeline. Contains the configurations for
            each step of the pipeline.
        enable_reverse_sorting (bool, optional): Enable reverse sorting. Defaults to True.
        files_to_process (str, optional): The files to process. Defaults to [].
        enable_debug_prints (bool, optional): Enable debug prints. Defaults to True.
        tmp_numpy_audio_folder (str, optional): The temporary folder for numpy audio. Here will be stored
            the audios extracted as numpy arrays, to be used also for the alignment. Defaults to "./tmp_numpy_audio/".
        tmp_intermediate_result_folder (str, optional): The temporary folder for intermediate results. Here will be
            stored the segments produced by the Whisper model. They are needed for the alignment. Defaults to "./tmp_intermediate_result/".
        supported_video_extensions (list, optional): The supported video extensions. Defaults to ["mp4", "avi", "mov", "wmv", "mkv"].
    """
    # clean up temp folders possibly left from interrupted execution.
    delete_folder_if_exists(folder_path=pipeline_config.tmp_numpy_audio_folder)
    delete_folder_if_exists(folder_path=pipeline_config.tmp_intermediate_result_folder)

    if enable_debug_prints:
        print("main.main:: input_folder_path: ", input_folder_path)
    sources_paths = read_all_paths_recursively(input_folder_path)

    # Filter the source paths to keep only videos
    sources_paths = filter_video_sources(
        sources_paths=sources_paths,
        supported_video_extensions=pipeline_config.supported_video_extensions,
    )

    # filter the audio paths: keep only the ones that contain the files to process.
    sources_paths = filter_video_sources_by_filenames(
        sources_paths=sources_paths,
        files_to_process=files_to_process,
    )

    sources_paths.sort(reverse=enable_reverse_sorting)

    if enable_debug_prints:
        print(
            "main.main:: All sources paths: \n - ", ",\n - ".join(sources_paths), "\n\n"
        )

    # Create the folders for the numpy audio and the intermediate results
    create_folder_if_not_exists(pipeline_config.tmp_numpy_audio_folder)
    create_folder_if_not_exists(pipeline_config.tmp_intermediate_result_folder)

    # ! Step 1: Extract vocals only audios
    extract_vocals_only_audio(
        videos_paths=sources_paths,
        pipeline_config=pipeline_config,
    )

    # ! Step 2: Transcribe the audios, producing the intermediate results/segments.
    transcribe_audios(
        audio_paths=sources_paths,
        pipeline_config=pipeline_config,
        enable_debug_prints=enable_debug_prints,
    )

    print("--- Aligning audios ---")

    # ! Step 3: Align the text-segments.
    generate_aligned_subtitles(
        audio_paths=sources_paths,
        pipeline_config=pipeline_config,
        debug_full_result_storage=True,
    )

    # clean up
    delete_folder_if_exists(folder_path=pipeline_config.tmp_numpy_audio_folder)
    delete_folder_if_exists(folder_path=pipeline_config.tmp_intermediate_result_folder)
