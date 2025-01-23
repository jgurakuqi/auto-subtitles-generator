# ./utils/utils.py

import logging, os
from datetime import datetime

video_extensions = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".flv",
    ".wmv",
    ".webm",
    ".mpeg",
    ".mpg",
)


def build_path(
    folder_path: str, file_rel_path: str, extension_replacement: str | None = None
) -> str:
    """Build a path by joining the folder path and the relative path with an optional extension replacement.

    Args:
        folder_path (str): The folder path.
        file_rel_path (str): The relative path of the file.
        extension_replacement (str, optional): The replacement extension. Defaults to None.

    Returns:
        str: The built path.
    """
    if extension_replacement is not None:
        built_path = os.path.join(
            folder_path, os.path.splitext(file_rel_path)[0] + extension_replacement
        )
    else:
        built_path = os.path.join(folder_path, file_rel_path)
    return built_path


def initialize_logging(
    logs_folder_path: str, log_level: int = logging.DEBUG, log_to_console: bool = True
) -> None:
    """Initialize logging for the application.

    Args:
        logs_folder_path (str): The path to the folder where log files will be saved.
        log_level (int, optional): The logging level. Defaults to logging.DEBUG.
        log_to_console (bool, optional): If True, also log to the console. Defaults to False.

    Returns:
        None
    """
    if not os.path.exists(logs_folder_path):
        os.makedirs(logs_folder_path)

    current_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    log_file = os.path.join(logs_folder_path, f"{current_datetime}.log")

    # Create a file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    # Get the root logger
    logger = logging.getLogger("auto-sub-gen")
    logger.setLevel(log_level)
    logger.addHandler(file_handler)

    # Optionally add a console handler
    if log_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)


# def sort_key(x):
#     if "special" in x.lower():
#         return -1
#     else:
#         base_filename = os.path.basename(x)
#         return int(base_filename.lower().split("episode ")[1].split(" - ")[0])


def recursively_read_video_paths(folder_path: str) -> list[str]:
    """Recursively read video paths from a folder.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        list[str]: A list of video paths.
    """
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files
        if file.endswith(video_extensions)
    ]


def format_time(seconds: float) -> str:
    """Format time in the format HH:MM:SS,SSS

    Args:
        seconds (float): Time in seconds

    Returns:
        str: Formatted time
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"


def create_folder_if_not_exists(folder_path: str) -> None:
    """Create a folder if it does not exist.

    Args:
        folder_path (str): The path to the folder.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
