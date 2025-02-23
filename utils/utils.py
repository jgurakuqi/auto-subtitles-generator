import json
import os
import shutil
from time import sleep
from torch.cuda import empty_cache as torch_empty_cache
from gc import collect as gc_collect
import os

import os


def get_path_without_file(file_path):
    """
    Gets the path of a file up to the file excluded.

    Args:
        file_path: The full path to the file.

    Returns:
        The path without the filename, or an empty string if the input is invalid or just a filename.
        Returns the directory even if it doesn't exist.
    """
    if not file_path:
        return ""

    directory = os.path.dirname(file_path)

    return directory


def build_path(
    folder_path: str, file_path: str, extension_replacement: str | None = None
) -> str:
    """Build a path by joining the folder path and the relative path with an optional extension replacement.

    Args:
        folder_path (str): The folder path.
        file_path (str): The path of the file.
        extension_replacement (str, optional): The replacement extension. Must include the dot. Defaults to None.

    Returns:
        str: The built path.
    """
    # if extension_replacement is not None: --> path = folder_path + base_filename + extension_replacement

    if extension_replacement is not None:
        if "." not in extension_replacement:
            extension_replacement = "." + extension_replacement
        built_path = os.path.join(
            folder_path,
            os.path.basename(file_path).replace(
                os.path.splitext(file_path)[-1], extension_replacement
            ),
        )
    else:
        built_path = os.path.join(folder_path, os.path.basename(file_path))
    return built_path


def get_filename_with_new_extension(filepath: str, new_extension: str) -> str | None:
    """
    Extracts the filename from a path and replaces its extension.

    Args:
        filepath: The path to the file (e.g., "./Sample/Trisam[;e/ADOMO/detrosium.mpg").
        new_extension: The new extension to use (e.g., "mp4").

    Returns:
        The filename with the new extension (e.g., "destrosium.mp4").
        Returns None if the input filepath is invalid or doesn't have an extension.
    """
    try:
        full_filename = os.path.basename(filepath)
        filename, extension = os.path.splitext(full_filename)

        if not extension:
            return None

        # Remove the leading dot from the new extension and return it
        return filename + "." + new_extension.lstrip(".")
    except Exception as e:
        print(f"utils.utils.get_filename_with_new_extension: An error occurred: {e}")
        return None


def read_all_paths_recursively(path):
    paths = []
    for root, _, filenames in os.walk(path):
        for filename in filenames:
            paths.append(os.path.join(root, filename))
    return paths


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def delete_folder_if_exists(folder_path: str) -> None:
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path, ignore_errors=True)


def store_json(data, output_path):
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def read_json(input_path):
    with open(input_path, "r") as f:
        data = json.load(f)
    return data


def cuda_force_collect(iterations=3):
    for _ in range(iterations):
        gc_collect()
        torch_empty_cache()
        gc_collect()
        sleep(0.5)
