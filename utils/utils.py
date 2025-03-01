from json import dump as json_dump, load as json_load
from os import walk as os_walk, makedirs as os_makedirs
from os.path import (
    join as os_path_join,
    basename as os_path_basename,
    splitext as os_path_splitext,
    exists as os_path_exists,
    dirname as os_path_dirname,
)
from shutil import rmtree as shutil_rmtree
from torch.cuda import empty_cache as torch_empty_cache
from gc import collect as gc_collect


def get_path_without_file(file_path: str) -> str:
    """
    Gets the path of a file up to the file excluded.

    Args:
        file_path (str): The full path to the file.

    Returns:
        str: The path without the filename, or an empty string if the input is invalid or just a filename.
    """
    if not file_path:
        return ""
    directory = os_path_dirname(file_path)
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
        built_path = os_path_join(
            folder_path,
            os_path_basename(file_path).replace(
                os_path_splitext(file_path)[-1], extension_replacement
            ),
        )
    else:
        built_path = os_path_join(folder_path, os_path_basename(file_path))
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
        full_filename = os_path_basename(filepath)
        filename, extension = os_path_splitext(full_filename)

        if not extension:
            return None

        # Remove the leading dot from the new extension and return it
        return filename + "." + new_extension.lstrip(".")
    except Exception as e:
        print(f"utils.utils.get_filename_with_new_extension: An error occurred: {e}")
        return None


def read_all_paths_recursively(path: str) -> list[str]:
    """
    Recursively reads all paths in a directory.

    Args:
        path (str): The path to the directory.

    Returns:
        list[str]: A list of all the paths in the directory.
    """
    paths = []
    for root, _, filenames in os_walk(path):
        for filename in filenames:
            paths.append(os_path_join(root, filename))
    return paths


def create_folder_if_not_exists(folder_path: str) -> None:
    """
    Creates a folder if it doesn't exist.

    Args:
        folder_path (str): The path to the folder.
    """
    if not os_path_exists(folder_path):
        os_makedirs(folder_path)


def delete_folder_if_exists(folder_path: str) -> None:
    """
    Deletes a folder if it exists.

    Args:
        folder_path (str): The path to the folder.
    """
    if os_path_exists(folder_path):
        shutil_rmtree(folder_path, ignore_errors=True)


def store_json(data: dict, output_path: str) -> None:
    """
    Store a dictionary as a json file.

    Args:
        data (dict): The dictionary to store.
        output_path (str): The path to the output file.
    """
    with open(output_path, "w") as f:
        json_dump(data, f, indent=4)


def read_json(input_path: str) -> dict | list[dict]:
    """
    Read a json file.

    Args:
        input_path (str): The path to the input file.

    Returns:
        dict: The dictionary/list of dictionaries read from the file.
    """
    with open(input_path, "r") as f:
        data = json_load(f)
    return data


def cuda_force_collect(iterations: int = 2) -> None:
    """
    Collect garbage and empty the CUDA cache multiple times.

    Args:
        iterations (int, optional): The number of times to collect garbage and empty the CUDA cache. Defaults to 2.
    """
    for _ in range(iterations):
        gc_collect()
        torch_empty_cache()
