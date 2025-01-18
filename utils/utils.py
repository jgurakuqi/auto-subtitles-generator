import os

# from faster_whisper import WhisperModel, BatchedInferencePipeline

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


def sort_key(x):
    if "special" in x.lower():
        return -1
    else:
        base_filename = os.path.basename(x)
        return int(base_filename.lower().split("episode ")[1].split(" - ")[0])


def recursively_read_video_paths(folder_path: str) -> list[str]:

    return [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files
        if file.endswith(video_extensions)
    ]


# def load_batched_whisper(
#     model_id: str, device: str, compute_type: str, num_workers: int
# ) -> BatchedInferencePipeline:
#     return BatchedInferencePipeline(
#         WhisperModel(
#             model_id, device=device, compute_type=compute_type, num_workers=num_workers
#         )
#     )


# def load_std_whisper(
#     model_id: str, device: str, compute_type: str, num_workers: int
# ) -> WhisperModel:
#     return WhisperModel(
#         model_id, device=device, compute_type=compute_type, num_workers=num_workers
#     )


# # Function to convert seconds to SRT time format
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"
