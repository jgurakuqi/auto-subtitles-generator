from utils.vocals_separator import extract_vocals_only_audio


if __name__ == "__main__":
    input_audio_paths = [
        "./exa.mp4",
    ]

    extract_vocals_only_audio(input_audio_paths)
