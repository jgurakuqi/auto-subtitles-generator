import subprocess


def extract_high_res_audio(input_path: str, output_path: str) -> None:
    subprocess.run(
        args=[
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vn",
            "-af",
            "volume=2.0",
            "-acodec",
            "pcm_s16le",
            "-ar",  # 41khz
            "41000",
            "-ac",
            "1",
            output_path,
        ],
        check=True,
    )
