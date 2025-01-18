# Project Title
Subtitle Generation with Faster Whisper

# Description
This project utilizes Faster Whisper to generate subtitles from audio files. It leverages ffmpeg for audio extraction and pre-processing, and Faster Whisper for generating .srt subtitle files.


# Table of Contents
* [Important Notes](#important-notes)
* [Installation](#installation)
* [Usage](#usage)
* [Features](#features)
* [Contributing](#contributing)
* [License](#license)

# Important Notes
- IMPORTANT: This project is based on Faster Whisper 1.1.1, and currently i cannot ensure the compatibility with future versions.
- Even if available, I largely discourage using the CHUNKED-BATCHED processing of videos, as there seems to be a bug related to Whisper itself which causes a large degradation of the accuracy in correspondence of the chunks' boundaries, leading to shifted, repeated or missing subtitles. Even disabling VAD or using VAD with the same settings across BATCHED and non-BATCHED processing, the issue persists.
- By default i keep enabled the VAD (Voice Activity Detection) filter, otherwise long silences cause evident hallucinations, with generation of completely unrelated subtitles. Unfortunately, the VAD filter decreases a bit the accuracy over the sections immediately adjacent to the silences, so you are presented with a tradeoff:
    - VAD filter disabled: hallucinations, but no loss of subtitles near silences.
    - VAD filter enabled: no hallucinations, but loss of subtitles near silences.
 I opted for the second option, as overall produces more accurate results. Also I used custom settings for VAD filter, with very permissive threshold and minimum silence duration, to avoid as much as possible hallucinations and at the same time avoid missing subtitles.

# Installation

## Prerequisites
- Python 3.x
- ffmpeg installed on your system
- [Faster Whisper library](https://github.com/SYSTRAN/faster-whisper)

## Steps
1. Clone the repository:

```bash
git clone https://github.com/yourusername/yourprojectname.git
cd yourprojectname
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Ensure ffmpeg is installed and accessible from the command line. You can download it from [FFmpeg's official website](https://www.ffmpeg.org/download.html).

## Usage
1. Extract audio from a video file using ffmpeg:

```bash
ffmpeg -i input_video.mp4 -q:a 0 -map a audio.wav
```

2. Run the subtitle generation script:

```bash	
python generate_subtitles.py --audio audio.wav --output subtitles.srt
```

3. The generated subtitles will be saved as subtitles.srt.

# Features
- Extract audio from video files using ffmpeg.
- Generate subtitles using Faster Whisper.

# Contributing
- ## Work in progress

```bash	
# Print the list of currently installed packages
pip freeze
```

# License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
