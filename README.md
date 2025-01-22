# Project Title
Subtitle Generation with Faster Whisper

# Description
This project utilizes Faster Whisper to generate subtitles from audio files. It leverages ffmpeg for audio extraction and pre-processing, and Faster Whisper for generating .srt subtitle files.


# Table of Contents
* [Important Notes](#important-notes)
* [Installation](#installation)
* [Usage](#usage)
* [Features](#features)
* [Planned Improvements](#planned-improvements)
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
git clone https://github.com/jgurakuqi/auto-subtitles-generator
cd auto-subtitles-generator
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

3. Ensure ffmpeg is installed and accessible from the command line. You can download it from [FFmpeg's official website](https://www.ffmpeg.org/download.html).

4. Follow the requirements for [Faster Whisper](https://github.com/SYSTRAN/faster-whisper).

## Usage

1. Work in progress: currently the whole execution is handled through the `main.py` file. You can run it as follows:

```bash
python main.py
```

2. The script will use the parameters in the main to determine the audio extraction and subtitle generation process.

3. At the end, the srt files will be generated in the same folder of the video files.

# Planned Improvements:

The project is quite recent, and I plan of:
- Introducing the possiblity of chosing the parameters through command line and/or yaml file.
- Make the code more modular.
- Provide more user-friendly output: e.g., path of the generated subtitles, ....
- POSSIBLY integrate Audio-tracks separator model for better audio quality. The model will entirely remove the background noise and music, leaving only the voices for the model to focus on. This, in conjunction with accurate checks over the source audio, could lead to much better subtitle quality, also allowing to disable the VAD filter (e.g., check in the voices only track if there are voices, and hence remove hallucations when no voice are detected).
- POSSIBLE IDEA: Introduce a GUI for a more user-friendly experience.


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
