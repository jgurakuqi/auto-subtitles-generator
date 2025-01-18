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
IMPORTANT: This project is based on Faster Whisper 1.1.1, and currently i cannot ensure the compatibility with future versions.


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
