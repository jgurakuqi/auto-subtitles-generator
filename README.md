# Project Title
Subtitle Generation with Faster Whisper

# Description
This project utilizes HDemucs and WhisperX to generate subtitles from audio files. The pipeline is divided into three steps: 
1. Audio extraction using ffmpeg.
2. Audio separation using HDemucs
3. Subtitle generation using WhisperX (Speech-to-text + Alignment).


# Table of Contents
* [Important Notes](#important-notes)
* [Installation](#installation)
* [Usage](#usage)
* [Features](#features)
* [Planned Improvements](#planned-improvements)
* [Contributing](#contributing)
* [License](#license)

# Installation

## Prerequisites
- Python 3.x
- ffmpeg installed on your system
- WhisperX requirements 



## Steps - WIP
1. Clone the repository:

```bash
git clone https://github.com/jgurakuqi/auto-subtitles-generator
cd auto-subtitles-generator
```

<!-- 2. Install the required Python packages:
```bash
pip install -r requirements.txt
``` -->

2. Ensure ffmpeg is installed and accessible from the command line. You can download it from [FFmpeg's official website](https://www.ffmpeg.org/download.html).

3. Follow the requirements for WhisperX.

## Usage - WIP

# Planned Improvements:

- Update WhisperX to match new CTranslate and Torch, for easier setup.
- Make the library accessible through:
    - Python import (PIP install)
    - CLI
    - Docker


# Contributing
- ## Work in progress

<!-- ```bash	
# Print the list of currently installed packages
pip freeze
``` -->

# License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
