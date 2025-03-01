# Project Title
Subtitle Generation with Faster Whisper


# Description
This project utilizes HDemucs and WhisperX to generate subtitles from audio files. The pipeline is divided into three steps: 
1. Audio extraction using ffmpeg.
2. Audio separation using HDemucs
3. Subtitle generation using WhisperX (Speech-to-text + Alignment).
The main goal of this project is to provide a simple and efficient way to generate subtitles from audio files, also allowing future users to extend or replace the current steps of the pipeline.


# Table of Contents
* [Important Notes](#important-notes)
* [Installation](#installation)
* [Usage](#usage)
* [Features](#features)
* [Planned Improvements](#planned-improvements)
* [Contributing](#contributing)
* [License](#license)


# Installation
Follow the instructions on WhisperX [installation's page](https://github.com/m-bain/whisperX) for installing the required dependencies.


# Usage
The library is still WIP, so the usage will be updated continuously.
The current way to run the pipeline is to run the main.py file, which will run the pipeline.py module for handling all the steps.
```bash
python main.py
```

# Planned Improvements:
- Complete the lib API and config:
    - A YAML config file is WIP: there will be stored all the default parameters.
    - The API will be in the main.py file, allowing to run the pipeline thorugh:
        - CLI.
        - Python import (PIP install).
        - Docker.
- Update WhisperX to match new CTranslate and Torch, for easier setup.
- Make the library more modular, configurable and extensible, allowing new users to replace or extend the current steps of the pipeline.
- Develop and tune a new model for direct subtitle generation, capable of handling complex and long audios even in presence of background effects or silences, without relying on external tools such as VADs or vocals separation.


# License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
