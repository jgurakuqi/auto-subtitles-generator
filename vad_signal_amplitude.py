# import json
# from time import sleep
# import librosa
# import numpy as np


# def compute_amplitude(audio_file, target_sr=41000, frame_length_ms=100):
#     try:
#         # Load and downsample the audio
#         y, sr = librosa.load(audio_file, sr=target_sr)
#         # Calculate the number of samples per frame
#         frame_length = int(sr * frame_length_ms / 1000)
#         # Calculate the amplitude for each frame
#         frame_amplitudes = [
#             np.mean(np.abs(y[i : i + frame_length])) * 100
#             for i in range(0, len(y), frame_length)
#         ]
#         return frame_amplitudes, sr
#     except Exception as e:
#         print(f"Error loading or processing audio: {e}")
#         return None, None


# # Example usage:
# # audio_file = "./vocals_only/test_1_vocals.wav"
# audio_file = "./vocals_only/NaruCannon S04E18 Tsunade's Choice (Dub)_vocals.wav"
# frame_length_ms = 100
# amplitude, sr = compute_amplitude(
#     audio_file=audio_file, frame_length_ms=frame_length_ms
# )
# print("Computed amplitude")

# if amplitude is not None:
#     # Calculate the time for each frame
#     times = np.arange(len(amplitude)) * (frame_length_ms / 1000)  # 100 ms per frame

#     conv_times = [
#         f"{int(t // 3600):02d}:{int((t % 3600) // 60):02d}:{(t % 60):02.1f}"
#         for t in times
#     ]

#     # Store each amplitude along with the related converted time in a txt file
#     with open("spectrum_analysis.txt", "w") as f:
#         for time, amp in zip(conv_times, amplitude):
#             f.write(f"{time} - {round(amp, 6)}\n")

#     print("Stored amplitudes in spectrum_analysis.txt")

#     # Filter away silences longer than 1.0 seconds
#     end_of_silence = None
#     end_of_silence_idx = None

#     conv_times : list[str] = list(conv_times)
#     times : list[float]  = list(times)
#     amplitude : list[float] = list(amplitude)

#     min_silence_duration_in_seconds = 1.0
#     for i in range(len(amplitude) - 1, -1, -1):
#         if amplitude[i] < 1.0:
#             if end_of_silence == None:
#                 end_of_silence = i
#                 end_of_silence_idx = i
#             elif i == 0 and times[end_of_silence_idx] - times[i] >= min_silence_duration_in_seconds: # type: ignore
#                 del conv_times[0 : end_of_silence + 1]
#                 del times[0 : end_of_silence + 1]
#                 del amplitude[0 : end_of_silence + 1]
#         else:
#             if end_of_silence != None:
#                 start_of_silence = i + 1
#                 if (
#                     times[end_of_silence_idx] - times[start_of_silence] # type: ignore
#                     >= min_silence_duration_in_seconds
#                 ):
#                     del conv_times[start_of_silence : end_of_silence + 1]
#                     del times[start_of_silence : end_of_silence + 1]
#                     del amplitude[start_of_silence : end_of_silence + 1]
#                 end_of_silence = None
#                 end_of_silence_idx = None

#     # Store each amplitude along with the related converted time in a txt file, but printing an extra \n if the difference
#     # between consecutive timestamps is greater than frame_length_ms
#     times = [round(time, 1) for time in times]
#     min_diff = frame_length_ms / 1000 * 1.001
#     start = None

#     all_segments : list[dict[str, float]] = []
#     with open("spectrum_analysis_filtered.txt", "w") as f:
#         for i in range(len(times) - 1):
#             if times[i + 1] - times[i] > min_diff:
#                 f.write(f"{conv_times[i]} - {round(amplitude[i], 6)}\n\n")
#                 if start != None:
#                     all_segments.append({"start": start, "end" : times[i]})
#                     start = None
#             else:
#                 if start == None:
#                     start = times[i]
#                 f.write(f"{conv_times[i]} - {round(amplitude[i], 6)}\n")


#     with open("spectrum_analysis_timestamps.json", "w") as f:
#         json.dump(all_segments, f, indent=4)


import json
from time import sleep
import librosa
import numpy as np
import os


def compute_amplitude(
    audio_file: str, target_sr: int, frame_length_ms: int
) -> tuple[np.ndarray, int | float] | tuple[None, None]:
    try:
        y, sr = librosa.load(audio_file, sr=target_sr)

        # Calculate the number of samples per frame
        frame_length = int(sr * frame_length_ms / 1000)

        # Calculate the amplitude for each frame
        frame_amplitudes = [
            np.mean(np.abs(y[i : i + frame_length])) * 100
            for i in range(0, len(y), frame_length)
        ]
        return frame_amplitudes, sr
    except Exception as e:
        print(f"Error loading or processing audio: {e}")
        return None, None


def perform_amplitude_vad(
    audio_paths,
    frame_length_ms=100,
    amplitude_threshold=1.0,
    min_silence_duration_in_seconds=1.0,
    debug=False,
    timestamps_folder="./timestamps/",
    sample_rate: int = 41000,
):

    for audio_path in audio_paths:
        amplitude, sr = compute_amplitude(
            audio_file=audio_path,
            frame_length_ms=frame_length_ms,
            target_sr=sample_rate,
        )

        if amplitude is not None:
            times = np.arange(len(amplitude)) * (
                frame_length_ms / 1000
            )  # 100 ms per frame

            if debug:
                with open(
                    os.path.join(
                        timestamps_folder,
                        os.path.basename(audio_path).replace(
                            ".wav", "_spectrum_raw.txt"
                        ),
                    ),
                    "w",
                ) as f:
                    for time, amp in zip(times, amplitude):
                        f.write(f"{time} - {round(amp, 6)}\n")

            end_of_silence = None
            end_of_silence_idx = None

            times: list[float] = list(times)
            amplitude: list[float] = list(amplitude)

            for i in range(len(amplitude) - 1, -1, -1):
                if amplitude[i] < amplitude_threshold:
                    if end_of_silence == None:
                        end_of_silence = i
                        end_of_silence_idx = i
                    elif i == 0 and times[end_of_silence_idx] - times[i] >= min_silence_duration_in_seconds:  # type: ignore
                        del times[0 : end_of_silence + 1]
                        del amplitude[0 : end_of_silence + 1]
                else:
                    if end_of_silence != None:
                        start_of_silence = i + 1
                        if (
                            times[end_of_silence_idx] - times[start_of_silence]  # type: ignore
                            >= min_silence_duration_in_seconds
                        ):
                            del times[start_of_silence : end_of_silence + 1]
                            del amplitude[start_of_silence : end_of_silence + 1]
                        end_of_silence = None
                        end_of_silence_idx = None

            times = [round(time, 1) for time in times]
            min_diff = frame_length_ms / 1000 * 1.001
            start = None

            all_segments: list[dict[str, float]] = []
            for i in range(len(times) - 1):
                if times[i + 1] - times[i] > min_diff:
                    if start != None:
                        all_segments.append({"start": start, "end": times[i]})
                        start = None
                elif start == None:
                    start = times[i]

            with open(
                os.path.join(
                    timestamps_folder,
                    os.path.basename(audio_path).replace(".wav", "_timestamps.json"),
                ),
                "w",
            ) as f:
                json.dump(all_segments, f, indent=4)
