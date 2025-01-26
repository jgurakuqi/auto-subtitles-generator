# import librosa
# import numpy as np
# import soundfile as sf


# def detect_speech_intervals(audio_path, threshold=0.01):
#     y, sr = librosa.load(audio_path, sr=36000)

#     # Root mean square (RMS) energy of the audio
#     rms = librosa.feature.rms(y=y, hop_length=512, frame_length=3200)[0]

#     # Convert RMS to time
#     times = librosa.times_like(rms, sr=sr)

#     # Detect speech intervals
#     speech_intervals = []
#     in_speech = False
#     start_time = 0

#     for i, energy in enumerate(rms):
#         current_time = times[i]

#         # If energy is above threshold and not currently in a speech segment
#         if energy > threshold and not in_speech:
#             start_time = current_time
#             in_speech = True

#         # If energy drops below threshold and was in a speech segment
#         elif energy <= threshold and in_speech:
#             end_time = current_time
#             speech_intervals.append((start_time, end_time))
#             in_speech = False

#     # Handle case if speech continues to the end of the file
#     if in_speech:
#         speech_intervals.append((start_time, times[-1]))

#     return speech_intervals


# # Example usage
# audio_file = "./vocals_only/test_1_vocals.wav"
# speech_segments = detect_speech_intervals(audio_file)


# def format_as_mm_ss(time):
#     minutes = int(time // 60)
#     seconds = int(time % 60)
#     return f"{minutes:02d}:{seconds:02d}"


# def merge_consecutive_timestamps(timestamps):
#     merged_timestamps = []
#     current_start = timestamps[0][0]
#     current_end = timestamps[0][1]

#     for start, end in timestamps[1:]:
#         if start - current_end <= 1.5:
#             current_end = end
#         else:
#             merged_timestamps.append((current_start, current_end))
#             current_start = start
#             current_end = end
#     merged_timestamps.append((current_start, current_end))
#     return merged_timestamps


# speech_segments = merge_consecutive_timestamps(speech_segments)

# with open("vad_v2_OUTPUT.txt", "w") as f:
#     for start, end in speech_segments:
#         f.write(f"{format_as_mm_ss(start)} - {format_as_mm_ss(end)}\n")


# # from time import sleep
# # import librosa
# # import numpy as np
# # import soundfile as sf


# # def detect_speech_intervals(audio_path, db_threshold=-40):
# #     # Load audio with a consistent sample rate
# #     y, sr = librosa.load(audio_path, sr=20000)

# #     # Calculate RMS energy and convert to decibels
# #     rms = librosa.feature.rms(y=y, hop_length=256, frame_length=2800)[0]
# #     db_levels = librosa.amplitude_to_db(rms)

# #     # Convert RMS to time
# #     times = librosa.times_like(rms, sr=sr)

# #     del rms

# #     # Detect speech intervals
# #     speech_intervals = []
# #     in_speech = False
# #     start_time = 0

# #     with open("vad_v2_RAW_OUTPUT.txt", "w") as f:
# #         for i, db_level in enumerate(db_levels):
# #             current_time = times[i]

# #             f.write(f"{format_as_mm_ss(current_time)} - {db_level:.4f}\n")

# #             # If decibel level is above threshold and not currently in a speech segment
# #             if db_level > db_threshold and not in_speech:
# #                 start_time = current_time
# #                 in_speech = True
# #             else:
# #                 # print curr decibel and its time

# #                 # If decibel level drops below threshold and was in a speech segment
# #                 if db_level <= db_threshold and in_speech:
# #                     end_time = current_time
# #                     speech_intervals.append((start_time, end_time))
# #                     in_speech = False

# #         # Handle case if speech continues to the end of the file
# #         if in_speech:
# #             speech_intervals.append((start_time, times[-1]))

# #     return speech_intervals


# # def format_as_mm_ss(time):
# #     minutes = int(time // 60)
# #     # seconds = int(time % 60)
# #     seconds = round(time % 60, 3)
# #     # return f"{minutes:02d}:{seconds:02d}"
# #     return f"{minutes:02d}:{seconds}"


# # def merge_consecutive_timestamps(timestamps, max_gap=1.5):
# #     if not timestamps:
# #         return []

# #     merged_timestamps = []
# #     current_start = timestamps[0][0]
# #     current_end = timestamps[0][1]

# #     for start, end in timestamps[1:]:
# #         if start - current_end <= max_gap:
# #             current_end = end
# #         else:
# #             merged_timestamps.append((current_start, current_end))
# #             current_start = start
# #             current_end = end

# #     merged_timestamps.append((current_start, current_end))
# #     return merged_timestamps


# # # Example usage
# # audio_file = "./vocals_only/test_1_vocals.wav"
# # speech_segments = detect_speech_intervals(audio_file, db_threshold=-40)
# # speech_segments = merge_consecutive_timestamps(speech_segments)

# # # Write output to file
# # with open("vad_v2_OUTPUT.txt", "w") as f:
# #     for start, end in speech_segments:
# #         f.write(f"{format_as_mm_ss(start)} - {format_as_mm_ss(end)}\n")


# # # open the file vad_v2_RAW_OUTPUT.txt and write a new file vad_v2_RAW_OUTPUT_SECOND.txt, where for each second
# # # it stores an avarage of the decibel levels. E.g., in the original file there is 00:01.1, 00:01.2, 00:01.3 etc.,
# # # each with a decibel level, so the new file should collect for a same second all the decibels and store its average
# # # hence there will be just 00:01  with the average decibel
# # def average_decibels():
# #     with open("vad_v2_RAW_OUTPUT.txt", "r") as f:
# #         lines = f.readlines()

# #     # create a dictionary to store the average decibel for each second
# #     decibels = {}

# #     for line in lines:
# #         time, decibel = line.strip().split(" - ")
# #         seconds = int(time.split(":")[1])
# #         if seconds not in decibels:
# #             decibels[seconds] = []
# #         decibels[seconds].append(float(decibel))

# #     # calculate the average decibel for each second
# #     for seconds, decibels in decibels.items():
# #         average = sum(decibels) / len(decibels)
# #         print(f"{seconds:02d} - {average:.4f}")

# #     # write the average decibels to a new file
# #     with open("vad_v2_RAW_OUTPUT_SECOND.txt", "w") as f:
# #         for seconds, decibels in decibels.items():
# #             average = sum(decibels) / len(decibels)
# #             f.write(f"{seconds:02d} - {average:.4f}\n")

# #     print("Average decibels written to vad_v2_RAW_OUTPUT_SECOND.txt")


# # average_decibels()


import os
import webrtcvad
import wave
import numpy as np


def perform_vad(audio_path, aggressiveness=3):
    """
    Perform Voice Activity Detection on an audio file.

    Parameters:
    - audio_path: Path to the audio file (must be a mono WAV file).
    - aggressiveness: VAD aggressiveness mode (0-3). Higher values are more aggressive.

    Returns:
    - List of tuples indicating the start and end times of detected voice segments.
    """
    # Initialize VAD with the specified aggressiveness
    vad = webrtcvad.Vad(aggressiveness)

    # Open the wave file
    with wave.open(audio_path, "rb") as wf:
        # Ensure the audio is mono
        if wf.getnchannels() != 1:
            raise ValueError("Audio must be mono channel")

        # Get audio parameters
        sample_rate = wf.getframerate()
        frame_duration = 30  # Frame duration in ms (valid options: 10, 20, 30 ms)
        frame_size = int(sample_rate * frame_duration / 1000)

        # List to store voice segments
        voice_segments = []
        current_segment_start = None

        while True:
            # Read a frame of audio
            frame = wf.readframes(frame_size)
            if len(frame) == 0:
                break

            # Convert frame to int16 numpy array
            frame_data = np.frombuffer(frame, dtype=np.int16).tobytes()

            # Check if the frame contains speech
            is_speech = vad.is_speech(frame_data, sample_rate)

            # Track voice segments
            if is_speech and current_segment_start is None:
                current_segment_start = wf.tell() / (sample_rate * wf.getsampwidth())
            elif not is_speech and current_segment_start is not None:
                segment_end = wf.tell() / (sample_rate * wf.getsampwidth())
                voice_segments.append((current_segment_start, segment_end))
                current_segment_start = None

        # Handle the last segment if the audio ends with speech
        if current_segment_start is not None:
            segment_end = wf.tell() / (sample_rate * wf.getsampwidth())
            voice_segments.append((current_segment_start, segment_end))

        return voice_segments


from pydub import AudioSegment


def convert_to_mono(input_path, output_path):
    # Load the audio file
    audio = AudioSegment.from_file(input_path)

    # Convert to mono
    mono_audio = audio.set_channels(1)

    # Export the mono audio
    mono_audio.export(output_path, format="wav")


if not os.path.exists("./vocals_only/test_1_vocals_mono.wav"):
    # Example usage
    input_audio_path = "./vocals_only/test_1_vocals.wav"
    output_audio_path = "./vocals_only/test_1_vocals_mono.wav"
    convert_to_mono(input_audio_path, output_audio_path)


# Example usage
audio_path = "./vocals_only/test_1_vocals_mono.wav"
voice_segments = perform_vad(audio_path)

print("Voice Segments (start_time, end_time):")
for start, end in voice_segments:
    print(f"{start:.2f}s - {end:.2f}s")
