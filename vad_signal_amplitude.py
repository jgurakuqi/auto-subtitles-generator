from time import sleep
import librosa
import numpy as np


def compute_amplitude(audio_file, target_sr=41000, frame_length_ms=100):
    try:
        # Load and downsample the audio
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


# Example usage:
audio_file = "./vocals_only/test_1_vocals.wav"
frame_length_ms = 100
amplitude, sr = compute_amplitude(
    audio_file=audio_file, frame_length_ms=frame_length_ms
)
print("Computed amplitude")

if amplitude is not None:
    # Calculate the time for each frame
    times = np.arange(len(amplitude)) * (100 / 1000)  # 100 ms per frame
    # Convert times to hh:mm:ss format
    # conv_times = [
    #     f"{int(t // 3600):02d}:{int((t % 3600) // 60):02d}:{(t % 60):02.4f}"
    #     for t in times
    # ]
    conv_times = [
        f"{int(t // 3600):02d}:{int((t % 3600) // 60):02d}:{(t % 60):02.1f}"
        for t in times
    ]

    # Store each amplitude along with the related converted time in a txt file
    with open("spectrum_analysis.txt", "w") as f:
        for time, amp in zip(conv_times, amplitude):
            f.write(f"{time} - {round(amp, 6)}\n")

    print("Stored amplitudes in spectrum_analysis.txt")

    # Filter away silences longer than 1.0 seconds
    end_of_silence = None
    end_of_silence_idx = None

    conv_times = list(conv_times)
    times = list(times)
    amplitude = list(amplitude)

    min_silence_duration_in_seconds = 1.0
    for i in range(len(amplitude) - 1, -1, -1):
        if amplitude[i] < 1.0:
            if end_of_silence == None:
                end_of_silence = i
                end_of_silence_idx = i
        else:
            if end_of_silence != None:
                start_of_silence = i + 1
                if (
                    times[end_of_silence_idx] - times[start_of_silence]
                    >= min_silence_duration_in_seconds
                ):
                    del conv_times[start_of_silence : end_of_silence + 1]
                    del times[start_of_silence : end_of_silence + 1]
                    del amplitude[start_of_silence : end_of_silence + 1]
                end_of_silence = None
                end_of_silence_idx = None

    # Store each amplitude along with the related converted time in a txt file, but printing an extra \n if the difference
    # between consecutive timestamps is greater than frame_length_ms
    times = [round(time, 1) for time in times]
    min_diff = frame_length_ms / 1000 * 1.001
    with open("spectrum_analysis_filtered.txt", "w") as f:
        for i in range(len(times) - 1):
            if times[i + 1] - times[i] > min_diff:
                # print(
                #     "Times: ", times[i], times[i + 1], "diff: ", times[i + 1] - times[i]
                # )
                # sleep(5)
                f.write(f"{conv_times[i]} - {round(amplitude[i], 6)}\n\n")
            else:
                f.write(f"{conv_times[i]} - {round(amplitude[i], 6)}\n")

    # Now store in a file "spectrum_analysis_timestamps.txt" all the intervals
