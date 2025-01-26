import webrtcvad
from pydub import AudioSegment


def read_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(16000)  # Set frame rate to 16kHz
    return audio


def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio.raw_data):
        yield audio.raw_data[offset : offset + n], timestamp
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = []
    triggered = False

    voiced_frames = []
    voiced_intervals = []
    start_time = 0

    for frame, timestamp in frames:
        is_speech = vad.is_speech(frame, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech, timestamp))
            if len(ring_buffer) > num_padding_frames:
                ring_buffer.pop(0)
            num_voiced = len([f for f, speech, _ in ring_buffer if speech])
            if num_voiced > 0.9 * num_padding_frames:
                triggered = True
                start_time = ring_buffer[0][2]
                voiced_frames.extend([f for f, s, _ in ring_buffer])
                ring_buffer = []
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech, timestamp))
            num_unvoiced = len([f for f, speech, _ in ring_buffer if not speech])
            if num_unvoiced > 0.9 * num_padding_frames:
                triggered = False
                end_time = timestamp + (float(len(frame)) / sample_rate) / 2.0
                voiced_intervals.append((start_time, end_time))
                ring_buffer = []
                voiced_frames = []

    if voiced_frames:
        end_time = timestamp + (float(len(frame)) / sample_rate) / 2.0
        voiced_intervals.append((start_time, end_time))

    return voiced_intervals


def merge_consecutive_timestamps(timestamps):
    merged_timestamps = []
    current_start = timestamps[0][0]
    current_end = timestamps[0][1]

    for start, end in timestamps[1:]:
        if start - current_end <= 2.0:
            current_end = end
        else:
            merged_timestamps.append((current_start, current_end))
            current_start = start
            current_end = end
    merged_timestamps.append((current_start, current_end))
    return merged_timestamps


def main(file_path, output_file):
    audio = read_audio(file_path)
    sample_rate = audio.frame_rate
    vad = webrtcvad.Vad(0)  # Set aggressiveness mode (0-3)

    frames = frame_generator(2, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 2, 1000, vad, frames)

    segments = merge_consecutive_timestamps(segments)

    with open(output_file, "w") as f:
        for start, end in segments:
            f.write(f"{start:.2f} {end:.2f}\n")


if __name__ == "__main__":
    main("./vocals_only/test_1_vocals.wav", "./silero_test/vad_web.txt")
