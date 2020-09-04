import numpy as np
from pydub import AudioSegment


def read_file(filename):
	return AudioSegment.from_file(filename, filename.split(".")[-1])


def flatten(segments):
    output = []
    for segment in segments:
        output.extend(segment.get_array_of_samples())
    return np.array(output, dtype=np.int16)


def amplify_segments(segments):
    x = flatten(segments)
    x = amplify(x, window_size=24000)
    output = to_segment(x, segments.frame_rate, segments.sample_width, segments.channels)
    file_handle = output.export("D:\\output.mp3", format="mp3")


def to_segment(samples, frame_rate=48000, sample_width=2, channels=1):
    """
    data: raw audio data (bytes)
    sample_width: 2 byte (16 bit) samples
    frame_rate: samples per second
    channels: Number of channels

    """
    data = samples.tobytes(order='C')
    print(len(data))
    print(frame_rate, sample_width, channels)
    return AudioSegment(data=data, sample_width=sample_width, frame_rate=frame_rate, channels=channels)


def play(segments):
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
    pydub_play(segments)

