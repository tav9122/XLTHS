import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt


def read_audio(audio_file_path):
    audio_samples, sample_rate = sf.read(audio_file_path)
    return audio_samples, sample_rate


def calculate_frame_and_hop_sizes(sample_rate, frame_time, hop_time):
    frame_size = int(frame_time * sample_rate)
    hop_size = int(hop_time * sample_rate)
    return frame_size, hop_size


def calculate_ste_and_zcr(audio_samples, frame_size, hop_size):
    ste = []
    zcr = []
    for i in range(0, len(audio_samples) - frame_size, hop_size):
        frame = audio_samples[i:i + frame_size]
        ste.append(np.sum(frame ** 2))
        zcr.append(np.sum(np.diff(np.sign(frame)) != 0))
    ste = (ste - min(ste)) / (max(ste) - min(ste))
    zcr = (zcr - min(zcr)) / (max(zcr) - min(zcr))
    return ste, zcr


def segment_audio(ste, zcr, ste_threshold, zcr_threshold):
    voiced_unvoiced_segments = dict()
    current_segment_start = None
    for i in range(len(ste)):
        if ste[i] > ste_threshold and zcr[i] < zcr_threshold:
            voiced_unvoiced_segments[i] = "green"
            current_segment_start = None
        elif ste[i] <= ste_threshold:
            voiced_unvoiced_segments[i] = "red"
            if current_segment_start is None:
                current_segment_start = i
            if (i - current_segment_start) >= 23\
                    and (i == len(ste) - 1 or (ste[i + 1] > ste_threshold)):
                for j in range(current_segment_start, i + 1):
                    voiced_unvoiced_segments[j] = "blue"
                current_segment_start = None
        else:
            voiced_unvoiced_segments[i] = "red"
            current_segment_start = None
    return voiced_unvoiced_segments


def filter_data(voiced_unvoiced_segments, kernel_size):
    color_to_value = {"red": 1, "green": 0, "blue": 2}
    data_as_numbers = list(map(lambda x: color_to_value[x], voiced_unvoiced_segments.values()))
    temp_list = medfilt(data_as_numbers, kernel_size=kernel_size)
    value_to_color = {1: "red", 0: "green", 2: "blue"}
    numbers_as_data = list(map(lambda x: value_to_color[x], temp_list))
    voiced_unvoiced_segments.update(zip(voiced_unvoiced_segments.keys(), numbers_as_data))
    return voiced_unvoiced_segments


def plot_results(audio_samples, sample_rate, ste, zcr, voiced_unvoiced_segments):
    time = np.arange(len(ste)) * hop_size / sample_rate
    fig, axs = plt.subplots(2, 1, figsize=(25, 15))

    audio_file_name = audio_file_path.split('/')[-1]
    fig.suptitle(audio_file_name, fontsize=16)

    axs[0].plot(time, ste, label='STE', color='yellow')
    axs[0].plot(time, zcr, label='ZCR', color='purple')
    axs[0].set_xticks(np.arange(0, time[-1], 0.1))
    axs[0].set_yticks(np.arange(0, 1.1, 0.1))

    for frame_idx, color in voiced_unvoiced_segments.items():
        if color == "blue":
            axs[0].axvspan(frame_idx * hop_size / sample_rate,
                           (frame_idx + 1) * hop_size / sample_rate,
                           alpha=0.2, color=color)
        else:
            axs[0].axvspan(frame_idx * hop_size / sample_rate,
                           (frame_idx + 1) * hop_size / sample_rate,
                           alpha=0.2, color=color)

    axs[0].legend()

    time = np.arange(len(audio_samples)) / sample_rate
    axs[1].plot(time, audio_samples, color='black')
    for frame_idx, color in voiced_unvoiced_segments.items():
        if color == "blue":
            axs[1].axvspan(frame_idx * hop_size / sample_rate,
                           (frame_idx + 1) * hop_size / sample_rate,
                           alpha=0.2, color=color)
        else:
            axs[1].axvspan(frame_idx * hop_size / sample_rate,
                           (frame_idx + 1) * hop_size / sample_rate,
                           alpha=0.2, color=color)

    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_xticks(np.arange(0, time[-1], 0.1))

    plt.show()


if __name__ == "__main__":
    audio_file_paths = ['./TinHieuKiemThu/studio_M1.wav']
    for audio_file_path in audio_file_paths:
        audio_samples, sample_rate = read_audio(audio_file_path)

        frame_size, hop_size = calculate_frame_and_hop_sizes(sample_rate, frame_time=0.025, hop_time=0.0125)

        ste, zcr = calculate_ste_and_zcr(audio_samples, frame_size, hop_size)

        voiced_unvoiced_segments = segment_audio(ste, zcr, ste_threshold=0.0015, zcr_threshold=0.12)

        voiced_unvoiced_segments = filter_data(voiced_unvoiced_segments, kernel_size=5)

        plot_results(audio_samples, sample_rate, ste, zcr, voiced_unvoiced_segments)
