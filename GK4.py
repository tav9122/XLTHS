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
        zcr.append(np.sum(np.abs(np.diff(np.sign(frame))) > 0))

    ste = (ste - min(ste)) / (max(ste) - min(ste))
    zcr = (zcr - min(zcr)) / (max(zcr) - min(zcr))
    return ste, zcr


def segment_audio(ste, zcr, ste_threshold, zcr_threshold):
    voiced_unvoiced_segments = dict()
    for i in range(len(ste)):
        if ste[i] > ste_threshold and zcr[i] < zcr_threshold:
            voiced_unvoiced_segments[i] = "green"
        else:
            voiced_unvoiced_segments[i] = "red"
    return voiced_unvoiced_segments


def filter_data(voiced_unvoiced_segments):
    color_to_value = {"red": 1, "green": 0}
    data_as_numbers = list(map(lambda x: color_to_value[x], voiced_unvoiced_segments.values()))
    temp_list = medfilt(data_as_numbers, kernel_size=5)
    value_to_color = {1: "red", 0: "green"}
    numbers_as_data = list(map(lambda x: value_to_color[x], temp_list))
    voiced_unvoiced_segments.update(zip(voiced_unvoiced_segments.keys(), numbers_as_data))
    return voiced_unvoiced_segments


def plot_results(audio_samples, sample_rate, ste, zcr, voiced_unvoiced_segments):
    time = np.arange(len(ste)) * hop_size / sample_rate
    fig, axs = plt.subplots(2, 1, figsize=(20, 5))

    audio_file_name = audio_file_path.split('/')[-1]
    fig.suptitle(audio_file_name, fontsize=16)

    axs[0].plot(time, ste, label='STE')
    axs[0].plot(time, zcr, label='ZCR')
    axs[0].set_xticks(np.arange(0, time[-1], 0.1))
    axs[0].set_yticks(np.arange(0, 1.1, 0.1))

    for frame_idx, color in voiced_unvoiced_segments.items():
        axs[0].axvspan(frame_idx * hop_size / sample_rate,
                       (frame_idx + 1) * hop_size / sample_rate,
                       alpha=0.2, color=color)

    axs[0].legend()

    time = np.arange(len(audio_samples)) / sample_rate
    axs[1].plot(time, audio_samples, color='black')
    for frame_idx, color in voiced_unvoiced_segments.items():
        axs[1].axvspan(frame_idx * hop_size / sample_rate,
                       (frame_idx + 1) * hop_size / sample_rate,
                       alpha=0.2, color=color)

    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_xticks(np.arange(0, time[-1], 0.1))

    plt.show()


if __name__ == "__main__":
    audio_file_paths = ['./TinHieuKiemThu/phone_F1.wav']
    for audio_file_path in audio_file_paths:
        audio_samples, sample_rate = read_audio(audio_file_path)

        frame_size, hop_size = calculate_frame_and_hop_sizes(sample_rate, frame_time=0.025, hop_time=0.0125)

        ste, zcr = calculate_ste_and_zcr(audio_samples, frame_size, hop_size)

        voiced_unvoiced_segments = segment_audio(ste, zcr, ste_threshold=0.004, zcr_threshold=0.3)

        voiced_unvoiced_segments = filter_data(voiced_unvoiced_segments)

        plot_results(audio_samples, sample_rate, ste, zcr, voiced_unvoiced_segments)
