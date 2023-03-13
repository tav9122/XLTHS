import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt


# Đọc file audio,trả về số mẫu và tần số lấy mẫu.
def read_audio(audio_file_path):
    audio_samples, sample_rate = sf.read(audio_file_path)
    return audio_samples, sample_rate


# Tính và trả  kích thước của frame và kích thước của bước nhảy dựa trên thời gian của frame và bước nhảy.
def calculate_frame_and_hop_sizes(sample_rate, frame_time, hop_time):
    frame_size = int(frame_time * sample_rate)
    hop_size = int(hop_time * sample_rate)
    return frame_size, hop_size


# Tính toán STE và ZCR, chuẩn hoá về giá trị từ 0 đến 1. Trả về 2 list gồm giá trị STE và ZCR của các frame.
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


# Phân đoạn âm thanh dựa trên giá trị của STE và ZCR.
def segment_audio(ste, zcr, ste_threshold, zcr_threshold):
    segments = dict()
    current_segment_start = None
    for i in range(len(ste)):
        if ste[i] > ste_threshold and zcr[i] < zcr_threshold: # Nếu thoả mãn điều kiện thì tính là voiced
            segments[i] = "green"
            current_segment_start = None
        elif ste[i] <= ste_threshold: # Nếu không thoả mãn điều kiện vì STE quá bé thì tính là unvoiced
            segments[i] = "red"
            if current_segment_start is None:
                current_segment_start = i
            # Tuy nhiên, nếu khoảng unvoiced có độ dài lớn hơn 23 frame, tức là 300ms, thì tính là silence
            if (i - current_segment_start) >= 23\
                    and (i == len(ste) - 1 or (ste[i + 1] > ste_threshold)):
                for j in range(current_segment_start, i + 1):
                    segments[j] = "blue"
                current_segment_start = None
        else:
            segments[i] = "red"
            current_segment_start = None
    return segments


# Lọc nhiễu cho các đoạn âm thanh.
def filter_data(segments, kernel_size):
    color_to_value = {"red": 1, "green": 0, "blue": 2}
    data_as_numbers = list(map(lambda x: color_to_value[x], segments.values()))
    temp_list = medfilt(data_as_numbers, kernel_size=kernel_size)
    value_to_color = {1: "red", 0: "green", 2: "blue"}
    numbers_as_data = list(map(lambda x: value_to_color[x], temp_list))
    segments.update(zip(segments.keys(), numbers_as_data))
    return segments


# Vẽ đồ thị  STE, ZCR và dãy âm thanh đã phân đoạn.
def plot_results(audio_samples, sample_rate, ste, zcr, segments):
    time = np.arange(len(ste)) * hop_size / sample_rate
    fig, axs = plt.subplots(2, 1, figsize=(25, 15))

    audio_file_name = audio_file_path.split('/')[-1]
    fig.suptitle(audio_file_name, fontsize=16)

    axs[0].plot(time, ste, label='STE', color='blue')
    axs[0].plot(time, zcr, label='ZCR', color='red')
    axs[0].set_xticks(np.arange(0, time[-1], 0.1))
    axs[0].set_yticks(np.arange(0, 1.1, 0.1))

    for frame_idx, color in segments.items():
        axs[0].axvspan(frame_idx * hop_size / sample_rate,
                       (frame_idx + 1) * hop_size / sample_rate,
                       alpha=0.2, color=color)

    axs[0].legend()

    time = np.arange(len(audio_samples)) / sample_rate

    axs[1].plot(time, audio_samples, color='black')
    for frame_idx, color in segments.items():
        axs[1].axvspan(frame_idx * hop_size / sample_rate,
                       (frame_idx + 1) * hop_size / sample_rate,
                       alpha=0.2, color=color)

    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_xticks(np.arange(0, time[-1], 0.1))

    plt.show()


def extract_1(ste, sample_rate, hop_size, start_time, end_time):
    start_frame = int(start_time * sample_rate / hop_size)
    end_frame = int(end_time * sample_rate / hop_size)
    return ste[start_frame:end_frame]


def extract_2(filename, data_list, sample_rate, hop_size, segment_type):
    with open(filename, 'r') as f:
        lines = f.readlines()

    lines = [line.split() for line in lines if line.split()[-1] == segment_type]
    new_data_list = []
    for line in lines:
        start_time, end_time = float(line[0]), float(line[1])
        segment = extract_1(data_list, sample_rate, hop_size, start_time, end_time)
        new_data_list.extend(segment)
    return np.array(new_data_list)


def calculate_mean_and_std(feature_values):
    mean = np.mean(feature_values)
    std = np.std(feature_values)
    return mean, std


def find_thresh_hold2(ste, zcr, ste_threshold, zcr_threshold):
    new_ste = []
    new_zcr = []
    for i in range(len(ste)):
        if ste[i] > ste_threshold and zcr[i] < zcr_threshold:
            new_ste.append(ste[i])
            new_zcr.append(zcr[i])
    return new_ste, new_zcr


def find_thresh_hold3(ste, zcr, sample_rate, hop_size):
    v_ste_mean, v_ste_std = calculate_mean_and_std(
        extract_2('./TinHieuKiemThu/studio_M1.lab', ste, sample_rate, hop_size, 'v'))
    v_zcr_mean, v_zcr_std = calculate_mean_and_std(
        extract_2('./TinHieuKiemThu/studio_M1.lab', zcr, sample_rate, hop_size, 'v'))

    best_ste_threshold = None
    best_zcr_threshold = None
    min_diff = float('inf')

    for ste_threshold in np.arange(min(ste), 0.3, 0.001):
        for zcr_threshold in np.arange(min(zcr), 0.3, 0.001):
            segmented_ste, segmented_zcr = find_thresh_hold2(ste, zcr, ste_threshold, zcr_threshold)

            # Check if the segmented arrays are empty
            if len(segmented_ste) == 0 or len(segmented_zcr) == 0:
                continue

            ste_mean, ste_std = calculate_mean_and_std(segmented_ste)
            zcr_mean, zcr_std = calculate_mean_and_std(segmented_zcr)

            diff = abs(ste_mean - v_ste_mean) + abs(zcr_mean - v_zcr_mean) + abs(ste_std - v_ste_std) + abs(zcr_std - v_zcr_std)

            if diff < min_diff:
                min_diff = diff
                best_ste_threshold = ste_threshold
                best_zcr_threshold = zcr_threshold

    return best_zcr_threshold, best_ste_threshold


if __name__ == "__main__":
    audio_file_paths = ['./TinHieuKiemThu/studio_F1.wav', './TinHieuKiemThu/studio_M1.wav',
                        './TinHieuKiemThu/phone_F1.wav','./TinHieuKiemThu/phone_M1.wav']
    for audio_file_path in audio_file_paths:
        audio_samples, sample_rate = read_audio(audio_file_path)

        frame_size, hop_size = calculate_frame_and_hop_sizes(sample_rate, frame_time=0.025, hop_time=0.0125)

        ste, zcr = calculate_ste_and_zcr(audio_samples, frame_size, hop_size)

        segments = segment_audio(ste, zcr, ste_threshold=0.0019, zcr_threshold=0.17)

        segments = filter_data(segments, kernel_size=5)

        plot_results(audio_samples, sample_rate, ste, zcr, segments)
