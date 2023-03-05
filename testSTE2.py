import numpy as np
import soundfile as sf


# Load the WAV file
y, sr = sf.read('/home/vu/PycharmProjects/XLTHS/TinHieuHuanLuyen/phone_F2.wav')

# Set the frame length (in samples) and hop length (in samples)

frame_length = int(0.02 * sr)  # 20ms
hop_length = int(0.01 * sr)  # 10ms
print("frame length:",frame_length)
print("hop length:",hop_length)
# Compute the STE and MA of each frame
n_frames = 1 + int(np.floor((len(y) - frame_length) / hop_length))
print("y",len(y))
print(n_frames)

ste = np.zeros(n_frames)
print("ste:",len(ste))
ma = np.zeros(n_frames)
print("ma:",len(ma))
for i in range(n_frames):
    start = i * hop_length
    end = start + frame_length
    frame = y[start:end]
    ste[i] = np.sum(frame ** 2)
    ma[i] = np.mean(ste[max(0, i-9):i+1])

# Set a threshold for detecting silence and speech
threshold = 0.09 * np.mean(ma)

# Initialize the start and end times for each segment
start_times = []
end_times = []

# Detect silence and speech segments
is_speech = False
for i in range(len(ste)):
    if ste[i] > threshold and not is_speech:
        start_times.append(i * hop_length)
        is_speech = True
    elif ste[i] <= threshold and is_speech:
        end_times.append(i * hop_length)
        is_speech = False

# Add an end time for the last speech segment
if is_speech:
    end_times.append(len(ste) * hop_length)

# Compute the duration of each segment (in seconds)
durations = (np.array(end_times) - np.array(start_times)) / sr

# Print the start times, end times, and durations of each segment
for i in range(len(start_times)):
    print(f"Segment {i+1}: start={start_times[i]/sr:.2f}s, end={end_times[i]/sr:.2f}s, duration={durations[i]:.2f}s")
