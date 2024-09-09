import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# 低通滤波器设计
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# 快速傅里叶变换计算呼吸频率
def calculate_breathing_rate(data, cutoff, fs):
    # 低通滤波
    filtered_data = lowpass_filter(data, cutoff, fs)

    # 使用FFT计算频谱
    n = len(filtered_data)  # 数据点数量
    T = 1 / fs  # 采样周期

    fft_result = np.fft.fft(filtered_data)
    fft_freqs = np.fft.fftfreq(n, T)  # 频率轴

    # 取正频率部分
    positive_freq_indices = np.where(fft_freqs >= 0)
    fft_result = np.abs(fft_result[positive_freq_indices])
    fft_freqs = fft_freqs[positive_freq_indices]

    # 设置有效频率范围，排除过高的频率（例如，0.1 Hz - 0.5 Hz 的范围）
    valid_range = (0.05, 0.5)
    valid_indices = np.where((fft_freqs >= valid_range[0]) & (fft_freqs <= valid_range[1]))[0]

    # 找到最大频率
    max_freq_index = np.argmax(fft_result[valid_indices])
    max_freq = fft_freqs[valid_indices][max_freq_index]

    # 将最大频率转换为 BPM
    bpm = max_freq * 60
    return bpm

# 读取IMU数据文件
file_path = 'zsj.txt'  # 替换为你的IMU数据文件路径
df = pd.read_csv(file_path, header=None)

# 提取数据列（压阻薄膜、加速度、角速度）
r2_values = df.iloc[:, 0].values  # 压阻薄膜数据
acc_x = df.iloc[:, 1].values  # IMU加速度 x
acc_y = df.iloc[:, 2].values  # IMU加速度 y
acc_z = df.iloc[:, 3].values  # IMU加速度 z
gyro_x = df.iloc[:, 4].values  # IMU角速度 x
gyro_y = df.iloc[:, 5].values  # IMU角速度 y
gyro_z = df.iloc[:, 6].values  # IMU角速度 z

# 设置滤波参数
cutoff = 1  # 低通滤波器的截止频率，设为0.2Hz用于呼吸率分析
fs = 100  # 假设采样率为100Hz

# 分别计算各列的呼吸频率
bpm_r2 = calculate_breathing_rate(r2_values, cutoff, fs)
bpm_acc_x = calculate_breathing_rate(acc_x, cutoff, fs)
bpm_acc_y = calculate_breathing_rate(acc_y, cutoff, fs)
bpm_acc_z = calculate_breathing_rate(acc_z, cutoff, fs)
bpm_gyro_x = calculate_breathing_rate(gyro_x, cutoff, fs)
bpm_gyro_y = calculate_breathing_rate(gyro_y, cutoff, fs)
bpm_gyro_z = calculate_breathing_rate(gyro_z, cutoff, fs)

# 输出各列的呼吸频率
print(f"Estimated Breathing Rate from Resistor Film = {bpm_r2:.2f} BPM")
print(f"Estimated Breathing Rate from IMU Acc X = {bpm_acc_x:.2f} BPM")
print(f"Estimated Breathing Rate from IMU Acc Y = {bpm_acc_y:.2f} BPM")
print(f"Estimated Breathing Rate from IMU Acc Z = {bpm_acc_z:.2f} BPM")
print(f"Estimated Breathing Rate from IMU Gyro X = {bpm_gyro_x:.2f} BPM")
print(f"Estimated Breathing Rate from IMU Gyro Y = {bpm_gyro_y:.2f} BPM")
print(f"Estimated Breathing Rate from IMU Gyro Z = {bpm_gyro_z:.2f} BPM")

# 如果需要可以绘制其中某列的频谱图，例如压阻薄膜数据的频谱图
fft_result = np.fft.fft(lowpass_filter(r2_values, cutoff, fs))
fft_freqs = np.fft.fftfreq(len(r2_values), 1 / fs)
positive_freq_indices = np.where(fft_freqs >= 0)
plt.figure(figsize=(10, 6))
plt.plot(fft_freqs[positive_freq_indices], np.abs(fft_result[positive_freq_indices]), label="FFT Amplitude")
plt.title('FFT of Resistor Film Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
