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

    # 设置有效频率范围，排除过高的频率（例如，0.05 Hz - 0.5 Hz 的范围）
    valid_range = (0.05, 0.5)
    valid_indices = np.where((fft_freqs >= valid_range[0]) & (fft_freqs <= valid_range[1]))[0]

    # 找到最大频率
    max_freq_index = np.argmax(fft_result[valid_indices])
    max_freq = fft_freqs[valid_indices][max_freq_index]

    # 将最大频率转换为 BPM
    bpm = max_freq * 60
    return bpm

# 多维卡尔曼滤波器
def kalman_filter_3d(data_x, data_y, data_z):
    n = len(data_x)
    x_est = np.zeros((n, 3))  # 状态估计，包含x, y, z三个轴
    P = np.zeros((n, 3))  # 协方差
    Q = 1e-5  # 过程噪声协方差，假设较小
    R = 0.01  # 测量噪声协方差
    x_est[0] = [0, 0, 0]  # 初始状态
    P[0] = [1, 1, 1]  # 初始协方差

    for k in range(1, n):
        # 预测阶段
        x_est[k] = x_est[k-1]
        P[k] = P[k-1] + Q

        # 更新阶段
        for i in range(3):  # 对 x, y, z 轴分别计算
            K = P[k][i] / (P[k][i] + R)  # 卡尔曼增益
            if i == 0:
                x_est[k][i] = x_est[k][i] + K * (data_x[k] - x_est[k][i])
            elif i == 1:
                x_est[k][i] = x_est[k][i] + K * (data_y[k] - x_est[k][i])
            elif i == 2:
                x_est[k][i] = x_est[k][i] + K * (data_z[k] - x_est[k][i])
            P[k][i] = (1 - K) * P[k][i]

    # 返回融合后的 x, y, z 数据的平均值作为单一输出信号
    return np.mean(x_est, axis=1)

# 读取IMU数据文件
file_path = 'zsj.txt'  # 替换为你的IMU数据文件路径
df = pd.read_csv(file_path, header=None)

# 提取xyz三轴加速度数据
acc_x = df.iloc[:, 1].values  # IMU加速度 x 数据（第二列）
acc_y = df.iloc[:, 2].values  # IMU加速度 y 数据（第三列）
acc_z = df.iloc[:, 3].values  # IMU加速度 z 数据（第四列）

# 使用卡尔曼滤波融合xyz三轴信号
fused_data_3d = kalman_filter_3d(acc_x, acc_y, acc_z)

# 设置滤波参数
cutoff = 1  # 低通滤波器的截止频率，设为1Hz用于呼吸率分析
fs = 100  # 假设采样率为100Hz

# 计算融合后信号的呼吸频率
bpm_fused = calculate_breathing_rate(fused_data_3d, cutoff, fs)

# 输出融合后信号的呼吸频率
print(f"Estimated Breathing Rate from Fused Acc Data = {bpm_fused:.2f} BPM")

# 绘制融合后的频谱图
fft_result = np.fft.fft(lowpass_filter(fused_data_3d, cutoff, fs))
fft_freqs = np.fft.fftfreq(len(fused_data_3d), 1 / fs)
positive_freq_indices = np.where(fft_freqs >= 0)
plt.figure(figsize=(10, 6))
plt.plot(fft_freqs[positive_freq_indices], np.abs(fft_result[positive_freq_indices]), label="FFT Amplitude")
plt.title('FFT of Fused IMU Data')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
