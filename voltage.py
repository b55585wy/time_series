import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
import neurokit2 as nk

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

# 读取IMU数据文件
file_path = 'wjc.txt'  # 替换为你的IMU数据文件路径
df = pd.read_csv(file_path, header=None)

# 提取电压数据并进行转换
r1 = 47000  # 47kΩ
voltages = df.iloc[:, 0].values
voltages_in_volts = voltages * (2.2 / 4095)
r2_values = (voltages_in_volts * r1) / (3.3 - voltages_in_volts)

# 对电压转换数据进行低通滤波
cutoff = 0.5  # 低通滤波器的截止频率，设为0.5Hz用于呼吸率分析
fs = 100  # 采样率（假设为100Hz）
filtered_r2_values = lowpass_filter(r2_values, cutoff, fs)

# 提取IMU数据并进行滤波
accel_gyro_data = df.iloc[:, 1:7].values  # 提取第二到第七列
filtered_accel_gyro_data = np.zeros_like(accel_gyro_data)
for i in range(6):
    filtered_accel_gyro_data[:, i] = lowpass_filter(accel_gyro_data[:, i], cutoff, fs)

# 计算电阻信号与加速度X轴之间的相位差
analytic_signal_r2 = hilbert(filtered_r2_values)
phase_r2 = np.unwrap(np.angle(analytic_signal_r2))

analytic_signal_x = hilbert(filtered_accel_gyro_data[:, 0])
phase_x = np.unwrap(np.angle(analytic_signal_x))

# 计算相位差
phase_difference = np.abs(phase_r2 - phase_x)
mean_phase_difference = np.mean(phase_difference)

# 绘制相位差
plt.figure(figsize=(10, 6))
plt.plot(phase_difference, label='Phase Difference (R2 - X-axis)')
plt.title('Phase Difference between Resistance and Accelerometer X-axis')
plt.xlabel('Sample')
plt.ylabel('Phase Difference (radians)')
plt.legend()
plt.show()

print(f"Mean Phase Difference between Resistance and Accelerometer X-axis: {mean_phase_difference:.2f} radians")

# 计算电阻信号与加速度Z轴之间的相位差
analytic_signal_z = hilbert(filtered_accel_gyro_data[:, 2])
phase_z = np.unwrap(np.angle(analytic_signal_z))

# 计算相位差
phase_difference_z = np.abs(phase_r2 - phase_z)
mean_phase_difference_z = np.mean(phase_difference_z)

# 绘制相位差
plt.figure(figsize=(10, 6))
plt.plot(phase_difference_z, label='Phase Difference (R2 - Z-axis)')
plt.title('Phase Difference between Resistance and Accelerometer Z-axis')
plt.xlabel('Sample')
plt.ylabel('Phase Difference (radians)')
plt.legend()
plt.show()

print(f"Mean Phase Difference between Resistance and Accelerometer Z-axis: {mean_phase_difference_z:.2f} radians")
