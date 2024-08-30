import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import serial
import time
from collections import Counter

# 读取Excel文件
file_path = 'data/data.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)

# 统计每个label的数量
label_counts = df.iloc[:, -1].value_counts()

# 打印统计结果
print("Label counts:")
print(label_counts)

# 数据预处理部分
# 提取特征和标签
features = df.iloc[:, :-1]  # 提取所有列除了最后一列作为特征
labels = df.iloc[:, -1]  # 最后一列是标签

# 对标签进行编码
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 特征标准化
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 设置序列长度
sequence_length = 120


# 函数：创建时间序列数据
def create_sequences(data, labels, seq_length):
    sequences = []
    seq_labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        seq_labels.append(labels[i + seq_length - 1])  # 标签对应序列的最后一个时间步
    return np.array(sequences), np.array(seq_labels)


# 构建训练集和测试集的时间序列数据
X_seq, y_seq = create_sequences(features_scaled, labels_encoded, sequence_length)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 创建数据集和数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# RNN模型定义
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # RNN层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# 模型、损失函数和优化器
input_size = features.shape[1]  # 特征数量
hidden_size = 64
output_size = len(label_encoder.classes_)  # 标签类别数量
num_layers = 1

model = RNNModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 模型训练
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs)  # 输入已经是 (batch_size, seq_length, input_size)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
torch.save(model.state_dict(), 'rnn_model.pth')

# 实时数据处理和推理部分
# 初始化串口
ser = serial.Serial('COM4', 115200, timeout=1)


def read_imu_data():
    data = []
    while len(data) < 120:  # 读取120个时间点
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                values = list(map(int, line.split(',')))  # 将字符串转换为整数列表
                if len(values) == 6:  # 确保每行有6个数据
                    data.append(values)
            except ValueError:
                pass  # 跳过不正确的行
    return np.array(data)


def normalize_data(data, min_value=0, max_value=65535):
    return (data - min_value) / (max_value - min_value)


def prepare_data(imu_data):
    imu_tensor = torch.tensor(imu_data, dtype=torch.float32)
    imu_tensor = imu_tensor.unsqueeze(0)  # (1, 120, 6)
    return imu_tensor


# 加载训练好的模型
model = RNNModel(input_size=6, hidden_size=64, output_size=output_size, num_layers=1)
model.load_state_dict(torch.load('rnn_model.pth'))
model.eval()

while True:
    imu_data = read_imu_data()
    imu_data_normalized = normalize_data(imu_data)
    input_tensor = prepare_data(imu_data_normalized)

    # 使用模型进行推理
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_label = torch.max(output, 1)

    # 将预测结果转换为实际标签
    predicted_label_str = label_encoder.inverse_transform(predicted_label.cpu().numpy())
    print(f'Predicted label: {predicted_label_str[0]}')

    time.sleep(1)  # 每秒推理一次，或者根据需要调整时间间隔
