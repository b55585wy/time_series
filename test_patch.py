import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import serial
import time

# 数据预处理部分
file_path = 'data/data.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
label_counts = df.iloc[:, -1].value_counts()

features = df.iloc[:, :-1]
labels = df.iloc[:, -1]
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

sequence_length = 10
def create_sequences(X, y, seq_length):
    X_seq = []
    y_seq = []
    for i in range(len(X) - seq_length + 1):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length - 1])
    return X_seq, y_seq

X_seq, y_seq = create_sequences(features_scaled, labels_encoded, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# RNN模型定义与训练部分
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        _, hn = self.rnn(x, h0)
        out = self.fc(hn[-1])
        return out

input_size = features.shape[1]
hidden_size = 64
output_size = len(label_encoder.classes_)
num_epochs = 50
batch_size = 32

model = RNNModel(input_size, hidden_size, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train).long())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 实时数据处理与推理部分
ser = serial.Serial('COM4', 115200, timeout=1)

def read_imu_data():
    imu_data = ser.readline().decode().strip().split(',')
    return [float(x) for x in imu_data]

model.load_state_dict(torch.load('rnn_model.pth'))
model.eval()
model.to(device)

while True:
    imu_data = read_imu_data()
    input_tensor = torch.Tensor([imu_data]).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_label = torch.max(output, 1)
    predicted_label_str = label_encoder.inverse_transform([predicted_label.item()])
    print(f'Predicted label: {predicted_label_str[0]}')
    time.sleep(1)
