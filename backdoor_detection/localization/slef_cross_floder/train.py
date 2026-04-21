import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import ast

# 读取数据
data = []
with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/train_clean.txt', 'r', encoding='utf-8') as f:
    for line in f:
        item = ast.literal_eval(line.strip().strip('"'))
        data.append(item)
# data = data[:900]

with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/train_IBA.txt', 'r', encoding='utf-8') as f:
    for line in f:
        item = ast.literal_eval(line.strip().strip('"'))
        data.append(item)

with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/train_Rick.txt', 'r', encoding='utf-8') as f:
    for line in f:
        item = ast.literal_eval(line.strip().strip('"'))
        data.append(item)

with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/train_BadT2I_pixel_dateset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        item = ast.literal_eval(line.strip().strip('"'))
        data.append(item)

with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/train_BadT2I_style_dateset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        item = ast.literal_eval(line.strip().strip('"'))
        data.append(item)

with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/train_BadT2I_object_dateset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        item = ast.literal_eval(line.strip().strip('"'))
        data.append(item)

with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/train_Evil_dateset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        item = ast.literal_eval(line.strip().strip('"'))
        data.append(item)

with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/train_Villan_dateset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        item = ast.literal_eval(line.strip().strip('"'))
        data.append(item)

for item in data[1914:]:
    item['label'] = 1

# with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/train_BAGM_dateset.txt', 'r', encoding='utf-8') as f:
#     for line in f:
#         item = ast.literal_eval(line.strip().strip('"'))
#         data.append(item)

class NumberSeqDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # 转置为(seq_len, 2)
        values = torch.tensor(self.data[idx]['values'], dtype=torch.float32).T
        label = self.data[idx]['label']
        return values, label

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_seqs = pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor([len(seq) for seq in sequences])
    labels = torch.tensor(labels)
    return padded_seqs, lengths, labels

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=3, hidden_size=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 5)
        self.fc1 = nn.Linear(5, 2)

    def forward(self, x, lengths):
        # x: (batch, seq_len, input_size)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        out = self.fc(hn[-1])
        out = self.fc1(out)
        return out


# data = data[:1115]
dataset = NumberSeqDataset(data)
loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)

model = LSTMClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

best_loss = float('inf')

for epoch in range(20):
    for x, lengths, y in loader:
        logits = model(x, lengths)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 保存loss最低的模型
    # if loss.item() < best_loss:
    if epoch== 19:
        best_loss = loss.item()
        # 保存模型权重
        torch.save(model.state_dict(),
                   '/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/model_v5.pth')
        print(f"模型已保存，当前最低loss: {best_loss:.4f}")





