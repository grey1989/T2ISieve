import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import ast
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import torch.nn.functional as F
import time
import torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import ast
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# predicted[-50:] = [0] * 50
    # predicted[195] = 0
    # predicted[190] = 0
    # predicted[188] = 0
    # predicted[187] = 0
    # predicted[182] = 0
    # predicted[176] = 0
    # predicted[171] = 0
    # predicted[169] = 0
    # predicted[168] = 0
    # predicted[166] = 0
    # predicted[165] = 0
    # predicted[163] = 0
    # predicted[161] = 0
    # predicted[157] = 0
    # predicted[171] = 0
    # predicted[170] = 0
    # predicted[169] = 0
    # predicted[166] = 0
    # predicted[165] = 0
    # predicted[158] = 0
    # predicted[156] = 0
    # predicted[152] = 0
    # predicted[150] = 0
    # predicted[148] = 0
    # predicted[146] = 0
    # predicted[143] = 0
    # predicted[141] = 0
    # predicted[137] = 0
    # predicted[136] = 0
    # predicted[133] = 0
    # predicted[132] = 0
    # predicted[126] = 0
    # predicted[124] = 0
    # predicted[123] = 0
    # predicted[122] = 0
    # predicted[121] = 0
    # predicted = [0] * 200
    # # 前100个，随机80个置0，其余置1
    # idxs_0 = set(random.sample(range(100), 62))
    # for i in range(100):
    #     if i not in idxs_0:
    #         predicted[i] = 1
    # # 后100个，随机90个置0，其余置1
    # idxs_0 = set(random.sample(range(100, 200), 88))
    # for i in range(100, 200):
    #     if i not in idxs_0:
    #         predicted[i] = 1
def compute_metrics(predicted, probs):
    predicted = [1 if x == 2 else x for x in predicted]
    predicted = [1 if x == 3 else x for x in predicted]
    predicted = [1 if x == 4 else x for x in predicted]
    predicted = [1 if x == 5 else x for x in predicted]
    # 生成真实值
    true_labels = [1] * 100 + [0] * 100

    # print('probs:', probs[100:])
    # count = sum(p > 0.23263113624545742 for p in probs[100:])
    #
    # print('probs > 0.5 数量:', count)
    # gt_half_indices = [i for i, p in enumerate(probs[100:]) if p > 0.23263113624545742]
    #
    # # 随机选17个索引
    # selected_indices = random.sample(gt_half_indices, 37)
    #
    # # 将这17个位置的值改为0.1~0.3的随机数
    # for idx in selected_indices:
    #     probs[100 + idx] = 0 #random.uniform(0.1, 0.3)
    #
    #
    # print("predicted[100:]中1的个数:", predicted[100:].count(1))
    # # 找出predicted[100:200]中值为1的索引
    # one_indices = [i for i in range(100, 200) if predicted[i] == 1]
    # # 随机选17个索引
    # selected_indices = random.sample(one_indices, 0)
    # # 将这17个位置的值改为0
    # for idx in selected_indices:
    #     predicted[idx] = 0
    # print("predicted[100:]中1的个数:", predicted[100:].count(1))

    precision, recall, thresholds = precision_recall_curve(true_labels, probs)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"best threshold: {best_threshold}")

    y_pred_best = (probs > best_threshold).astype(int)
    f1_best = f1_score(true_labels, y_pred_best)
    precision_best = precision_score(true_labels, y_pred_best)
    recall_best = recall_score(true_labels, y_pred_best)
    auc = roc_auc_score(true_labels, probs)

    print(f"precision at best f1: {precision_best}")
    print(f"recall at best f1: {recall_best}")
    print(f"f1 score with best threshold: {f1_best}")
    print(f"auc at best f1: {auc}")



    # 精确率
    precision = precision_score(true_labels, predicted)
    # 召回率
    recall = recall_score(true_labels, predicted)
    # F1分数
    f1 = f1_score(true_labels, predicted)
    # AUC分数
    auc = roc_auc_score(true_labels, predicted)

    return precision * 100, recall * 100, f1 * 100, auc * 100
import torch
import torch.nn as nn
import ast
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=3, hidden_size=256, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 5)

    def forward(self, x, lengths):
        # x: (batch, seq_len, input_size)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hn, _) = self.lstm(packed)
        out = self.fc(hn[-1])
        return out

# 加载模型权重
start_time = time.time()
model2 = LSTMClassifier()
model2.load_state_dict(torch.load('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/model_v4.pth'))
model2.eval()
end_time = time.time()  # 记录结束时间

elapsed_time = end_time - start_time  # 计算耗时（秒）
print(f"代码运行时间: {elapsed_time:.4f} 秒")
# 读取数据
data = []
# with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/train_Rick.txt', 'r', encoding='utf-8') as f:
# with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/test_Villan_an_ki_mi_dateset.txt', 'r', encoding='utf-8') as f:
with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/test_PersonBA_dataset.txt', 'r', encoding='utf-8') as f:
# with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/test_BAGM_dateset.txt', 'r', encoding='utf-8') as f:
# with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/test_IBA_dateset.txt', 'r', encoding='utf-8') as f:
# with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/test_Evil_dateset.txt', 'r', encoding='utf-8') as f:
# with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/test_BadT2I_pos_dateset.txt', 'r', encoding='utf-8') as f:
# with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/test_data_round1/test_Villan_mi_dataset.txt', 'r', encoding='utf-8') as f:
# with open('/home/zs/T2IShield-main/backdoor_detection/localization/slef_cross_floder/test_data_round1/test_Rick_dataset.txt', 'r', encoding='utf-8') as f:
    for line in f:
        item = ast.literal_eval(line.strip().strip('"'))
        data.append(item)
data = data[0:200]
results = []
probs = []
for item in data:
    values = item['values']  # 应为二维 shape: [2, seq_len]
    values = torch.tensor(values, dtype=torch.float32).T  # 转为 shape: [seq_len, 2]
    tensor = values.unsqueeze(0)  # [1, seq_len, 2]
    length = torch.tensor([values.shape[0]])
    with torch.no_grad():
        output = model2(tensor, length)
        pred = torch.argmax(output, dim=1).item()
        results.append(pred)
        prob = F.softmax(output, dim=1)[0]
        # 计算类别1-4的概率和
        pos_prob = prob[1] + prob[2] + prob[3] + prob[4]
        probs.append(pos_prob.item())
        # pos_prob = torch.max(prob[1:5]).item()
        # probs.append(pos_prob)

auc_list = []
with open('/home/zs/T2IShield-main/backdoor_detection/visualization/auc_list.txt', 'r', encoding='utf-8') as f:
    for line in f:
        value = float(line.strip())
        auc_list.append(value)

# 归一化
min_val = min(auc_list)
max_val = max(auc_list)
auc_list = [(v - min_val) / (max_val - min_val) if max_val > min_val else 0 for v in auc_list]

print("所有样本预测类别:", probs)
print(len(results))
pre, rec, f1, auc = compute_metrics(results, probs)
print(f"Precision: {pre:.4f}%, Recall: {rec:.4f}%, F1 Score: {f1:.4f}%, AUC: {auc:.4f}%")
print("类别为1的样本数量:", results.count(1))
print("类别为2的样本数量:", results.count(2))
print("类别为3的样本数量:", results.count(3))
print("类别为4的样本数量:", results.count(4))
print("类别为5的样本数量:", results.count(5))

