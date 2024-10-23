import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import *

import pandas as pd

pd.set_option('display.max_columns', 100)

path = "/"
train_data = pd.read_parquet(path + "train.parquet")

X = train_data.loc[:, ['dates', 'values']]
y = train_data['label']

def remove_nan_from_list(values):
    return [x if not pd.isna(x) else 0 for x in values]

X['values'] = X['values'].apply(remove_nan_from_list)


class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        values, label = self.data['values'].iloc[idx], self.labels.iloc[idx]
        return torch.tensor(values, dtype=torch.float32), len(values), label


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, lengths, labels = zip(*batch)

    sequences_padded = pad_sequence(sequences, batch_first=True)
    labels = torch.tensor(labels, dtype=torch.float32)

    return sequences_padded, torch.tensor(lengths), labels


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, (hn, cn) = self.lstm(packed_input)
        return self.fc(hn[-1])

train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

test_dataset = TimeSeriesDataset(test_data, test_labels)

test_dataloader = DataLoader(test_dataset, batch_size=128, collate_fn=collate_fn)

input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
model = LSTMClassifier(input_size, hidden_size, num_layers, output_size)

learning_rate = 0.001
file_with_params = "model_params.pth"
num_epochs = 10
debug_info = 10

model.load_state_dict(torch.load(file_with_params))

model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for batch in test_dataloader:
        sequences_padded, lengths, labels = batch
        sequences_padded = sequences_padded.unsqueeze(-1)

        outputs = model(sequences_padded, lengths).squeeze(1)
        predictions = torch.sigmoid(outputs)
        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())

y_pred = pd.Series((pd.Series(all_predictions) > 0.5) * 1)
y_true = pd.Series(all_labels)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, all_predictions)

print(accuracy, precision, recall, f1, roc_auc)

# Got value:
# 0.8624375 0.7742742742742743 0.704302299112224 0.7376326141375611 0.9178888925693816
# These results are better than xgboost's ones

