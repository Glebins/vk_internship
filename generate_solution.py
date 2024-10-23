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
test_data = pd.read_parquet(path + "test.parquet")

X = test_data.loc[:, ['dates', 'values']]

def remove_nan_from_list(values):
    return [x if not pd.isna(x) else 0 for x in values]

X['values'] = X['values'].apply(remove_nan_from_list)


class TimeSeriesDatasetWithoutLabels(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        values = self.data['values'].iloc[idx]
        return torch.tensor(values, dtype=torch.float32), len(values), idx


def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    sequences, lengths, idx = zip(*batch)

    sequences_padded = pad_sequence(sequences, batch_first=True)
    return sequences_padded, torch.tensor(lengths), idx


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, (hn, cn) = self.lstm(packed_input)
        return self.fc(hn[-1])

solution_dataset = TimeSeriesDatasetWithoutLabels(X)

solution_dataloader = DataLoader(solution_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)

input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
model = LSTMClassifier(input_size, hidden_size, num_layers, output_size)

file_to_save_result = "submission.csv"
file_with_params = "model_params.pth"

model.load_state_dict(torch.load(file_with_params))

model.eval()
all_predictions = []
all_idx = []

with torch.no_grad():
    for batch in solution_dataloader:
        sequences_padded, lengths, idx = batch
        sequences_padded = sequences_padded.unsqueeze(-1)

        outputs = model(sequences_padded, lengths).squeeze(1)
        predictions = torch.sigmoid(outputs)
        all_predictions.extend(predictions.tolist())
        all_idx.extend(idx)

y_pred = pd.Series(all_predictions)

y_pred = y_pred.sort_index()

print(y_pred)
y_pred.to_csv(file_to_save_result, index=True)
