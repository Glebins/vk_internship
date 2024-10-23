import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import torch.optim as optim
from sklearn.model_selection import train_test_split

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

train_dataset = TimeSeriesDataset(train_data, train_labels)

train_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)

input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
model = LSTMClassifier(input_size, hidden_size, num_layers, output_size)

learning_rate = 0.0001
file_to_save_weights = "model_params.pth"
num_epochs = 10
debug_info = 10

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# If there exists pre-trained model:
model.load_state_dict(torch.load('model_params.pth'))

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0
    for i, batch in enumerate(train_dataloader):
        sequences_padded, lengths, labels = batch
        sequences_padded = sequences_padded.unsqueeze(-1)

        optimizer.zero_grad()

        outputs = model(sequences_padded, lengths).squeeze(1)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        if debug_info and debug_info % 10 == 0:
            print(i)

    print(f'Epoch {epoch + 1} out of {num_epochs}, Loss: {epoch_loss / len(train_dataloader)}')
    torch.save(model.state_dict(), file_to_save_weights)
