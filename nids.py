import torch
import pandas as pd

class PacketDataset(torch.utils.data.Dataset):
    def __init__(self, packet_df, labels_df):
        self.packets = packet_df
        self.labels = labels_df
    def __len__(self):
        return len(self.packets)
    def __getitem__(self, idx):
        return self.packets.iloc[idx].to_numpy(), self.labels.loc[idx, "x"]

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        self.linear = torch.nn.Linear(input_features, output_features)

    def forward(self, input):
        logits = self.linear(input)

        return logits
