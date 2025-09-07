import torch
import pandas as pd

class PacketDataset(torch.utils.data.Dataset):
    def __init__(self, packet_capture_file, labels_file, max_number_of_packets):
        self.packets = pd.read_csv(packet_capture_file, nrows=max_number_of_packets)
        self.labels = pd.read_csv(labels_file, nrows=max_number_of_packets)
    def __len__(self):
        return len(self.packets)
    def __getitem__(self, idx):
        return pd.concat([self.packets.iloc[idx], self.labels.iloc[idx]])

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        number_of_output_features = 1

        return torch.zeros(input.shape[0], number_of_output_features)
