import torch
import pandas as pd

class PacketDataset(torch.utils.data.Dataset):
    def __init__(self, packet_df, labels_df):
        self.packets = packet_df
        self.labels = labels_df
    def __len__(self):
        return len(self.packets)
    def __getitem__(self, idx):
        packet = self.packets.iloc[idx].to_numpy()
        packet.setflags(write=True)

        return packet, self.labels.loc[idx, "x"]

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        self.linear_relu_stack = torch.nn.Sequential(torch.nn.Linear(input_features, 20),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Linear(20, output_features))

    def forward(self, input):
        logits = self.linear_relu_stack(input)

        return logits
