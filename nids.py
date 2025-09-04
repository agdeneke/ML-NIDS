import torch
import pandas as pd

class PacketDataset(torch.utils.data.Dataset):
    def __init__(self, packet_capture_file):
        self.packets = pd.read_csv(packet_capture_file)
    def __len__(self):
        return len(self.packets)
    def __getitem__(self, idx):
        return self.packets.iloc[idx]

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
