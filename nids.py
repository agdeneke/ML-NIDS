import torch
import pandas as pd
import numpy as np
from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import ARP, Ether
import scapy.sendrecv

class PacketDataset(torch.utils.data.Dataset):
    def __init__(self, packet_df, labels_df):
        self.packets = packet_df
        self.labels = labels_df
    def __len__(self):
        return len(self.packets)
    def __getitem__(self, idx):
        packet = self.packets.iloc[idx].to_numpy(dtype=np.float64)
        packet.setflags(write=True)

        return packet, self.labels.loc[idx, "x"]

class PacketSniffer():
    def __init__(self, prediction_model, device):
        self.captured_packets_df = pd.DataFrame(columns=["Source", "Length", "Protocol_ARP", "Protocol_TCP", "arp_request_rate", "tcp_rate"])
        self.prediction_model = prediction_model
        self.device = device

        print(scapy.sendrecv.sniff(prn=self.packet_handler))

    def packet_handler(self, pkt: scapy.packet.Packet):
        self.prediction_model.eval()

        source_mac = pkt[Ether].src
        dest_mac = pkt[Ether].dst

        if pkt.haslayer(IP):
            source_ip = pkt[IP].src
            dest_ip = pkt[IP].dst

        packet_time = pd.to_datetime(pkt.time, unit="s")

        packet_df = pd.DataFrame([[packet_time,
                                    source_mac,
                                    len(pkt),
                                    int(pkt.haslayer(ARP)),
                                    int(pkt.haslayer(TCP))]], columns=["Time", "Source", "Length", "Protocol_ARP", "Protocol_TCP"])
        self.captured_packets_df = pd.concat([self.captured_packets_df, packet_df], ignore_index=True)
        preprocessed_captured_packets_df = preprocess(self.captured_packets_df)
        packet_df = preprocessed_captured_packets_df.iloc[-1]

        X = torch.tensor(packet_df.to_numpy(dtype="float32"), dtype=torch.float32).to(self.device)
        logits = self.prediction_model(X).to(self.device)
        softmax_model = torch.nn.Softmax(dim=0)
        is_attack = bool(softmax_model(logits).argmax())

        if is_attack:
            print("Attack detected!")
            print(f"Source MAC: {source_mac} Destination MAC: {dest_mac}")
            print(f"Source IP: {source_ip} Destination IP: {dest_ip} Length: {len(pkt)}")

def find_arp_request_rate(packet_df):
    source_arp_request_rate_column = pd.Series().rename("arp_request_rate")
    for source, group in packet_df[packet_df["Protocol_ARP"] == 1].groupby(["Source"]):
        arp_request_rate = group.rolling("1.0s", on="Time").count()["Source"].rename("arp_request_rate")
        source_arp_request_rate_column = pd.concat([source_arp_request_rate_column, arp_request_rate])

    packet_df = packet_df.drop(["arp_request_rate"], axis="columns", errors="ignore")
    packet_df = packet_df.join(source_arp_request_rate_column)

    return packet_df

def find_tcp_rate(packet_df: pd.DataFrame):
    tcp_rate = packet_df[packet_df["Protocol_TCP"] == 1].rolling("1.0s", on="Time").count()["Protocol_TCP"].rename("tcp_rate")

    return packet_df.drop(["tcp_rate"], axis="columns", errors="ignore").join(tcp_rate)

def preprocess(packet_df):
    if "Protocol" in packet_df.columns:
        packet_df = pd.get_dummies(packet_df, columns=["Protocol"], dtype=float)
    packet_df = packet_df.sort_values("Time")
    packet_df["Time"] = pd.to_datetime(packet_df["Time"], unit="s")

    features_to_keep = ["Source", "Time", "Length", "Protocol_ARP", "Protocol_TCP"]
    features_to_drop = [col for col in packet_df.columns if col not in features_to_keep]
    packet_df = packet_df.drop(features_to_drop, axis="columns", errors="ignore")

    packet_df = find_arp_request_rate(packet_df)
    packet_df = find_tcp_rate(packet_df)
    packet_df = packet_df.drop(["Source", "Time"], axis="columns").reset_index(drop=True)

    packet_df = packet_df.fillna(0)

    packet_df["Length"] = packet_df["Length"] / 1500

    return packet_df
