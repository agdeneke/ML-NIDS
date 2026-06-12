import pandas as pd
import time


def preprocess(packet_df: pd.DataFrame) -> pd.DataFrame:
    packet_df = pd.get_dummies(packet_df, columns=["Protocol"], dtype=float)
    packet_df = packet_df.sort_values("Time")
    packet_df["Time"] = pd.to_datetime(packet_df["Time"], unit="s")

    packet_df["arp_request_rate"] = (
        packet_df.groupby("Source")
        .rolling("1s", on="Time")["Protocol_ARP"]
        .sum()
        .reset_index(drop=True)
    )
    packet_df["tcp_rate"] = (
        packet_df.rolling("1s", on="Time")["Protocol_TCP"]
        .sum()
    )

    features_to_keep = ["Length", "Protocol_ARP", "Protocol_TCP",
                        "arp_request_rate", "tcp_rate"]
    features_to_drop = [col for col in packet_df.columns
                        if col not in features_to_keep]
    packet_df = packet_df.drop(features_to_drop, axis="columns", errors="ignore").reset_index(drop=True)

    packet_df = packet_df.fillna(0)

    packet_df["Length"] = packet_df["Length"] / 1500

    return packet_df


def find_arp_request_rate(source_address: str, packets: list) -> list:
    current_time = time.time()
    packets_within_window_are_arp = [pkt[3] for pkt in packets
                                     if current_time - pkt[0] <= 1.0
                                     and pkt[1] == source_address]

    return packets_within_window_are_arp.count(True)


def find_tcp_request_rate(packets: list) -> list:
    current_time = time.time()
    packets_within_window_are_tcp = [pkt[4] for pkt in packets
                                     if current_time - pkt[0] <= 1.0]

    return packets_within_window_are_tcp.count(True)
