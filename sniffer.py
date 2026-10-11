from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import ARP, Ether
import scapy.sendrecv
import torch
import features
import model
import threading


class PacketSniffer:
    def __init__(self, prediction_model: model.NeuralNetwork, device: str,
                 capture_file: str | None = None):
        self.captured_packets = []
        self.lock = threading.Lock()
        self.prediction_model = prediction_model
        self.device = device

        scapy.sendrecv.AsyncSniffer(prn=self.packet_handler, offline=capture_file).start()

    def snapshot(self) -> list:
        with self.lock:
            return list(self.captured_packets)

    def packet_handler(self, pkt: scapy.packet.Packet):
        source_mac = pkt[Ether].src
        dest_mac = pkt[Ether].dst
        source_ip = dest_ip = None

        if pkt.haslayer(IP):
            source_ip = pkt[IP].src
            dest_ip = pkt[IP].dst

        packet = [pkt.time, source_mac, len(pkt) / 1500,
                  int(pkt.haslayer(ARP)), int(pkt.haslayer(TCP))]

        window = self.captured_packets + [packet]
        packet.append(features.find_arp_request_rate(source_mac, window))
        packet.append(features.find_tcp_request_rate(window))

        input_packet = packet[2:7]

        X = torch.tensor(input_packet, dtype=torch.float32).to(self.device)
        logits = self.prediction_model(X).to(self.device)
        softmax_model = torch.nn.Softmax(dim=0)
        is_attack = bool(softmax_model(logits).argmax())

        packet.append(is_attack)

        with self.lock:
            self.captured_packets.append(packet)

        if is_attack:
            print("Attack detected!")
            print(f"Source MAC: {source_mac} Destination MAC: {dest_mac}")
            print(f"Source IP: {source_ip} Destination IP: {dest_ip} Length: {len(pkt)}")
