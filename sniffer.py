from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import ARP, Ether
import scapy.sendrecv
import torch
import features
import model

class PacketSniffer:
    def __init__(self, prediction_model: model.NeuralNetwork, device: str, capture_file: str | None = None):
        self.captured_packets = []
        self.prediction_model = prediction_model
        self.device = device

        scapy.sendrecv.AsyncSniffer(prn=self.packet_handler, offline=capture_file).start()

    def packet_handler(self, pkt: scapy.packet.Packet):
        source_mac = pkt[Ether].src
        dest_mac = pkt[Ether].dst

        if pkt.haslayer(IP):
            source_ip = pkt[IP].src
            dest_ip = pkt[IP].dst

        packet = [pkt.time, source_mac, len(pkt) / 1500, int(pkt.haslayer(ARP)), int(pkt.haslayer(TCP))]
        self.captured_packets.append(packet)

        packet.append(features.find_arp_request_rate(source_mac, self.captured_packets))
        packet.append(features.find_tcp_request_rate(self.captured_packets))

        input_packet = packet[2:7]

        X = torch.tensor(input_packet, dtype=torch.float32).to(self.device)
        logits = self.prediction_model(X).to(self.device)
        softmax_model = torch.nn.Softmax(dim=0)
        is_attack = bool(softmax_model(logits).argmax())

        if is_attack:
            print("Attack detected!")
            print(f"Source MAC: {source_mac} Destination MAC: {dest_mac}")
            print(f"Source IP: {source_ip} Destination IP: {dest_ip} Length: {len(pkt)}")
