import pandas as pd
import nids
import torch
import sys
from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import ARP, Ether
import scapy.sendrecv

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()

    for X, y in dataloader:
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y).to(device)

        optimizer.zero_grad()
        pred = model(X).to(device)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

def find_arp_request_rate(packet_df):
    source_arp_request_rates = []
    for source, group in packet_df[packet_df["Protocol_ARP"] == 1].groupby(["Source"]):
        arp_request_rate = group.rolling("1.0s", on="Time").count()["Source"].rename("arp_request_rate")
        source_arp_request_rates.append(arp_request_rate)

    if (len(source_arp_request_rates) > 0):
        packet_df = packet_df.drop(["arp_request_rate"], axis="columns", errors="ignore")
        source_arp_request_rate_column = pd.concat(source_arp_request_rates)
        packet_df = packet_df.join(source_arp_request_rate_column)

    return packet_df

def find_tcp_rate(packet_df: pd.DataFrame):
    tcp_rate = packet_df[packet_df["Protocol_TCP"] == 1].rolling("1.0s", on="Time").count()["Protocol_TCP"].rename("tcp_rate")

    return packet_df.drop(["tcp_rate"], axis="columns", errors="ignore").join(tcp_rate)

def packet_handler(pkt: scapy.packet.Packet):
    global captured_packets_df

    source_mac = pkt[Ether].src
    dest_mac = pkt[Ether].dst

    print(f"Source MAC: {source_mac} Destination MAC: {dest_mac}")

    if pkt.haslayer(IP):
        source_ip = pkt[IP].src
        dest_ip = pkt[IP].dst

        print(f"Source IP: {source_ip} Destination IP: {dest_ip} Length: {len(pkt)}")

    packet_time = pd.to_datetime(pkt.time, unit="s")

    packet_df = pd.DataFrame([[packet_time,
                               source_mac,
                               len(pkt),
                               int(pkt.haslayer(ARP)),
                               int(pkt.haslayer(TCP))]], columns=["Time", "Source", "Length", "Protocol_ARP", "Protocol_TCP"])
    captured_packets_df = pd.concat([captured_packets_df, packet_df], ignore_index=True)

    captured_packets_df = find_arp_request_rate(captured_packets_df)
    captured_packets_df = find_tcp_rate(captured_packets_df)
    packet_df = packet_df.drop(["Source", "Time"], axis="columns").reset_index(drop=True)
    packet_df["arp_request_rate"] = captured_packets_df["arp_request_rate"]
    packet_df["tcp_rate"] = captured_packets_df["tcp_rate"]

    print(captured_packets_df)

    X = torch.tensor(packet_df.to_numpy(dtype="float32"), dtype=torch.float32).to(device)
    logits = model(X).to(device)
    softmax_model = torch.nn.Softmax()
    is_attack = bool(softmax_model(logits).argmax(dim=1))

    if is_attack:
        print("Attack detected!")
    else:
        print("Normal traffic")
    print("")

def preprocess(packet_df, label_df):
    packet_df = pd.get_dummies(packet_df, columns=["Protocol"], dtype=float)
    packet_df["Time"] = pd.to_datetime(packet_df["Time"], unit="s")

    features_to_drop = ["No.", "Info", "Destination", "Protocol_0xe812", "Protocol_H1", "Protocol_RTCP", "Protocol_RTSP", "Protocol_SSDP", "Protocol_SSLv2", "Protocol_TLSv1", "Protocol_UDP", "Protocol_DHCPv6", "Protocol_DNS", "Protocol_ICMP", "Protocol_ICMPv6", "Protocol_IGMPv3", "Protocol_LLMNR", "Protocol_MDNS", "Protocol_NBNS", "Protocol_TCP, HiPerConTracer"]
    packet_df = packet_df.drop(features_to_drop, axis="columns")

    packet_df = find_arp_request_rate(packet_df)
    packet_df = find_tcp_rate(packet_df)
    packet_df = packet_df.drop(["Source", "Time"], axis="columns").reset_index(drop=True)

    packet_df = packet_df.fillna(0)

    packet_df["Length"] = (packet_df["Length"] - packet_df["Length"].min()) / (packet_df["Length"].max() - packet_df["Length"].min())

    return nids.PacketDataset(packet_df, label_df)

def train(packet_dataset):
    training_packet_dataset, validation_packet_dataset = torch.utils.data.random_split(packet_dataset, [0.5, 0.5])
    training_dataloader = torch.utils.data.DataLoader(training_packet_dataset, batch_size=64)
    validation_dataloader = torch.utils.data.DataLoader(validation_packet_dataset, batch_size=64, drop_last=True)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    epoch = 1
    for i in range(0, epoch):
        train_loop(training_dataloader, model, loss_fn, optimizer)
        test_loop(validation_dataloader, model)

    torch.save(model.state_dict(), 'model_weights.pth')

def test_loop(dataloader, model):
    model.eval()

    size = len(dataloader.dataset)
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = torch.tensor(X, dtype=torch.float32).to(device)
            y = torch.tensor(y).to(device)

            logits = model(X).to(device)
            softmax_model = torch.nn.Softmax()

            pred = softmax_model(logits)

            correct += (pred.argmax(dim=1) == y).sum().item()

    print(f"Accuracy: {(correct / size) * 100}%")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = nids.NeuralNetwork(input_features=5, output_features=2).to(device)

try:
    model.load_state_dict(torch.load('model_weights.pth', weights_only=True, map_location=torch.device(device)))
except FileNotFoundError:
    print("Model not found.")

captured_packets_df = pd.DataFrame(columns=["Source", "Protocol_ARP", "Protocol_TCP", "Length", "arp_request_rate", "tcp_rate"])

if len(sys.argv) > 1:
    if sys.argv[1] not in ["--train", "--test"]:
        print("Usage:")
        print("")
        print("main.py <--train/test> <training_or_test_dataset.csv> <training_or_test_labels.csv>")
        sys.exit(1)

    packet_capture_filename = sys.argv[2]
    labels_filename = sys.argv[3]
    packet_df = pd.read_csv(packet_capture_filename)
    label_df = pd.read_csv(labels_filename)

    packet_dataset = preprocess(packet_df, label_df)

    if sys.argv[1] == "--train":
        train(packet_dataset)
    elif sys.argv[1] == "--test":
        test_dataloader = torch.utils.data.DataLoader(packet_dataset, batch_size=64)

        model.eval()
        test_loop(test_dataloader, model)
else:
    model.eval()
    print(scapy.sendrecv.sniff(prn=packet_handler))
