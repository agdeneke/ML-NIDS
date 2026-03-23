import pandas as pd
import nids
import torch
import sys
from scapy.layers.inet import IP
from scapy.layers.l2 import ARP, Ether
import scapy.sendrecv

def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()

    for X, y in dataloader:
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y).to(device)

        pred = model(X).to(device)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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

def packet_handler(pkt: scapy.packet.Packet):
    print(f"Source MAC: {pkt[Ether].src} Destination MAC: {pkt[Ether].dst}")

    if pkt.haslayer(IP):
        source = pkt[IP].src
        dest = pkt[IP].dst

        print(f"Source IP: {source} Destination IP: {dest} Length: {len(pkt)}")

    packet_df = pd.DataFrame([[len(pkt), int(pkt.haslayer(ARP))]], columns=["Length", "Protocol_ARP"])
    #packet_df = pd.get_dummies(packet_df, columns=["Source", "Destination"], dtype=float)

    X = torch.tensor(packet_df.to_numpy(), dtype=torch.float32).to(device)
    logits = model(X).to(device)
    softmax_model = torch.nn.Softmax()
    is_attack = bool(softmax_model(logits).argmax(dim=1))

    if is_attack:
        print("Attack detected!")
    else:
        print("Normal traffic")
    print("")

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = nids.NeuralNetwork(input_features=2, output_features=2).to(device)

try:
    model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
except FileNotFoundError:
    print("Model not found.")

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

    features_to_drop = ["No.", "Time", "Info", "Source", "Destination"]
    packet_df = packet_df.drop(features_to_drop, axis="columns")

    packet_df = pd.get_dummies(packet_df, columns=["Protocol"], dtype=float)
    features_to_drop = ["Protocol_0xe812", "Protocol_H1", "Protocol_RTCP", "Protocol_RTSP", "Protocol_SSDP", "Protocol_SSLv2", "Protocol_TCP", "Protocol_TLSv1", "Protocol_UDP", "Protocol_DHCPv6", "Protocol_DNS", "Protocol_ICMP", "Protocol_ICMPv6", "Protocol_IGMPv3", "Protocol_LLMNR", "Protocol_MDNS", "Protocol_NBNS", "Protocol_TCP, HiPerConTracer"]
    packet_df = packet_df.drop(features_to_drop, axis="columns")

    print("Packet DataFrame:")
    print(packet_df)

    packet_dataset = nids.PacketDataset(packet_df, label_df)

    if sys.argv[1] == "--train":
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
    elif sys.argv[1] == "--test":
        test_dataloader = torch.utils.data.DataLoader(packet_dataset, batch_size=64)

        model.eval()
        test_loop(test_dataloader, model)
else:
    model.eval()
    print(scapy.sendrecv.sniff(prn=packet_handler))
