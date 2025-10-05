import pandas as pd
import nids
import torch

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

def test_loop(dataloader, model, loss_fn):
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

packet_df = pd.read_csv("out.csv")
label_df = pd.read_csv("archive/Fuzzing/Fuzzing_labels.csv")

features_to_drop = ["No.", "Time", "Info"]
packet_df = packet_df.drop(features_to_drop, axis="columns")

packet_df = pd.get_dummies(packet_df, columns=["Protocol", "Source", "Destination"], dtype=float)

packet_dataset = nids.PacketDataset(packet_df, label_df)

training_packet_dataset, test_packet_dataset = torch.utils.data.random_split(packet_dataset, [0.5, 0.5])
training_dataloader = torch.utils.data.DataLoader(training_packet_dataset, batch_size=64)
test_dataloader = torch.utils.data.DataLoader(test_packet_dataset, batch_size=64, drop_last=True)

model = nids.NeuralNetwork(input_features=29, output_features=2).to(device)
try:
    torch.load('model_weights.pth', weights_only=True)
    model.eval()
except FileNotFoundError:
    print("Model not found. Creating a new model...")

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    epoch = 1
    for i in range(0, epoch):
        train_loop(training_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

    torch.save(model.state_dict(), 'model_weights.pth')
