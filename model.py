import torch
import pandas as pd
from config import MODEL_CONFIG


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()

        self.linear_relu_stack = torch.nn.Sequential(torch.nn.Linear(MODEL_CONFIG["input_features"], MODEL_CONFIG["hidden_dim"]),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Linear(MODEL_CONFIG["hidden_dim"], MODEL_CONFIG["output_features"]))

    def forward(self, input):
        logits = self.linear_relu_stack(input)

        return logits


class PacketDataset(torch.utils.data.Dataset):
    def __init__(self, packet_df: pd.DataFrame, labels_df: pd.DataFrame):
        self.packets = torch.tensor(packet_df.values, dtype=torch.float32)
        self.labels = torch.tensor(labels_df["x"].values, dtype=torch.long)

    def __len__(self):
        return len(self.packets)

    def __getitem__(self, idx):
        return self.packets[idx], self.labels[idx]


class ModelTrainer():
    def __init__(self, model: NeuralNetwork, device: str, model_file: str):
        self.model = model
        self.model_file = model_file
        self.device = device

    def train_loop(self, dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.modules.loss._WeightedLoss,
                   optimizer: torch.optim.Optimizer):
        self.model.train()

        for X, y in dataloader:
            X = X.detach().type(dtype=torch.float32).to(self.device)
            y = y.detach().to(self.device)

            optimizer.zero_grad()
            pred = self.model(X).to(self.device)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

    def train(self, packet_dataset: PacketDataset):
        train_packet_dataset, validation_packet_dataset = torch.utils.data.random_split(packet_dataset, [0.9, 0.1])
        training_dataloader = torch.utils.data.DataLoader(train_packet_dataset,
                                                          batch_size=64,
                                                          num_workers=3,
                                                          pin_memory=True)

        total = len(packet_dataset.labels)
        normal_label_count = torch.sum(packet_dataset.labels == 0)
        attack_label_count = torch.sum(packet_dataset.labels == 1)
        normal_weight = total / normal_label_count
        attack_weight = total / attack_label_count
        weights = torch.tensor([normal_weight, attack_weight], dtype=torch.float32).to(self.device)

        print("Total Labels: ", total)
        print("Normal Labels: ", torch.sum(packet_dataset.labels == 0))
        print("Attack Labels: ", torch.sum(packet_dataset.labels == 1))
        print("Normal Weight: ", total / torch.sum(packet_dataset.labels == 0))
        print("Attack Weight: ", total / torch.sum(packet_dataset.labels == 1))

        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=MODEL_CONFIG["learning_rate"])

        model_tester = ModelTester(self.model, self.device)
        for i in range(0, MODEL_CONFIG["epochs"]):
            self.train_loop(training_dataloader, loss_fn, optimizer)
            model_tester.test(validation_packet_dataset)

        torch.save(self.model.state_dict(), self.model_file)


class ModelTester():
    def __init__(self, model: NeuralNetwork, device: str):
        self.model = model
        self.device = device

    def test(self, packet_dataset: PacketDataset):
        self.model.eval()

        test_dataloader = torch.utils.data.DataLoader(packet_dataset,
                                                      batch_size=64,
                                                      num_workers=3,
                                                      pin_memory=True)

        size = len(test_dataloader.dataset)
        correct = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        with torch.no_grad():
            for X, y in test_dataloader:
                X = X.detach().type(dtype=torch.float32).to(self.device)
                y = y.detach().to(self.device)

                logits = self.model(X).to(self.device)
                softmax_model = torch.nn.Softmax(dim=1)

                pred = softmax_model(logits).argmax(dim=1)

                true_positives += ((pred == 1) & (y == 1)).sum().item()
                true_negatives += ((pred == 0) & (y == 0)).sum().item()
                false_positives += ((pred == 1) & (y == 0)).sum().item()
                false_negatives += ((pred == 0) & (y == 1)).sum().item()
                correct += (pred == y).sum().item()

        labeled_positives = true_positives + false_negatives
        labeled_negatives = true_negatives + false_positives

        sensitivity = true_positives / labeled_positives if labeled_positives != 0 else 0
        specificity = true_negatives / labeled_negatives if labeled_negatives != 0 else 0

        print(f"Accuracy: {(correct / size) * 100}%")
        print(f"Sensitivity: {sensitivity * 100}%")
        print(f"Specificity: {specificity * 100}")
