import torch
import nids

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()

        self.linear_relu_stack = torch.nn.Sequential(torch.nn.Linear(input_features, 20),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Linear(20, output_features))

    def forward(self, input):
        logits = self.linear_relu_stack(input)

        return logits

class ModelTrainer():
    def __init__(self, model: NeuralNetwork, device: str, model_file: str):
        self.model = model
        self.model_file = model_file
        self.device = device

    def train_loop(self, dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.modules.loss._WeightedLoss, optimizer: torch.optim.Optimizer):
        self.model.train()

        for X, y in dataloader:
            X = X.detach().type(dtype=torch.float32).to(self.device)
            y = y.detach().to(self.device)

            optimizer.zero_grad()
            pred = self.model(X).to(self.device)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

    def train(self, packet_dataset: nids.PacketDataset):
        training_packet_dataset, validation_packet_dataset = torch.utils.data.random_split(packet_dataset, [0.9, 0.1])
        training_dataloader = torch.utils.data.DataLoader(training_packet_dataset, batch_size=64, num_workers=3, pin_memory=True)
        validation_dataloader = torch.utils.data.DataLoader(validation_packet_dataset, batch_size=64, drop_last=True, num_workers=3, pin_memory=True)

        label_counts = packet_dataset.labels["x"].value_counts()
        total = len(packet_dataset.labels)
        weights = torch.tensor([total / label_counts[0], total / label_counts[1]], dtype=torch.float32).to(self.device)
        print("Total Labels: ", total)
        print("Normal Labels: ", label_counts[0])
        print("Attack Labels: ", label_counts[1])
        print("Normal Weight: ", total / label_counts[0])
        print("Attack Weight: ", total / label_counts[1])

        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4)

        epoch = 10
        for i in range(0, epoch):
            self.train_loop(training_dataloader, loss_fn, optimizer)
            ModelTester(self.model, self.device).test_loop(validation_dataloader)

        torch.save(self.model.state_dict(), self.model_file)

class ModelTester():
    def __init__(self, model: NeuralNetwork, device: str):
        self.model = model
        self.device = device

    def test_loop(self, dataloader):
        self.model.eval()

        size = len(dataloader.dataset)
        correct = 0
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0

        with torch.no_grad():
            for X, y in dataloader:
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
