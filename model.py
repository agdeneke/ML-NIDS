import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()

        self.linear_relu_stack = torch.nn.Sequential(torch.nn.Linear(input_features, 20),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Linear(20, output_features))

    def forward(self, input):
        logits = self.linear_relu_stack(input)

        return logits

class ModelTrainer():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def train_loop(self, dataloader, loss_fn, optimizer):
        self.model.train()

        for X, y in dataloader:
            X = X.detach().type(dtype=torch.float32).to(self.device)
            y = y.detach().to(self.device)

            optimizer.zero_grad()
            pred = self.model(X).to(self.device)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()

    def train(self, packet_dataset):
        training_packet_dataset, validation_packet_dataset = torch.utils.data.random_split(packet_dataset, [0.5, 0.5])
        training_dataloader = torch.utils.data.DataLoader(training_packet_dataset, batch_size=64)
        validation_dataloader = torch.utils.data.DataLoader(validation_packet_dataset, batch_size=64, drop_last=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4)

        epoch = 1
        for i in range(0, epoch):
            self.train_loop(training_dataloader, loss_fn, optimizer)
            ModelTester(self.model, self.device).test_loop(validation_dataloader)

        torch.save(self.model.state_dict(), 'model_weights.pth')

class ModelTester():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def test_loop(self, dataloader):
        self.model.eval()

        size = len(dataloader.dataset)
        correct = 0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.detach().type(dtype=torch.float32).to(self.device)
                y = y.detach().to(self.device)

                logits = self.model(X).to(self.device)
                softmax_model = torch.nn.Softmax(dim=1)

                pred = softmax_model(logits)

                correct += (pred.argmax(dim=1) == y).sum().item()

        print(f"Accuracy: {(correct / size) * 100}%")
