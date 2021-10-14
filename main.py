import torch
from torch import nn
import torchvision
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

mnist_trainset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

crossEntropy = nn.CrossEntropyLoss()

# torch.manual_seed(42)

BATCH_SIZE = 256

train_loader = torch.utils.data.DataLoader(
    mnist_trainset, batch_size=BATCH_SIZE, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    mnist_trainset, batch_size=BATCH_SIZE, shuffle=True
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5


print("Using {}".format(DEVICE))


def calc_accuracy(
    model, train=False
):  # add train param to calculate accuracy on both train and test
    # Calculate Accuracy
    correct = 0
    total = 0

    d_loader = train_loader if train else test_loader
    # Iterate through test dataset
    for images, labels in d_loader:
        # Load images
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # Forward pass only to get logits/output
        outputs = model(images)

        # Get predictions from the maximum value
        _, predicted = torch.max(outputs.data, 1)

        # Total number of labels
        total += labels.size(0)

        # Total correct predictions
        correct += (predicted == labels).sum()

    accuracy = 100 * correct / total
    return accuracy.item()


class fully_connected(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.block = nn.Sequential(
            # self.input =
            nn.Linear(28 * 28, 784),
            # self.in_relu =
            nn.LeakyReLU(),
            # self.hl1 =
            nn.Linear(784, 400),
            # self.hl_relu =
            nn.LeakyReLU(),
            # self.hl2 =
            nn.Linear(400, 100),
            # self.hl2_relu =
            nn.LeakyReLU(),
            # self.output =
            nn.Linear(100, 10),
            # self.output =
            nn.Softmax(dim=1),
        )

    def forward(self, X):
        return self.block(self.flatten(X))


model = fully_connected().to(DEVICE)

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

print(f"Starting training, initial accuracy: {calc_accuracy(model)}")


print()
for epoch in range(EPOCHS):
    for i, (images, labels) in enumerate(train_loader):
        pred = model(images)
        optimizer.zero_grad()
        loss = crossEntropy(pred, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}, Accuracy: {calc_accuracy(model)}")
