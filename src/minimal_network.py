import os
import random
import numpy as np
import torch
from torch import nn


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("Using {}".format(DEVICE))


class CNN(nn.Module):
    def __init__(self, weights_path):

        super().__init__()
        self.block = nn.Sequential(
            # self.input
            nn.Conv2d(4, 16, 8, 4),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            # self.conv2
            nn.Conv2d(16, 32, 4, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            # self.fc
            nn.Flatten(),
            nn.Linear(2816, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 4),
        )
        if os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path))

    def forward(self, X):
        return self.block(X)

    # model = CNN().to(DEVICE)


def q_loss(previous_pred, current_pred, actions, rewards):
    discount = 0.99
    reward_vec = np.zeros((32, 4))
    reward_vec[np.arange(32), actions] = rewards
    bellman = current_pred * discount + torch.FloatTensor(reward_vec).to(DEVICE)
    loss = torch.mean((previous_pred - bellman) ** 2)
    # print(loss)
    return loss


def train(network, buffer):
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-2)
    optimizer.zero_grad()

    print('Starting train loop')
    for iteration in range(100):
        mini_batch = random.sample(buffer, 32)

        previous = torch.FloatTensor([s[0] for s in mini_batch]).to(DEVICE)
        actions = np.array([s[1] for s in mini_batch])
        rewards = [s[2] for s in mini_batch]
        current = torch.FloatTensor([s[3] for s in mini_batch]).to(DEVICE)

        previous_pred = network(previous)
        current_pred = network(current)
        loss = q_loss(previous_pred, current_pred, actions, rewards)
        print('Loss:', loss)
        loss.backward()
        optimizer.step()
