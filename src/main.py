import copy
import threading
import gym
import types
import ale_py
import functools
from gym import wrappers
from gym.envs.classic_control import rendering
import numpy as np
from collections import deque
import torch
import random
import torchvision
import queue

from minimal_network import CNN, rgb2gray, train


env = gym.make("BreakoutDeterministic-v4", render_mode="human")
observation = env.reset()

MILLION = int(1e7)
it = 10
REPLAY_BUFFER = deque(maxlen=MILLION)
EPISODES = MILLION
frames = 0
EPSILON = 1

initial = np.load("../image_sample.npy")
q = CNN()

print("Beginning training")
for ep in range(EPISODES):
    sequence = deque(initial, maxlen=4)
    done = False

    while not done:
        if frames <= MILLION:
            EPSILON -= 4 / MILLION
        if random.random() < EPSILON:
            action = env.action_space.sample()
        else:
            action = torch.argmax(q()).item()

        observation, reward, done, info = env.step(action)
        print(reward)
        previous_sequence = copy.deepcopy(sequence)
        sequence.append(
            rgb2gray(observation)[::2, ::2].astype(np.float32),
        )
        REPLAY_BUFFER.append(
            (
                np.asarray(previous_sequence),
                action,
                reward,
                np.asarray(sequence),
            )
        )  # (state, action, reward, state')
        if len(REPLAY_BUFFER) >= 32:
            train(q, REPLAY_BUFFER)

        frames += 4

    observation = env.reset()
    # np.save("../image_sample.npy", np.asarray(sequence))

env.close()
