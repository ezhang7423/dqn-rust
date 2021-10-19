import copy
import gym
import numpy as np
from collections import deque
import torch
import random
import signal
import sys
from src.minimal_network import CNN, rgb2gray, train


def save_experiment(signal, frame):
    print("Saving experiment info...")
    np.save("q_vals.npy", np.array(avg_q_vals))
    torch.save(q, "q-network.torch")
    print("Finished saving!")
    env.close()
    sys.exit(0)


signal.signal(signal.SIGINT, save_experiment)


env = gym.make("BreakoutDeterministic-v4", render_mode="rgb_array")
observation = env.reset()

MILLION = int(1e7)
it = 10
REPLAY_BUFFER = deque(maxlen=MILLION)
EPISODES = MILLION
iters = 0
EPSILON = 1

initial = np.load("image_sample.npy")
q = CNN()
avg_q_vals = []

try:
    print("Beginning training")
    for ep in range(EPISODES):
        print(f"Starting episode {ep}, iterations: {iters}")
        sequence = deque(initial, maxlen=4)
        done = False
        avg_q_val = 0

        while not done:
            if iters <= MILLION:
                EPSILON -= iters / MILLION
            if random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = torch.argmax(q(sequence)).item()

            observation, reward, done, info = env.step(action)
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

            avg_q_val = (avg_q_val * iters + action) / (iters + 1)
            iters += 1

        avg_q_vals.append(avg_q_val)
        observation = env.reset()
        # np.save("image_sample.npy", np.asarray(sequence))

finally:
    env.close()
    save_experiment(None, None)
