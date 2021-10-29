import traceback
import copy
import gym
import numpy as np
from collections import deque
import torch
import random
import sys
from src.minimal_network import CNN, DEVICE, rgb2gray, train


def save_experiment(signal, frame):
    print("Saving experiment info...")
    np.save("q-vals.npy", np.array(avg_q_vals))
    torch.save(q.state_dict(), "q-network.torch")
    print("Finished saving!")
    env.close()
    sys.exit(0)


env = gym.make("BreakoutDeterministic-v4", render_mode="rgb_array")
observation = env.reset()

MILLION = int(1e7)
it = 10
REPLAY_BUFFER = deque(maxlen=50000)
EPISODES = MILLION
iters = 0
EPSILON = 1

initial = np.load("image_sample.npy")
q = CNN("q-network.torch").to(DEVICE)
avg_q_vals = np.load("q-vals.npy").tolist()

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
                action = torch.argmax(q(torch.FloatTensor(sequence)[None, ...])).item()

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
            if len(REPLAY_BUFFER) >= 3200:
                train(q, REPLAY_BUFFER)

            avg_q_val = (avg_q_val * iters + action) / (iters + 1)
            iters += 1
        print(f"Avg q val: {avg_q_val}")
        avg_q_vals.append(avg_q_val)
        observation = env.reset()
        # np.save("image_sample.npy", np.asarray(sequence))
except Exception as e:
    print(traceback.format_exc())
finally:
    env.close()
    save_experiment(None, None)
