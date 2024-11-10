from collections import defaultdict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from ale_py import ALEInterface
from matplotlib.patches import Patch
from tqdm import tqdm

# env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
env = gym.make("ALE/Breakout-v5")

obs, info = env.reset()


class BreakoutAgent:
    def __init__(
        self,
        env,
        learning_rate,
        start_epsilon,
        epsilon_decay,
        final_epsilon,
        discount_factor,
    ):
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = start_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, env, obs):
        if np.random.random() < self.epsilon:
            return env.action_space.sample()
        else:
            obs_tuple = tuple(obs.flatten())
            return int(np.argmax(self.q_values[obs_tuple]))

    def update(
        self,
        obs,
        action,
        reward,
        terminated,
        next_obs,
    ):
        obs_tuple = tuple(obs.flatten())
        next_obs_tuple = tuple(next_obs.flatten())

        future_q_value = (not terminated) * np.max(self.q_values[next_obs_tuple])
        temporal_difference = (
            reward
            + self.discount_factor * future_q_value
            - self.q_values[obs_tuple][action]
        )

        self.q_values[obs_tuple][action] = (
            self.q_values[obs_tuple][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


learning_rate = 0.01
episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (episodes / 2)
final_epsilon = 0.1
discount_factor = 0.95
training_period = 1000
agent = BreakoutAgent(
    env=env,
    learning_rate=learning_rate,
    start_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
    discount_factor=discount_factor,
)
"""env = gym.wrappers.RecordVideo(
    env,
    video_folder="breakout_video",
    name_prefix="training",
    episode_trigger=lambda x: x % training_period == 0,
)"""
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=episodes)
for episode in tqdm(range(episodes)):
    obs, info = env.reset()
    done = False
    while not done:
        action = agent.get_action(env, obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        agent.update(obs, action, reward, terminated, next_obs)

        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()
