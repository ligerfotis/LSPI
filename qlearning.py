# a q learning agent on the cartpole environment with the following code:
#
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env.action_space.seed(42)
# Hyperparameters
NUM_EPISODES = 100000
MAX_STEPS = 500
NUM_BINS = 100
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.3

# Initialize Q(s,a) table
Q = np.zeros((NUM_BINS, NUM_BINS, NUM_BINS, NUM_BINS, env.action_space.n))

# Initialize statistics
episode_lengths = np.zeros(NUM_EPISODES)
episode_rewards = np.zeros(NUM_EPISODES)

# Discretize the state space
def discretize(observation):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], np.math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -np.math.radians(50)]
    ratios = [(observation[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(observation))]
    new_obs = [int(round((NUM_BINS - 1) * ratios[i])) for i in range(len(observation))]
    new_obs = [min(NUM_BINS - 1, max(0, new_obs[i])) for i in range(len(observation))]
    return tuple(new_obs)

# Q-learning
for i_episode in range(NUM_EPISODES):
    observation, info = env.reset(seed=42)

    observation = discretize(observation)
    terminated = False
    t = 0
    total_reward = 0
    while not terminated:
        # env.render()
        if np.random.uniform(0, 1) < EPSILON:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[observation])
        next_observation, reward, terminated, truncated, info = env.step(action)
        next_observation = discretize(next_observation)
        # Update Q
        Q[observation][action] += ALPHA * (reward + GAMMA * np.max(Q[next_observation]) - Q[observation][action])
        observation = next_observation
        t += 1
        total_reward += reward
    episode_lengths[i_episode] = t
    episode_rewards[i_episode] = total_reward
    if i_episode % 100 == 0:
        print("Episode: {}, reward: {}, length: {}".format(i_episode, total_reward, t))

env.close()

