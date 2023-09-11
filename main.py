import gym
import numpy as np
from sklearn.linear_model import LinearRegression
from stable_baselines3 import PPO


class LSPIPolicy:
    def __init__(self, env, discount_factor=0.9, epsilon=0.1):
        self.action_space_size = env.action_space.n
        self.env = env
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.model = LinearRegression()

    def get_action(self, state):
        if np.random.rand() < self.epsilon or not hasattr(self.model, 'coef_'):
            return self.env.action_space.sample()
        else:
            # Compute the Q-values for all actions
            q_values = []
            for action in range(self.action_space_size):
                feature_vector = self.block_basis(state, action)
                q_values.append(self.model.predict([feature_vector]))
            # Return the action with the highest Q-value
            return np.argmax(q_values)

    def block_basis(self, state, action):
        """
        Create a block basis feature vector for the given state and action.

        Args:
        state: A numpy array representing the state.
        action: An integer representing the action.
        action_space: A list of all possible actions.

        Returns:
        A numpy array representing the block basis feature vector.
        """
        d = len(state)  # Size of the state vector

        # Initialize the feature vector matrix of size d x |A|
        feature_vector = np.zeros((d, self.action_space_size))

        # Set the feature vector to the state vector
        feature_vector[:, action] = state[0]

        # flatten the feature vector across the columns
        feature_vector = feature_vector.flatten("F")

        return feature_vector


    def update(self, transitions):
        X = []
        y = []
        for state, action, reward, next_state, terminated in transitions:
            # Compute the target
            if terminated or not hasattr(self.model, 'coef_'):
                target = reward
            else:
                next_state_feature_vector = self.block_basis(next_state, action)
                target = reward + self.discount_factor * np.max(self.model.predict([next_state_feature_vector]))

            # Compute the feature vector
            feature_vector = self.block_basis(state, action)

            # Add the sample to the training data
            X.append(feature_vector)
            y.append(target)

        # Fit the model
        self.model.fit(X, y)


def evaluate_policy(policy, env, n_episodes=10):
    rewards = []
    for _ in range(n_episodes):
        state = env.reset()[0]
        # action = policy.get_action(state)
        # feature_vector = policy.block_basis(state, action)
        episode_reward = 0
        done = False
        count = 0
        while not done:
            if count > 500:
                done = True
            count += 1
            action = policy.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return np.mean(rewards)


# Create environment
env = gym.make('CartPole-v0')

# LSPI policy
lspi_policy = LSPIPolicy(env)
# Train LSPI policy
print('Training LSPI policy...')
for i_episode in range(1000000):
    state = env.reset()[0]
    transitions = []
    for t in range(500):
        action = lspi_policy.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        transitions.append((state, action, reward, next_state, terminated))
        state = next_state
        if terminated:
            break
    lspi_policy.update(transitions)
    if i_episode % 10000 == 0:
        print('Episode:', i_episode)
        # Evaluate policies
        lspi_reward = evaluate_policy(lspi_policy, env, n_episodes=5)
        print('LSPI Reward:', lspi_reward)
print('Done training LSPI policy')

# PPO policy
# print('Training PPO policy...')
# ppo_policy = PPO('MlpPolicy', env, verbose=0)
# # Train PPO policy
# ppo_policy.learn(total_timesteps=10000)
# print('Done training PPO policy')

print('Evaluating policies...')
# Evaluate policies
lspi_reward = evaluate_policy(lspi_policy, env)
# ppo_reward = evaluate_policy(ppo_policy, env)

print('LSPI Reward:', lspi_reward)
# print('PPO Reward:', ppo_reward)
