import gymnasium as gym
from lspi_modules import LSPI, BlockBasis, RadialBasisFunction, PolynomialBasisFunction

env = gym.make("CartPole-v1")
env.action_space.seed(42)

observation, info = env.reset(seed=42)


basis_function = BlockBasis(2, 4)
# basis_function = RadialBasisFunction(num_centers=4, state_dim=4)
# basis_function = PolynomialBasisFunction(2)

lspi = LSPI(basis_function, discount=0.9)

num_episodes = 10000

for i_episode in range(num_episodes):
    terminated = False
    observation, info = env.reset()
    reward_sum = 0
    while not terminated:
        if i_episode < 1000:
            action = env.action_space.sample()
        else:
            action = lspi.policy(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        # Update the policy
        lspi.update(observation, action, reward, next_observation)

        observation = next_observation
        reward_sum += reward
    if i_episode % 100 == 0:
        print("Episode: {}, reward: {}".format(i_episode, reward_sum))

env.close()