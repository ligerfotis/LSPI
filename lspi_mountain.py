import gymnasium as gym
from lspi_modules import LSPI, BlockBasis, RadialBasisFunction, PolynomialBasisFunction, IdentityBasisFunction

# env = gym.make("MountainCar-v0", render_mode="human")
env = gym.make("MountainCar-v0")

action_space = 3
observation_space = 2
env.action_space.seed(42)

observation, info = env.reset(seed=42)

basis_function = BlockBasis(action_space, observation_space)
# basis_function = RadialBasisFunction(num_centers=4, state_dim=4)
# basis_function = PolynomialBasisFunction(2)
# basis_function = IdentityBasisFunction(observation_space, action_space)

lspi = LSPI(basis_function, discount=0.99, action_space_size=action_space)

num_episodes = 10000

for i_episode in range(num_episodes):
    terminated = False
    truncated = False
    observation, info = env.reset()
    reward_sum = 0
    while not terminated or not truncated:
        if i_episode < 1000:
            action = env.action_space.sample()
        else:
            action = lspi.policy(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        # Update the policy
        lspi.update(observation, action, reward, next_observation)

        observation = next_observation
        reward_sum += reward
    if i_episode % 10 == 0:
        print("Episode: {}, reward: {}".format(i_episode, reward_sum))

env.close()
