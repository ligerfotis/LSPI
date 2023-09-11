import numpy as np
import gymnasium as gym
import pygame

from lspi_modules import LinearBasisFunction, RadialBasisFunction, LSPI

# Create the environment
env = gym.make("CartPole-v1", render_mode="human")

# Create the basis function
# basis_function = LinearBasisFunction()
# basis_function = RadialBasisFunction(4)

# Initialize LSPI with the basis function and a discount factor
lspi = LSPI(basis_function, discount=0.99)

# Main training loop
num_episodes = 100000
for i_episode in range(num_episodes):
    # Reset the environment and state
    state = env.reset()[0]
    done = False
    reward_sum = 0
    while not done:
        # Select an action
        # action = lspi.policy(state)
        action = env.action_space.sample()

        # Execute the action
        next_state, reward, terminated, truncated, _ = env.step(action)
        # env.render()
        if terminated or truncated:
            done = True
        next_state = None if done else next_state

        # Update the policy
        # lspi.update(state, action, reward, next_state)

        # Update the state
        state = next_state
        reward_sum += reward



    if i_episode % 10000 == 0:
        # # render a game
        # state = env.reset()[0]
        # done = False
        # count = 0
        # while not done:
        #     if count > 500:
        #         break
        #     action = lspi.policy(state)
        #     next_state, reward, done, _, _ = env.step(action)
        #     env.render()
        #     next_state = None if done else next_state
        #     state = next_state
        print("Episode: {}, reward: {}".format(i_episode, reward_sum))
    env.close()
