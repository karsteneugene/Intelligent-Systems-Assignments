import gym
import random
import numpy as np

env = gym.make('FrozenLake-v1', is_slippery=False)
env.reset()

qtable = np.zeros((env.observation_space.n, env.action_space.n))

episodes = 1000
lr = 0.5
discount = 0.9
epsilon = 1.0
epsilon_decay = 0.001

outcomes = []

print('Q-table before training:')
print(qtable)

for _ in range(episodes):
    state = env.reset()
    done = False

    outcomes.append("Failure")

    while not done:
        rnd = np.random.random()

        if rnd < epsilon:
            action = env.action_space.sample()
        else:
            if type(state) == int:
                pass
            else:
                state = state[0]
            action = np.argmax(qtable[state])

        new_state, reward, done, _, info = env.step(action)

        if type(state) == int:
            pass
        else:
            state = state[0]

        qtable[state, action] = qtable[state, action] + lr * (reward + discount * np.max(qtable[new_state]) - qtable[state, action])

        state = new_state

        if reward:
            outcomes[-1] = "Success"

    epsilon = max(epsilon - epsilon_decay, 0)

print()
print('===========================================')
print('Q-table after training:')
print(qtable)

episodes = 100
success = 0

for _ in range(100):
    state = env.reset()

    if type(state) == int:
        pass
    else:
        state = state[0]

    done = False
    
    while not done:
        action = np.argmax(qtable[state])

        new_state, reward, done, _, info = env.step(action)

        state = new_state

        success += reward

# Let's check our success rate!
print (f"Success rate = {success/episodes*100}%")