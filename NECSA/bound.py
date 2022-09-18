import gym
import mujoco_py


env = gym.make('Hopper-v3')

env.reset()

high = 0
low = 0

while True:
    obs = env.observation_space.sample()
    high = max(high, max(obs))
    low = min(low, min(obs))
    print(high, low)
