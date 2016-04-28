import sys
import gym
from dqn import Agent

num_episodes = 20

env_name = sys.argv[1]
env = gym.make(env_name)

agent = Agent(state_size=env.observation_space.shape, 
              number_of_actions=env.action_space.n, 
              save_name=env_name)

for e in xrange(num_episodes):
    observation = env.reset()
    done = False
    agent.new_episode()
    while not done:
        env.render()
        action, values = agent.act(observation)
        #action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        agent.observe(reward)
