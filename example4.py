import sys
import gym
from dqn3 import Agent

num_episodes = 10000

env_name = sys.argv[1] if len(sys.argv) > 1 else "Breakout-v0"
env = gym.make(env_name)
env.monitor.start('/tmp/Breakout4-v0', video_callable=lambda count: count % 100 == 0)

agent = Agent(state_size=env.observation_space.shape,
              number_of_actions=env.action_space.n,
              save_name=env_name)

for e in xrange(num_episodes):
    if e < 100 and (e % 100 == 0) == False:
        epsilon = 1
    if 99 < e < 200 and (e % 100 == 0) == False:
        epsilon = .9
    if 199 < e < 300 and (e % 100 == 0) == False:
        epsilon = .8
    if 299 < e < 400 and (e % 100 == 0) == False:
        epsilon = .7
    if 399 < e < 500 and (e % 100 == 0) == False:
        epsilon = .6
    if 499 < e < 600 and (e % 100 == 0) == False:
        epsilon = .5
    if 599 < e < 700 and (e % 100 == 0) == False:
        epsilon = .4
    if  699 < e < 800 and (e % 100 == 0) == False:
        epsilon = .3
    if 799 < e < 900 and (e % 100 == 0) == False:
        epsilon = .2
    if 899 < e < 1000 and (e % 100 == 0) == False:
        epsilon = .1
    if e < 1000 and (e % 100 == 0) == False:
        epsilon = .05
    if e % 100 == 0:
        epsilon = 0
    if e == 0:
        epsilon = 0
        
    observation = env.reset()
    done = False
    agent.new_episode()
    total_cost = 0.0
    total_reward = 0.0
    frame = 0
    while not done:
        frame += 1
        #env.render()
        action, values = agent.act(observation)
        #action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_cost += agent.observe(reward)
        total_reward += reward
    print "total reward", total_reward
    print "mean cost", total_cost/frame

env.monitor.close()
