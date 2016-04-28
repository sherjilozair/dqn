# dqn
This is a very basic DQN (with experience replay) implementation, which uses OpenAI's gym environment and Keras/Theano neural networks. 

# Requirements
- gym
- keras
- theano
- numpy

and all their dependencies.

# Usage
To run, `python example.py <env_name>`. It runs `MsPacman-v0` if no env is specified.
Uncommend the `env.render()` line to see the game while training, however this is likely to make training slow.

# Purpose
This is meant to be a very simple implementation, to be used as a started code. I aimed it to be easy-to-comprehend rather than feature-complete.

Pull requests welcome!
