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
Uncomment the `env.render()` line to see the game while training, however, this is likely to make training slow.

Currently, it assumes that the observation is an image, i.e. a 3d array, which is the case for all Atari games, and other Atari-like environments.

# Purpose
This is meant to be a very simple implementation, to be used as a starter code. I aimed it to be easy-to-comprehend rather than feature-complete.

Pull requests welcome!

# References
- https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

# TODO
- Extend to other environemnts. Currently only works for Atari and Atari-like environments where the observation space is a 3D Box.
