from __future__ import division, print_function, unicode_literals

# Handle arguments (before slow imports so --help can be fast)
import argparse
parser = argparse.ArgumentParser(
    description="Train a DQN net to play MsMacman.")
parser.add_argument("-n", "--number-steps", type=int, default=1000000,
    help="total number of training steps")
parser.add_argument("-l", "--learn-freq", type=int, default=4,
    help="number of game steps between each training step")
parser.add_argument("-s", "--save-steps", type=int, default=1000,
    help="number of training steps between saving checkpoints")
parser.add_argument("-c", "--copy-steps", type=int, default=1000,
    help="number of training steps between copies of online DQN to target DQN")
parser.add_argument("-r", "--render", action="store_true", default=False,
    help="render the game during training or testing")
parser.add_argument("-j", "--jobid", default="123132",
    help="SLURM job ID")
parser.add_argument("-t", "--test", action="store_true", default=False,
    help="test (no learning and minimal epsilon)")
parser.add_argument("-v", "--verbosity", action="count", default=1,
    help="increase output verbosity")
args = parser.parse_args()

from collections import deque
import gym
import numpy as np
import os
import tensorflow as tf
from util import  wrap_dqn
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import shutil
if not os.path.exists(args.jobid):
  os.makedirs(args.jobid)
shutil.copy(__file__, os.path.join(args.jobid, "code.py"))

env = wrap_dqn(gym.make("PongNoFrameskip-v4"))
done = True  # env needs to be reset

# First let's build the two DQNs (online & target)
input_height = 84
input_width = 84
input_channels = 4
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3 
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 11  # conv3 has 64 maps of 11x10 each
n_hidden = 256
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # 9 discrete actions are available
initializer = tf.contrib.layers.variance_scaling_initializer()

def q_network(X_state, name):
    prev_layer = X_state
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, strides, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides,
                conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(
                prev_layer, filters=n_maps, kernel_size=kernel_size,
                strides=strides, padding=padding, activation=activation,
                kernel_initializer=initializer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs,
                                  kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width,
                                            input_channels])
online_q_values, online_vars = q_network(X_state, name="q_networks/online")
target_q_values, target_vars = q_network(X_state, name="q_networks/target")

# We need an operation to copy the online DQN to the target DQN
copy_ops = [target_var.assign(online_vars[var_name])
            for var_name, target_var in target_vars.items()]
copy_online_to_target = tf.group(*copy_ops)

# Now for the training operations
learning_rate = 1e-4

with tf.variable_scope("train"):
    X_action = tf.placeholder(tf.int32, shape=[None])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    q_value = tf.reduce_sum(online_q_values * tf.one_hot(X_action, n_outputs),
                            axis=1, keep_dims=True)
    error = tf.abs(y - q_value)
    clipped_error = tf.clip_by_value(error, 0.0, 1.0)
    linear_error = 2 * (error - clipped_error)
    loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

    global_step = tf.Variable(0, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Let's implement a simple replay memory
replay_memory_size = 10000
replay_memory = deque([], maxlen=replay_memory_size)

def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3],
           cols[4].reshape(-1, 1))

# And on to the epsilon-greedy policy with decaying epsilon
eps_min = 0.01
eps_max = 1.0 if not args.test else eps_min
exploration_fraction = 0.1
eps_decay_steps = args.number_steps * exploration_fraction

def epsilon_greedy(q_values, step):
    epsilon = max(eps_min, eps_max - (eps_max-eps_min) * step/eps_decay_steps)
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs) # random action
    else:
        return np.argmax(q_values) # optimal action

# TensorFlow - Execution phase
training_start = 10000  # start training after 10,000 game steps
discount_rate = 0.99
batch_size = 32
done = True # env needs to be reset

# We will keep track of the max Q-Value over time and compute the mean per game
loss_val = np.infty
game_length = 0
total_max_q = 0
mean_max_q = 0.0
returnn = return_display = 0.0
returns = []

path = os.path.join(args.jobid, "model")
with tf.Session() as sess:
    if os.path.isfile(path + ".index"):
        saver.restore(sess, path)
    else:
        init.run()
        copy_online_to_target.run()
    for step in range(args.number_steps):
        training_iter = global_step.eval() 
        if done: # game over, start again
            if args.verbosity > 0:
                print("Step {}/{} ({:.1f})% Training iters {}   "
                      "Loss {:5f}    Mean Max-Q {:5f}   Return: {:5f}".format(
                step, args.number_steps, step * 100 / args.number_steps,
                training_iter, loss_val, mean_max_q, return_display))
                sys.stdout.flush()
            state = env.reset()
        if args.render:
            env.render()

        # Online DQN evaluates what to do
        q_values = online_q_values.eval(feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values, step)

        # Online DQN plays
        next_state, reward, done, info = env.step(action)
        returnn += reward

        # Let's memorize what happened
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        if args.test:
            continue

        # Compute statistics for tracking progress (not shown in the book)
        total_max_q += q_values.max()
        game_length += 1
        if done:
            return_display = returnn
            returns.append(return_display)
            returnn = 0.
            mean_max_q = total_max_q / game_length
            total_max_q = 0.0
            game_length = 0

        if step < training_start or step % args.learn_freq != 0:
            continue # only train after warmup period and at regular intervals
        
        # Sample memories and use the target DQN to produce the target Q-Value
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            sample_memories(batch_size))
        next_q_values = target_q_values.eval(
            feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values

        # Train the online DQN
        _, loss_val = sess.run([training_op, loss], feed_dict={
            X_state: X_state_val, X_action: X_action_val, y: y_val})

        # Regularly copy the online DQN to the target DQN
        if step % args.copy_steps == 0:
            copy_online_to_target.run()

        # And save regularly
        if step % args.save_steps == 0:
            saver.save(sess, path)

# Plot
plt.plot(returns)
plt.legend([args.jobid])
plt.savefig(os.path.join(args.jobid, "plot.png"))
np.save(os.path.join(args.jobid, "returns.npy"), np.array(returns))

