#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 23:55:27 2018

@author: loganrooks
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected, dropout
import gym

n_games = 100
n_steps_per_game = 10000
n_inputs = 4

logdir = "./train"
modelname = "CartPole-model.ckpt"

gym.envs.register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 5000},
    reward_threshold=4750.0,
)


n_inputs = 4
n_hidden = 6
n_outputs = 1
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
keep_prob = tf.placeholder(tf.float32)

hidden1 = fully_connected(X, n_hidden, activation_fn=tf.nn.elu, weights_initializer=initializer)
dropout1 = dropout(hidden1, keep_prob=keep_prob)
hidden2 = fully_connected(dropout1, n_hidden, activation_fn=tf.nn.elu, weights_initializer=initializer)
dropout2 = dropout(hidden2, keep_prob=keep_prob)
logits = fully_connected(dropout2, n_outputs, activation_fn=None, weights_initializer=initializer)
outputs = tf.nn.sigmoid(logits)

p_left_and_right = tf.concat(axis=1, values=[outputs, 1-outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

y = 1.0 - tf.to_float(action)


learning_rate = 0.01

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits( \
                        labels=y, logits=logits)

optimizer = tf.train.AdamOptimizer(learning_rate)

grads_and_vars = optimizer.compute_gradients(cross_entropy)

gradients = [grad for grad, variable in grads_and_vars]

gradients_placeholder = []
grads_and_vars_feed = []

for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradients_placeholder.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
    
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init_op = tf.global_variables_initializer()

saver = tf.train.Saver()

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    normalized_discounted_rewards = [(discounted_rewards - reward_mean) / reward_std \
                                     for discounted_rewards in all_discounted_rewards]
    return normalized_discounted_rewards

env = gym.make('CartPole-v0')



with tf.Session() as sess:
    saver.restore(sess, "{}/{}".format(logdir, modelname))
    for game in range(n_games):
        obs = env.reset()
        for step in range(n_steps_per_game):
            env.render()
            action_val = sess.run(action, feed_dict = {X: obs.reshape(1, n_inputs), keep_prob:1.0})
            obs, reward, done, info = env.step(action_val[0][0])
            if done:
                break