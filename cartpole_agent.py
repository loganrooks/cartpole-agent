#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected, dropout
import gym
import os
import sys

n_iterations = 2000
n_max_steps = 5000
n_games_per_update = 10
save_iterations = 10
discount_rate = 0.95

logdir = "./train"
modelname = "CartPole-model.ckpt"


n_inputs = 4
n_hidden = 6
n_outputs = 1
initializer = tf.contrib.layers.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden1 = fully_connected(X, n_hidden, activation_fn=tf.nn.elu, weights_initializer=initializer)
dropout1 = dropout(hidden1)
hidden2 = fully_connected(dropout1, n_hidden, activation_fn=tf.nn.elu, weights_initializer=initializer)
dropout2 = dropout(hidden2)
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
    try:
        saver.restore(sess, "{}/{}".format(logdir, modelname))
    except:
        sess.run(init_op)
    finally:
        for iteration in range(n_iterations):
            all_rewards = []
            all_gradients = []
            for game in range(n_games_per_update):
                current_rewards = []
                current_gradients = []
                obs = env.reset()
                for step in range(n_max_steps):
                    action_val, gradients_val = sess.run([action, gradients], feed_dict = {X: obs.reshape(1, n_inputs)})
                    obs, reward, done, info = env.step(action_val[0][0])
                    current_rewards.append(reward)
                    current_gradients.append(gradients_val)
                    if done:
                        break
                all_rewards.append(current_rewards)
                all_gradients.append(current_gradients)
            if iteration % 100 == 0:
                print("Iteration: {}, Mean Reward = {:.2}".format(iteration, np.concatenate(all_rewards).mean()))

            all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=0.95)
            feed_dict = {}
            for var_index, grad_placeholder in enumerate(gradients_placeholder):
                mean_gradients = np.mean(
                    [reward * all_gradients[game_index][step][var_index]
                    for game_index, rewards in enumerate(all_rewards)
                    for step, reward in enumerate(rewards)], axis=0)
                feed_dict[grad_placeholder] = mean_gradients
            sess.run(training_op, feed_dict=feed_dict)
            if iteration % save_iterations == 0:
                saver.save(sess, "{}/{}".format(logdir, modelname))
                
