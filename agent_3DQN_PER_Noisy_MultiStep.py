import pickle
import os
path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(path)
import random

import gym

from pathlib import Path
import pickle

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

'''import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop'''

import os
from collections import deque
import numpy as np
'''from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Conv2D, Flatten, Lambda, Add
from keras.models import Model

import keras.backend as K'''

#Store losses for visualization
'''class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs = {}):
        self.losses = []

    def on_batch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))'''

class TestAgent():
    #globel graph
    #graph = tf.get_default_graph()
    def __init__(self):
        #DQN parameters
        self.now_phase = {}
        #self.green_sec = 40
        self.green_sec = 30
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}

        #self.wt = 1 #for PRE ISWeights
        #self.history = LossHistory()
        #self.buffer_length = 2000
        self.memory = deque(maxlen = 2000) #maxlen can be adjusted
        self.priority = deque(maxlen = 2000)
        self.learning_start = 2000
        self.update_model_freq = 1
        self.update_target_model_freq = 20

        self.n_multi_step = 4

        self.gamma = 0.9 #discount rate #Can be adjusted
        #
        self.epsilon = 1 #exploration rate #Can be adjusted
        self.epsilon_min = 0.01 #Can be adjusted
        self.epsilon_decay = 0.995 # Can be adjusted
        self.learning_rate = 0.005 # Can be adjusted
        #
        self.batch_size = 64
        self.ob_length = 24

        self.action_space = 8
        #self.action_space = 4

        #tf.reset_default_graph()
        self.model = self._build_model()

        #Remember to uncomment the following two lines
        #path = os.path.split(os.path.realpath(__file__))[0]
        #self.load_model(path, 99)
        #self.target_model = self._build_model()
        self.target_model = self.model
        self.update_target_network()

        #Init Session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    ####################################################
    def load_agent_list(self,agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)

    def load_roadnet(self,intersections,roads,agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents
    #######################################################
    
    '''def _huber_loss(self, y_true, y_pred, clip_delta = 1.0):
    	error = y_true - y_pred
    	cond = K.abs(error) <= clip_delta
    	squared_loss = 0.5 * K.square(error)
    	quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    	return self.wt * K.mean(tf.where(cond, squared_loss, quadratic_loss))'''

    def _build_model(self):
        #tf.reset_default_graph()
        self.input = tf.placeholder(dtype = tf.float32, shape = [None, self.ob_length], name = "Input")
        self.target = tf.placeholder(dtype = tf.float32, shape = [None, self.action_space], name = "Target")
        self.importance = tf.placeholder(dtype = tf.float32, shape = [None], name = "Importance")

        #out = tf.nn.relu(self.noisy_dense(24, self.input,"layer1"))
        #value = tf.nn.relu(self.noisy_dense(24, out, "value_layer1"))
        #value = self.noisy_dense(1, value,"value_layer2")
        out = tf.nn.relu(self.noisy_dense(24, self.input))
        #value = tf.nn.relu(self.noisy_dense(24, out))
        value = self.noisy_dense(1, out)

        #advantage = tf.nn.relu(self.noisy_dense(24, out))
        advantage = tf.nn.relu(self.noisy_dense(self.action_space, out))
        advantage = tf.subtract(advantage, tf.reduce_mean(advantage, axis = 1, keepdims = True))

        self.output = tf.add(value, advantage)

        loss = tf.reduce_mean(tf.multiply(tf.square(self.output - self.target), self.importance))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate).minimize(loss)
        return self.output

    def noisy_dense(self, units, inputs):
        #Use Factorised Gaussian Noise
        #https://zhuanlan.zhihu.com/p/136476579
        def f(x):
            return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
        #Initializer for mu and sigma
        #mu_init = tf.random_uniform_initializer(minval = -1 * 1 / np.power(inputs.shape[1].value, 0.5), maxval = 1 * 1 / np.power(inputs.shape[1].value, 0.5))
        #sigma_init = tf.constant_initializer(0.4 / np.power(inputs.shape[1].value, 0.5))
        #Sample noise from Gaussian Distribution
        p = tf.random_normal([inputs.shape[1].value, 1], dtype = tf.float32)
        q = tf.random_normal([1, units], dtype = tf.float32)
        f_p = f(p)
        f_q = f(q)
        w_epsilon = f_p * f_q
        b_epsilon = tf.squeeze(f_q)

        w_shape = [inputs.shape[1].value, units]
        ##w = w_mu + w_sigma * w_epsilon
        #w_mu = tf.get_variable(name + "/w_mu", w_shape, initializer = mu_init)
        #w_sigma = tf.get_variable(name + "/w_sigma", w_shape, initializer = sigma_init)
        w_mu = tf.Variable(initial_value = tf.random_uniform(w_shape, minval = -1 * 1 / np.power(inputs.shape[1].value, 0.5), maxval = 1 * 1 / np.power(inputs.shape[1].value, 0.5), dtype = tf.float32))
        w_sigma = tf.Variable(initial_value = tf.constant(0.4 / np.power(inputs.shape[1].value, 0.5), shape = w_shape, dtype = tf.float32))
        w = tf.add(w_mu, tf.multiply(w_sigma, w_epsilon))
        ret = tf.matmul(inputs, w)
        b_shape = [units]
        ##b = b_mu + b_sigma * b_epsilon
        #b_mu = tf.get_variable(name + "/b_mu", [units], initializer = mu_init)
        #b_sigma = tf.get_variable(name + "/b_sigma", [units], initializer = sigma_init)
        b_mu = tf.Variable(initial_value = tf.random_uniform(b_shape, minval = -1 * 1 / np.power(inputs.shape[1].value, 0.5), maxval = 1 * 1 / np.power(inputs.shape[1].value, 0.5), dtype = tf.float32))
        b_sigma = tf.Variable(initial_value = tf.constant(0.4 / np.power(inputs.shape[1].value, 0.5), shape = b_shape, dtype = tf.float32))
        b = tf.add(b_mu, tf.multiply(b_sigma, b_epsilon))

        return ret + b
        
        '''w_shape = [units, inputs.shape[1].value]
        mu_w = tf.Variable(initial_value = tf.truncated_normal(shape = w_shape))
        sigma_w = tf.Variable(initial_value = tf.constant(0.017, shape = w_shape))
        epsilon_w = tf.random_uniform(shape = w_shape)

        b_shape = [units]
        mu_b = tf.Variable(initial_value = tf.truncated_normal(shape = b_shape))
        sigma_b = tf.Variable(initial_value = tf.constant(0.017, shape = b_shape))
        epsilon_b = tf.random_uniform(shape = b_shape)

        w = tf.add(mu_w, tf.multiply(sigma_w, epsilon_w))
        b = tf.add(mu_b, tf.multiply(sigma_b, epsilon_b))

        return tf.matmul(inputs, tf.transpose(w)) + b'''
        
    def predict(self, inputs, model):
        #inputs = tf.to_float(inputs, name = "ToFloat")
        inputs = inputs.astype(np.float32)
        #print(inputs)
        #print("Shape is {}".format(self.sess.run(tf.shape(inputs))))
        #print("Dtype is {}".format(inputs.dtype))
        #print("Type of inputs is {}".format(type(inputs)))
        #print("self.input's name is {}".format(self.input.name))
        return self.sess.run(model, feed_dict = {self.input: inputs})

    def fit(self, inputs, target, importance):
        #tf.to_float(inputs, name = "ToFloat1")
        #print("self.target's name is {}".format(self.target.name))
        #print("self.importance's name is {}".format(self.importance.name))
        self.sess.run(self.optimizer, feed_dict = {self.input: inputs, self.target: target, self.importance: importance})
        
    def remember(self, ob, action, reward, next_ob, done):
        #Add transition to Memory Buffer
        ob = self._reshape_ob(ob)
        next_ob = self._reshape_ob(next_ob)
        self.prioritize(ob, action, reward, next_ob, done)

    def prioritize(self, ob, action, reward, next_ob, done, alpha = 0.6):
        q_next = reward + self.gamma * self.predict(next_ob, self.target_model)[0][np.argmax(self.predict(next_ob, self.model)[0])]
        q = self.predict(ob, self.model)[0][action]
        p = (np.abs(q_next - q) + (np.e ** -10)) ** alpha
        self.priority.append(p)
        self.memory.append((ob, action, reward, next_ob, done))

    def get_priority_experience_sample(self):
        p_sum = np.sum(self.priority)
        prob = self.priority / p_sum
        prob2 = prob[self.n_multi_step:len(prob)]
        #print(prob2)
        #prob2 = prob.copy()
        sample_indices = random.choices(range(self.n_multi_step, len(prob)), k = self.batch_size, weights = prob2)
        importance = (1 / prob) * (1 / self.batch_size)
        importance = np.array(importance)[sample_indices]
        #samples = np.array(self.memory)[sample_indices]
        #obs, actions, rewards, next_obs, dones = [], [], [], [], []
        samples = []
        for j in range(self.batch_size):
            one_sample = []
            one_sample.append(self.memory[sample_indices[j]][0])
            one_sample.append(self.memory[sample_indices[j]][1])
            #obs.append(self.memory[sample_indices[j]][0])
            #actions.append(self.memory[sample_indices[j]][1])
            reward_sum = 0.0
            for z in range(self.n_multi_step):
                reward_sum += self.gamma ** z * self.memory[sample_indices[j] - (self.n_multi_step - z)][2]
                if self.memory[sample_indices[j] - (self.n_multi_step - z)][4]:
                    next_ob = self.memory[sample_indices[j] - (self.n_multi_step - z)][3]
                    done = True
                    break
                else:
                    next_ob = self.memory[sample_indices[j] - (self.n_multi_step - z)][3]
                    done = False
            one_sample.append(reward_sum)
            one_sample.append(next_ob)
            one_sample.append(done)
            samples.append(one_sample)
            #rewards.append(reward_sum)
            #next_obs.append(next_ob)
            #dones.append(done)
        samples = np.array(samples)

        return samples, importance

    def replay(self):
        #self.time_step += 1
        #Step 1: obtain random minibatch from replay memory
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        minibatch, importance = self.get_priority_experience_sample()
        for b, i in zip(minibatch, importance):
            ob, action, reward, next_ob, done = b
            target = reward
            if not done:
                target = reward + self.gamma ** self.n_multi_step * self.predict(next_ob, self.target_model)[0][np.argmax(self.predict(next_ob, self.model)[0])]
            target_f = self.predict(ob, self.model)
            target_f[0][action] = target
            imp = i ** (1 - self.epsilon)
            imp = np.reshape(imp, 1)
            self.fit(ob, target_f, imp)
            #self.model.fit(ob, target, epochs = 1, verbose = 0, callbacks = [history])
            #
            '''with open('losses.txt', 'a', encoding = 'utf-8') as f:
                f.write(history.losses)
            f.close()'''
            #

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1,-1))
        #all elements in array become one row

    def act_(self, observations_for_agent):
        #Use this act_() for training
        #while keep act() function for evaluation unchanged
        actions = {}
        for agent_id in self.agent_list:
            #
            #print("Ob for act_ function is {}".format(observations_for_agent[agent_id]['lane']))
            action = self.get_action(observations_for_agent[agent_id]['lane'])
            actions[agent_id] = action
        return actions

    def act(self,obs):
        observations = obs['observations']
        info = obs['info']
        actions = {}

        #Get State
        observations_for_agent = {}
        for key, val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_') + 1:]
            if (observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val[1:]

        #Get Actions
        for agent in self.agent_list:
            self.epsilon = 0
            actions[agent] = self.get_action(observations_for_agent[agent]['lane_vehicle_num']) + 1

        return actions

    def get_action(self, ob):
        #Use the epsilon-greedy method to choose action
        '''if np.random.rand() <= self.epsilon:
        	return self.sample()
        ob = self._reshape_ob(ob)
        act_values = self.model.predict(ob)
        return np.argmax(act_values[0])'''
        ob = self._reshape_ob(ob)
        prediction = self.predict(ob, self.model)[0]
        return np.argmax(prediction)

    def update_target_network(self):
        #weights = self.model.get_weights()
        #self.target_model.set_weights(weights)
        self.target_model = self.model

    def load_model(self, dir = "model/dqn", step = 0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        print("load from " + model_name)
        #self.model.load_weights(model_name)
        #saver = tf.train.Saver()
        self.saver.restore(self.sess, model_name)

    def save_model(self, dir = "model_dqn", step = 0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        #self.model.save_weights(model_name)
        #saver = tf.train.Saver()
        self.saver.save(self.sess, model_name)


scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    #initialize an AgentSpec instance
    agent_specs[k] = TestAgent()


