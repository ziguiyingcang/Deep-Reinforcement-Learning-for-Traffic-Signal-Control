import pickle
import os
path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(path)
import random
#import math

import gym

from pathlib import Path
import pickle

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.optimizers import Adam, RMSprop, SGD
import os
from collections import deque
import numpy as np
#from tensorflow.python.framework import ops
#from keras.layers.merge import concatenate
#from keras.layers import Input, Dense, Conv2D, Flatten
#from keras.models import Model

# contains all of the intersections

#Store losses for visualization
'''class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs = {}):
        self.losses = []

    def on_batch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))'''

class TestAgent():
    def __init__(self):

        # DQN parameters

        self.now_phase = {}
        self.green_sec = 30
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}

        #self.history = LossHistory()

        self.memory = deque(maxlen=2000)
        #self.memory = deque(maxlen = 1200)
        self.learning_start = 2000
        #self.learning_start = 1200
        self.update_model_freq = 1
        self.update_target_model_freq = 20
        #self.update_target_model_freq = 10

        self.gamma = 0.9  # discount rate
        #self.epsilon = 0.1  # exploration rate
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        #
        self.batch_size = 64
        self.ob_length = 24

        self.action_space = 8

        self.model = self._build_model()

        # 注意去掉一下两行的注释！！Remember to uncomment the following lines when submitting, and submit your model file as well.
        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 99)
        self.target_model = self.model
        self.update_target_network()

        #Init Session
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self,agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)
    # intersections[key_id] = {
    #     'have_signal': bool,
    #     'end_roads': list of road_id. Roads that end at this intersection. The order is random.
    #     'start_roads': list of road_id. Roads that start at this intersection. The order is random.
    #     'lanes': list, contains the lane_id in. The order is explained in Docs.
    # }
    # roads[road_id] = {
    #     'start_inter':int. Start intersection_id.
    #     'end_inter':int. End intersection_id.
    #     'length': float. Road length.
    #     'speed_limit': float. Road speed limit.
    #     'num_lanes': int. Number of lanes in this road.
    #     'inverse_road':  Road_id of inverse_road.
    #     'lanes': dict. roads[road_id]['lanes'][lane_id] = list of 3 int value. Contains the Steerability of lanes.
    #               lane_id is road_id*100 + 0/1/2... For example, if road 9 have 3 lanes, then their id are 900, 901, 902
    # }
    # agents[agent_id] = list of length 8. contains the inroad0_id, inroad1_id, inroad2_id,inroad3_id, outroad0_id, outroad1_id, outroad2_id, outroad3_id
    def load_roadnet(self,intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents
    ################################

    def act_(self, observations_for_agent):
        # Instead of override, We use another act_() function for training,
        # while keep the original act() function for evaluation unchanged.

        actions = {}
        for agent_id in self.agent_list:
            action = self.get_action(observations_for_agent[agent_id]['lane'])
            actions[agent_id] = action
        return actions

    def act(self, obs):
        observations = obs['observations']
        info = obs['info']
        actions = {}

        # Get state
        observations_for_agent = {}
        for key,val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_')+1:]
            if(observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val[1:]

        # Get actions
        for agent in self.agent_list:
            self.epsilon = 0
            actions[agent] = self.get_action(observations_for_agent[agent]['lane_vehicle_num']) + 1

        return actions

    def get_action(self, ob):

        # The epsilon-greedy action selector.

        #if np.random.rand() <= self.epsilon:
        #    return self.sample()
        #?Need reshape or not?
        ob = self._reshape_ob(ob)
        '''z = self.model.predict([ob]) #Return a list [1x51, 1x51, 1x51]
        z_concat = np.vstack(z)
        q = np.sum(bp.multiply(z_concat, np.array(self.z)), axis = 1)
        action_idx = np.argmax(q)
        return action_idx'''
        prediction = self.predict(ob, self.model)[0]
        return np.argmax(prediction)


    def _build_model(self):
        self.input = tf.placeholder(dtype = tf.float32, shape = [None, self.ob_length], name = "Input")
        self.target = tf.placeholder(dtype = tf.float32, shape = [None, self.action_space], name = "Target")

        out = tf.nn.relu(self.noisy_dense(24, self.input))
        self.output = self.noisy_dense(self.action_space, out)

        loss = tf.reduce_mean(tf.square(self.output - self.target))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate = self.learning_rate).minimize(loss)

        return self.output

    def noisy_dense(self, units, inputs):
        def f(x):
            return tf.multiply(tf.sign(x), tf.pow(tf.abs(x), 0.5))
        p = tf.random_normal([inputs.shape[1].value, 1], dtype = tf.float32)
        q = tf.random_normal([1, units], dtype = tf.float32)
        f_p = f(p)
        f_q = f(q)
        w_epsilon = f_p * f_q
        b_epsilon = tf.squeeze(f_q)

        w_shape = [inputs.shape[1].value, units]
        w_mu = tf.Variable(initial_value = tf.random_uniform(w_shape, minval = -1 * 1 / np.power(inputs.shape[1].value, 0.5), 
            maxval = 1 * 1 / np.power(inputs.shape[1].value, 0.5), dtype = tf.float32))
        w_sigma = tf.Variable(initial_value = tf.constant(0.4 / np.power(inputs.shape[1].value, 0.5), 
            shape = w_shape, dtype = tf.float32))
        w = tf.add(w_mu, tf.multiply(w_sigma, w_epsilon))
        ret = tf.matmul(inputs, w)

        b_shape = [units]
        b_mu = tf.Variable(initial_value = tf.random_uniform(b_shape, minval = -1 * 1 / np.power(inputs.shape[1].value, 0.5), 
            maxval = 1 * 1 / np.power(inputs.shape[1].value, 0.5), dtype = tf.float32))
        b_sigma = tf.Variable(initial_value = tf.constant(0.4 / np.power(inputs.shape[1].value, 0.5), 
            shape = b_shape, dtype = tf.float32))
        b = tf.add(b_mu, tf.multiply(b_sigma, b_epsilon))

        return tf.add(ret, b)

    def predict(self, inputs, model):
        inputs = inputs.astype(np.float32)
        return self.sess.run(model, feed_dict = {self.input: inputs})

    def fit(self, inputs, target):
        self.sess.run(self.optimizer, feed_dict = {self.input: inputs, self.target: target})

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        #weights = self.model.get_weights()
        #self.target_model.set_weights(weights)
        self.target_model = self.model

    def remember(self, ob, action, reward, next_ob, done):
        ob = self._reshape_ob(ob)
        next_ob = self._reshape_ob(next_ob)
        self.memory.append((ob, action, reward, next_ob, done))

    def replay(self):
        # Update the Q network from the memory buffer.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.batch_size > len(self.memory):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, self.batch_size)

        i = 0
        q_sum = 0.0
        loss_sum = 0.0

        for ob, action, reward, next_ob, done in minibatch:
            i += 
            print("i is {}".format(i))
            target = reward
            if not done:
                target += self.gamma * np.amax(self.predict(next_ob, self.target_model)[0])
            #To store Q value
            q_sum += target

            target_f = self.predict(ob, self.model)
            #To store loss
            loss_sum += np.square(target_f[0][action] - target)
            target_f[0][action] = target
            self.fit(ob, target_f)

        avg_q = str(q_sum / i)
        print("Average Q value is {}".format(avg_q))
        with open("avg_q_noisy_dqn.txt", 'a', encoding = 'utf-8') as f:
            f.write(avg_q + "\n")
        f.close()

        avg_loss = str(loss_sum / i)
        print("Loss is {}".format(avg_loss))
        with open("loss_noisy_dqn.txt", 'a', encoding = 'utf-8') as x:
            x.write(avg_loss + "\n")
        x.close()


                #print("Target is {}".format(target))
                #print("Predictions are {}".format(self.predict(next_ob, self.target_model)))
        #obs, actions, rewards, next_obs, dones = [np.stack(x) for x in np.array(minibatch).T]
        #print("minibatch is {}".format(minibatch))
        '''target = rewards + self.gamma * np.amax(self.target_model.predict([next_obs]), axis=1)
        target_f = self.model.predict([obs])
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        self.model.fit([obs], target_f, epochs=1, verbose=0, callbacks = [self.history])
        print("History.loss is {}".format(self.history.losses))
        #
        with open('losses.txt', 'a', encoding = 'utf-8') as f:
            f.write(self.history.losses)
        f.close()
        #'''


    def load_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        print("load from " + model_name)
        #self.model.load_weights(model_name)
        self.saver.restore(self.sess, model_name)

    def save_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        #self.model.save_weights(model_name)
        self.saver.save(self.sess, model_name)

scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`

