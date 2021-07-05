import pickle
import os
path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(path)
import random

import gym

from pathlib import Path
import pickle

##import tensorflow.compat.v1 as tf
##tf.disable_v2_behavior()
import tensorflow as tf

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop

import os
from collections import deque
import numpy as np
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model

#import keras.backend as K


class SumTree(object):
    data_pointer = 0

    def __init__(self,capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype = object)

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, p)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        parent_idx = 0
        while True:
            cl_idx = 2 * parent_idx + 1
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0] #the root


#For Prioritized Experience Replay
class Memory(object): #stored as (s, a, r, s_) in SumTree
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4 #from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype = np.int32), [], np.empty((n, 1), dtype = np.float32)
        #b_idx, b_memory, ISWeights = np.empty((n,), dtype = np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        if min_prob == 0:
            min_prob = 0.00001
        #max_weight = (min_prob * n) ** (-self.beta)
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            #ISWeights[i, 0] = np.power(n * prob, -self.beta) / max_weight
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i] = idx
            experience = data
            b_memory.append(experience)
            #b_memory[i, :] = data
            
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

#Store losses for visualization
'''class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs = {}):
        self.losses = []

    def on_batch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))'''

class TestAgent():
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

        #New indexes
        #self.replay_total = 0
        #self.time_step = 0

        self.wt = 1 #for PRE ISWeights
        #self.history = LossHistory()
        #self.memory = deque(maxlen = 2000) #maxlen can be adjusted
        self.memory = Memory(capacity = 2000)
        self.learning_start = 2000
        self.update_model_freq = 1
        self.update_target_model_freq = 20

        self.gamma = 0.9 #discount rate #Can be adjusted
        self.epsilon = 1 #exploration rate #Can be adjusted
        self.epsilon_min = 0.01 #Can be adjusted
        self.epsilon_decay = 0.995 # Can be adjusted
        self.learning_rate = 0.005 # Can be adjusted
        self.batch_size = 64
        self.ob_length = 24

        self.action_space = 8
        #self.action_space = 4

        self.model = self._build_model()

        #Remember to uncomment the following two lines
        #path = os.path.split(os.path.realpath(__file__))[0]
        #self.load_model(path, 99)
        self.target_model = self._build_model()
        self.update_target_network()
        #self.create_Q_network()
        #self.create_training_method()

        #Init Session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

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

    def _pre_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.multiply(tf.square(y_true - y_pred), self.wt))
        return loss

    def _build_model(self):
        #Build Q-network
        #Dueling DQN
        '''X_input = Input(shape = (self.ob_length,))
        #X = X_input
        X = Dense(24, input_shape = (self.ob_length,),activation = 'relu')(X_input)
        #X = Dense(20, activation = 'relu')(X)
        #X = Dense(20, activation = 'relu')(X)
        state_value = Dense(1)(X)
        state_value = Lambda(lambda v: v, output_shape = (self.action_space,))(state_value)
        action_advantage = Dense(self.action_space)(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims = True), output_shape = (self.action_space,))(action_advantage)

        #X = (state_value + (action_advantage - tf.math.reduce_mean(action_advantage, axis = 1, keepdims = True)))
        X = Add()([state_value, action_advantage])

        model = Model(inputs = X_input, outputs = X)
        model.compile(loss = self._huber_loss, optimizer = RMSprop(lr = self.learning_rate))'''
        
        model = Sequential()
        model.add(Dense(24, input_dim = self.ob_length, activation = 'relu'))
        model.add(Dense(self.action_space, activation = 'linear'))
        model.compile(
            loss = self._pre_loss,
            optimizer = RMSprop(lr = self.learning_rate)
        )
        return model
        
    def remember(self, ob, action, reward, next_ob, done):
        #Add transition to Memory Buffer
        #transition = np.hstack((ob, action, reward, next_ob, done))
        self.memory.store((ob, action, reward, next_ob, done))
        #self.memory.append((ob, action, reward, next_ob, done))

    def replay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay        
        #Step 1: obtain random minibatch from replay memory
        i = 0
        abs_err = []
        q_sum = 0.0
        loss_sum = 0.0
        
        tree_idx, minibatch, ISWeights = self.memory.sample(self.batch_size)
        for ob, action, reward, next_ob, done in minibatch:
            print(str(reward))
            self.wt = ISWeights[i]
            #print(self.wt)
            i += 1
            print("i is {}".format(i))
            ob = self._reshape_ob(ob)
            next_ob = self._reshape_ob(next_ob)
            target = self.model.predict(ob)
        	#q = self.target_model.predict(next_ob)
        	#next_action = np.argmax(self.model.predict(next_ob))
            #To store all losses
            y_pred = target[0][action]

            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + self.gamma * np.amax(self.target_model.predict(next_ob)[0])
            
            #To store Q value
            y_true = reward + self.gamma * np.amax(self.target_model.predict(next_ob)[0])
            q_sum += y_true
            loss_sum += np.multiply(self.wt, np.square(y_pred - y_true))
            print(loss_sum)

            abs_err.append(np.abs(np.sum(self.model.predict(ob) - target)))
        	#self.model.fit(ob, target, epochs = 1, verbose = 0)
            self.model.fit(ob, target, epochs = 1, verbose = 0)
        self.memory.batch_update(tree_idx, np.array(abs_err))

        #print(type(q_sum / i))
        avg_q = str(q_sum / i)
        print("Average Q value is {}".format(avg_q))
        with open("avg_q_PRE_dqn.txt", 'a', encoding = 'utf-8') as f:
            f.write(avg_q + "\n")
        f.close()

        #print("Loss_sum is {}".format(loss_sum))
        avg_loss = str((loss_sum / i)[0])
        print("Loss is {}".format(avg_loss))
        with open("loss_PRE_dqn.txt", 'a', encoding = 'utf-8') as x:
            x.write(avg_loss + "\n")
        x.close()
        '''obs, actions, rewards, next_obs, dones = [np.stack(x) for x in np.array(minibatch).T]
        target = rewards + self.gamma * np.amax(self.target_model.predict([next_obs]), axis = 1)
        #To store average Q values
        avg_q = str(np.mean(target))
        print("Average Q value is {}".format(avg_q))
        with open("avg_q_PRE_dqn.txt", 'a', encoding = 'utf-8') as f:
            f.write(avg_q + "\n")
        f.close()
        #

        target_f = self.model.predict([obs])
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        #abs_err.append(np.abs(np.sum(self.model.predict([obs]) - target_f)))
        for i, ob in enumerate(obs):
            ob = self._reshape_ob(ob)
            abs_err.append(np.abs(np.sum(self.model.predict([ob]) - target_f[i])))
        #print("Abs errors are {}".format(abs_err))
        self.model.fit([obs], target_f, epochs = 1, verbose = 0)
        self.memory.batch_update(tree_idx, np.array(abs_err))

        #To store losses
        #print("Target are {}".format(target))
        #print("Y_predict are {}".format(self.model.predict([obs])))
        y_predict = []
        for i, action in enumerate(actions):
            y_predict.append(self.model.predict([obs])[i][action])
        #print("Y_predict are {}".format(y_predict))
        loss = np.mean(np.multiply(np.square(target - y_predict), self.wt))
        avg_loss = str(loss)
        print("Loss is {}".format(avg_loss))
        with open("loss_PRE_dqn.txt", 'a', encoding = 'utf-8') as x:
            x.write(avg_loss + "\n")
        x.close()
        #'''

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1,-1))
        #all elements in array become one row

    def act_(self, observations_for_agent):
        #Use this act_() for training
        #while keep act() function for evaluation unchanged
        actions = {}
        for agent_id in self.agent_list:
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
        if np.random.rand() <= self.epsilon:
        	return self.sample()
        ob = self._reshape_ob(ob)
        act_values = self.model.predict(ob)
        return np.argmax(act_values[0])

    def sample(self):
        #Get random samples
        return np.random.randint(0, self.action_space)

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def load_model(self, dir = "model/dqn", step = 0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        print("load from " + model_name)
        self.model.load_weights(model_name)

    def save_model(self, dir = "model_dqn", step = 0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)

scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    #initialize an AgentSpec instance
    agent_specs[k] = TestAgent()
    #Policy Builder?? A callable function??


