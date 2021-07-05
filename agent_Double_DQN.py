""" Required submission file

In this file, you should implement your `AgentSpec` instance, and **must** name it as `agent_spec`.
As an example, this file offers a standard implementation.
"""

import pickle
import os
path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(path)
import random

import gym

from pathlib import Path
import pickle
import gym

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop, SGD
import os
from collections import deque
import numpy as np
from keras.layers.merge import concatenate
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.models import Model

# contains all of the intersections

'''class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs = {}):
        self.losses = []
    
    def on_batck_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))''' 


class TestAgent():
    def __init__(self):

        # DQN parameters

        self.now_phase = {}
        self.green_sec = 30
        #
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}

        self.memory = deque(maxlen=2000)
        self.learning_start = 2000
        self.update_model_freq = 1
        self.update_target_model_freq = 20

        self.gamma = 0.9  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        #
        self.learning_rate = 0.005
        #
        self.batch_size = 64
        self.ob_length = 24

        self.action_space = 8
        
        #self.history = LossHistory()
        self.all_losses = []

        self.model = self._build_model()

        # Remember to uncomment the following lines when submitting, and submit your model file as well.
        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 99)
        self.target_model = self._build_model()
        self.update_target_network()



    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self,agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list,1)
        self.last_change_step = dict.fromkeys(self.agent_list,0)
        #
        print(len(agent_list))
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

        if np.random.rand() <= self.epsilon:
            return self.sample()
        ob = self._reshape_ob(ob)
        act_values = self.model.predict([ob])
        return np.argmax(act_values[0])

    def sample(self):

        # Random samples

        return np.random.randint(0, self.action_space)

    def _build_model(self):

        # Neural Net for Deep-Q learning Model
        
        model = Sequential()
        #
        model.add(Dense(24, input_dim=self.ob_length, activation='relu'))
        # model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=RMSprop()
        )
        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, ob, action, reward, next_ob):
        self.memory.append((ob, action, reward, next_ob))

    def replay(self):
        # Update the Q network from the memory buffer.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if self.batch_size > len(self.memory):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, self.batch_size)
        obs, actions, rewards, next_obs, = [np.stack(x) for x in np.array(minibatch).T]
        qs = self.target_model.predict([next_obs])
        next_actions = np.argmax(self.model.predict([next_obs]), axis = 1)
        target = []
        for i, q_values in enumerate(qs):
            target.append(rewards[i] + self.gamma * q_values[next_actions[i]])
        
        # To store average Q values
        avg_q_value = str(np.mean(target))
        print("Average Q value is {}".format(avg_q_value))
        with open('avg_q_double_dqn.txt', 'a', encoding = 'utf-8') as a:
            a.write(avg_q_value + "\n")
        a.close()
        #
        
        target_f = self.model.predict([obs])
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        self.model.fit([obs], target_f, epochs=1, verbose=0)
        
        # To store all the loss values
        History = self.model.fit([obs], target_f, epochs = 1, verbose = 0)
        train_loss = History.history['loss']
        train_loss = str(train_loss[0])
        print("Train_loss is {}".format(train_loss))
        with open('loss_double_dqn.txt', 'a', encoding = 'utf-8') as f:
            f.write(train_loss + "\n")
        f.close()
        #
        '''if len(self.all_losses) < len(self.agent_list):
            self.all_losses.append(train_loss[0])
        if len(self.all_losses) == len(self.agent_list):
            avg_loss = str(np.mean(self.all_losses))
            print("All losses are {}".format(self.all_losses))
            print("Average loss is {}".format(avg_loss))
            with open('loss.txt', 'a', encoding = 'utf-8') as f:
                f.write(avg_loss + "\n")
            f.close()
            self.all_losses = []'''
        #train_loss = np.array(train_loss)
        #avg_loss = str(np.mean(train_loss))
        
    def load_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        print("load from " + model_name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)

scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`

