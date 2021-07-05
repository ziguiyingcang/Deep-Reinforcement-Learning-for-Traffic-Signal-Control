""" Required submission file

In this file, you should implement your `AgentSpec` instance, and **must** name it as `agent_spec`.
As an example, this file offers a standard implementation.
"""
#Has added history to store all the losses for visualization
#import tensorflow as tf

import pickle
import os
path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(path)
import random
import math

import gym

from pathlib import Path
import pickle
import gym

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
from tensorflow.python.framework import ops
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

def dense(inputs, units, bias_shape, weights, bias = None, activation = tf.nn.relu):
    if not isinstance(inputs, ops.Tensor):
        #print("Not tensor!!")
        #print(inputs)
        inputs = ops.convert_to_tensor(inputs, dtype = 'float')
    if len(inputs.shape) > 2:
        inputs = tf.layers.flatten(inputs)
    flatten_shape = inputs.shape[1]
    weights = tf.get_variable('weights', shape = [flatten_shape, units], initializer = weights)
    dense = tf.matmul(inputs, weights)
    if bias_shape is not None:
        assert bias_shape[0] == units
        biases = tf.get_variable('biases', shape = bias_shape, initializer = bias)
        return activation(dense + biases) if activation is not None else dense + biases

    return activation(dense) if activation is not None else dense


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

        self.memory = deque(maxlen=20)
        #self.memory = deque(maxlen = 1200)
        self.learning_start = 20
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
        self.batch_size = 10
        self.ob_length = 24

        self.action_space = 8

        #For distributional DQN
        self.v_min = -100
        self.v_max = 100
        self.atoms = 51
        self.delta_z = (self.v_max - self.v_min) / float(self.atoms - 1)
        self.z = [self.v_min + i * self.delta_z for i in range(self.atoms)]

        #Initialize the target observation shape
        target_ob_shape = [1]
        target_ob_shape.extend([self.ob_length])

        #Define the placeholder for the observation
        self.ob_ph = tf.placeholder(tf.float32, target_ob_shape)

        #Define the placeholder for the action
        self.action_ph = tf.placeholder(tf.int32, [1, 1])

        #Define the placeholder for the m value (distributed probability of target distribution)
        self.m_ph = tf.placeholder(tf.float32, [self.atoms])

        #self.model = self._build_model()

        # 注意去掉一下两行的注释！！Remember to uncomment the following lines when submitting, and submit your model file as well.
        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 99)
        #self.target_model = self._build_model()
        self.build_categorical_DQN()
        #self.update_target_network()

        #Initializer all the TensorFlow variables
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

        if np.random.rand() <= self.epsilon:
            return self.sample()
        return np.argmax([self.sess.run(self.main_Q, feed_dict = {self.ob_ph: [ob], self.action_ph: [[a]]}) for a in range(self.action_space)])
        #ob = self._reshape_ob(ob)
        #act_values = self.model.predict([ob])
        #return np.argmax(act_values[0])

    def sample(self):

        # Random samples

        return np.random.randint(0, self.action_space)


    '''def _build_model(self):

        # Neural Net for Deep-Q learning Model
        
        model = Sequential()
        model.add(Dense(20, input_dim=self.ob_length, activation='relu'))
        # model.add(Dense(20, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=Adam(lr = self.learning_rate)
        )
        return model'''
    
    '''def dense(inputs, units, bias_shape, weights, bias = None, activation = tf.nn.relu):
        if not isinstance(inputs, ops.Tensor):
            print("Not tensor!!")
            print(inputs)
            inputs = ops.convert_to_tensor(inputs, dtype = 'float')
        if len(inputs.shape) > 2:
            inputs = tf.layers.flatten(inputs)
        flatten_shape = inputs.shape[1]
        weights = tf.get_variable('weights', shape = [flatten_shape, units], initializer = weights)
        dense = tf.matmul(inputs, weights)
        if bias_shape is not None:
            assert bias_shape[0] == units
            biases = tf.get_variable('biases', shape = bias_shape, initializer = bias)
            return activation(dense + biases) if activation is not None else dense + biases

        return activation(dense) if activation is not None else dense'''

    #Units is for dense layer
    def build_network(self, ob, action, name, units_1, weights, bias, reg = None):
        with tf.variable_scope('dense1'):
            dense1 = dense(ob, units_1, [units_1], weights, bias)
        #print("Dense work!!")

        #concatenate the dense layer with the action
        with tf.variable_scope('concat'):
            concatenated = tf.concat([dense1, tf.cast(action, tf.float32)], 1)

        #obtain the probabilities for each of the atoms
        with tf.variable_scope('dense2'):
            dense2 = dense(concatenated, self.atoms, [self.atoms], weights, bias)
        return tf.nn.softmax(dense2)

    #Define build_categorical_DQN for building model and target_model
    def build_categorical_DQN(self):
        #main model
        with tf.variable_scope('main_net'):
            name = ['main_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            weights = tf.random_uniform_initializer(-0.1, 0.1)
            bias = tf.constant_initializer(0.1)

            self.model = self.build_network(self.ob_ph, self.action_ph, name, 24, weights, bias)

        #target model
        with tf.variable_scope('target_net'):
            name = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            weights = tf.random_uniform_initializer(-0.1, 0.1)
            bias = tf.constant_initializer(0.1)

            self.target_model = self.build_network(self.ob_ph, self.action_ph, name, 24, weights, bias)

        #compute the main Q value with probabilities from the main model
        self.main_Q = tf.reduce_sum(self.model * self.z)

        #compute the target Q value with probabilities from the target model
        self.target_Q = tf.reduce_sum(self.target_model * self.z)

        #define the cross entropy loss
        self.cross_entropy_loss = -tf.reduce_sum(self.m_ph * tf.log(self.model))

        #define optimizer and minimize teh cross entropy loss
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cross_entropy_loss)


    #def _reshape_ob(self, ob):
    #    return np.reshape(ob, (1, -1))

    def update_target_network(self):
        #weights = self.model.get_weights()
        #self.target_model.set_weights(weights)
        main_net_params = tf.get_collection("main_net_params")
        target_net_params = tf.get_collection("target_net_params")
        self.update_target_net = [tf.assign(t, e) for t, e in zip(target_net_params, main_net_params)]
        self.sess.run(self.update_target_net)

    def remember(self, ob, action, reward, next_ob):
        self.memory.append((ob, action, reward, next_ob))

    def replay(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # Update the Q network from the memory buffer.

        if self.batch_size > len(self.memory):
            minibatch = self.memory
        else:
            minibatch = random.sample(self.memory, self.batch_size)
        obs, actions, rewards, next_obs, = [np.stack(x) for x in np.array(minibatch).T]
        sum_target_q = 0.0
        sum_loss = 0.0
        i = 0
        
        for ob, action, r, ob_ in zip(obs, actions, rewards, next_obs):
            #get target Q values
            list_q_ = [self.sess.run(self.target_Q, feed_dict = {self.ob_ph: [ob_], self.action_ph: [[a]]}) for a in range(self.action_space)]
            #To store the Q-value
            sum_target_q += r + self.gamma * max(list_q_)
            i += 1
            print("i is {}".format(i))

            #select the next state action with the max q value
            a_ = tf.argmax(list_q_).eval()
            #initialize an array m denoting the distributed probability of the target probability after the projection step
            m = np.zeros(self.atoms)
            #get the probability for each atom using target DQN
            p = self.sess.run(self.target_model, feed_dict = {self.ob_ph: [ob_], self.action_ph: [[a_]]})[0]

            #perform the projection step
            for j in range(self.atoms):
                Tz = min(self.v_max, max(self.v_min, r + self.gamma * self.z[j]))
                bj = (Tz - self.v_min) / self.delta_z
                l, u = math.floor(bj), math.ceil(bj)

                pj = p[j]

                m[int(l)] += pj * (u - bj)
                m[int(u)] += pj * (bj - l)

            #train the network by minimizing the loss
            self.sess.run(self.optimizer, feed_dict = {self.ob_ph: [ob], self.action_ph: [[action]], self.m_ph: m})
            
            sum_loss += self.sess.run(self.cross_entropy_loss, feed_dict = {self.ob_ph: [ob], self.action_ph: [[action]], self.m_ph: m})
            #To store each loss
            '''print ("Loss is {}".format(self.cross_entropy_loss))
            with open('loss_dist_dqn.txt', 'a', encoding = 'utf-8') as f:
                f.write(str(self.cross_entropy_loss) + "\n")
            f.close()'''

        #To store both average Q value and average loss
        avg_q = str(sum_target_q / i)
        print("Average Q value is {}".format(avg_q))
        '''with open("avg_q_dist_dqn.txt", 'a', encoding = 'utf-8') as f:
            f.write(avg_q + "\n")
        f.close()'''

        avg_loss = str(sum_loss / i)
        print("Average loss is {}".format(avg_loss))
        '''with open("loss_dist_dqn.txt", 'a', encoding = 'utf-8') as x:
            x.write(avg_loss + "\n")
        x.close()'''

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

