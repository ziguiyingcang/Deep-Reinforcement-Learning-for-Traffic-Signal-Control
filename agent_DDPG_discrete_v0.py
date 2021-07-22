#Compared to DDPG-v0
#Add delta time to OU Noise
#Do hard update before training Actor and Critic netwokrs 

import pickle
import os
path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(path)
import random
import math

import gym

from pathlib import Path

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#import tensorlayer as tl

from collections import deque
import numpy as np
import numpy.random as nr

class ActorNetwork(object):
	def __init__(self, sess, ob_length, action_space, learning_rate, tau):
		self.sess = sess
		self.ob_length = ob_length
		self.action_space = action_space
		self.layer1_size = 24
		self.learning_rate = learning_rate
		self.tau = tau
		#self.batch_size = 64
		#create actor network
		self.ob_input, self.action_output, self.net = self.create_network(ob_length, action_space)

		#create target actor network
		self.target_ob_input, self.target_action_output, self.target_update, self.target_net = self.create_target_network(ob_length, action_space, self.net)

		#define training rules
		self.create_training_method()
		
		'''#This gradient will be provided by the critic network
		self.q_gradient_input = tf.placeholder(tf.float32, [None, self.action_space])

		#Combine the gradients
		self.parameters_gradients = tf.gradients(self.action_output, self.net, -self.q_gradient_input)

		#Optimize actor by applying gradients, variable pairs
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(self.parameters_gradients, self.net)'''

		#initialization
		self.sess.run(tf.initialize_all_variables())

		self.update_target()
		#self.save_network()
		#self.load_network()

		self.saver = tf.train.Saver()

	def create_training_method(self):
		self.q_gradient_input = tf.placeholder("float", [None, self.action_space]) #input from critic network
		self.parameters_gradients = tf.gradients(self.action_output, self.net, -self.q_gradient_input)
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.parameters_gradients, self.net))


	'''def create_training_method(self):
		#Define training optimizer
		self.y_input = tf.placeholder("float", [None, 1])
		weight_decay = tf.add_n([self.l2 * tf.nn.l2_loss(var) for var in self.net])
		#tf.nn_l2_loss: 利用L2范数来计算张量的误差值
		self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
		self.action_gradients = tf.gradients(self.q_value_output, self.action_input)'''

	def create_network(self, ob_length, action_space):
		layer1_size = self.layer1_size

		ob_input = tf.placeholder("float", [None, ob_length])

		W1 = self.variable([ob_length, layer1_size], ob_length)
		b1 = self.variable([layer1_size], ob_length)
		W2 = tf.Variable(tf.random_uniform([layer1_size, action_space], -3e-3, 3e-3))
		b2 = tf.Variable(tf.random_uniform([action_space], -3e-3, 3e-3))

		layer1 = tf.nn.relu(tf.matmul(ob_input, W1) + b1)
		action_output = tf.tanh(tf.matmul(layer1, W2) + b2)

		return ob_input, action_output, [W1, b1, W2, b2]

	def create_target_network(self, ob_length, action_space, net):
		ob_input = tf.placeholder("float", [None, ob_length])
		#指数加权平均的求法，具体的公式是：total = a * total + (1-a) * next
		ema = tf.train.ExponentialMovingAverage(decay = 1 - self.tau)
		#传入变量列表，可同时计算多个加权平均数
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		layer1 = tf.nn.relu(tf.matmul(ob_input, target_net[0]) + target_net[1])
		action_output = tf.tanh(tf.matmul(layer1, target_net[2]) + target_net[3])

		return ob_input, action_output, target_update, target_net

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self, q_gradient_batch, ob_batch):
		self.sess.run(self.optimizer, feed_dict = {
			self.q_gradient_input: q_gradient_batch,
			self.ob_input: ob_batch
			})

	def actions(self, ob_batch):
		return self.sess.run(self.action_output, feed_dict = {
			self.ob_input: ob_batch
			})

	def action(self, ob):
		return self.sess.run(self.action_output, feed_dict = {
			self.ob_input: [ob]
			})[0]

	def target_actions(self, ob_batch):
		return self.sess.run(self.target_action_output, feed_dict = {
			self.target_ob_input: ob_batch
			})

	def variable(self, shape, f):
		return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))

	def load_network(self, dir = "model/ddpg_actor", step = 0):
		'''self.saver = tf.train.Saver()
		checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
			print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
			print("Could not find old network weights")'''
		#self.saver = tf.train.Saver()
		name = "ddpg_agent_{}.h5".format(step)
		model_name = os.path.join(dir, name)
		print("load from " + model_name)
		self.saver.restore(self.sess, model_name)

	def save_network(self, dir = "model/ddpg_actor", step = 0):
		name = "ddpg_agent_{}.h5".format(step)
		model_name = os.path.join(dir, name)
		self.saver.save(self.sess, model_name)

class CriticNetwork(object):
	def __init__(self, sess, ob_length, action_space, c_learning_rate, l2, c_tau):
		self.sess = sess
		self.layer1_size = 24
		self.learning_rate = c_learning_rate
		self.tau = c_tau
		self.l2 = l2

		#create q network
		self.ob_input, self.action_input, self.q_value_output, self.net = self.create_q_network(ob_length, action_space)

		#create target q network (same structure with q network)
		self.target_ob_input, self.target_action_input, self.target_q_value_output, self.target_update = self.create_target_q_network(ob_length, action_space, self.net)

		self.create_training_method()

		#initialization
		self.sess.run(tf.initialize_all_variables())

		self.update_target()
		#self.load_q_network()
		#self.save_q_network()

		self.saver = tf.train.Saver()


	def create_training_method(self):
		#Define training optimizer
		self.y_input = tf.placeholder("float", [None, 1])
		weight_decay = tf.add_n([self.l2 * tf.nn.l2_loss(var) for var in self.net])
		#tf.nn_l2_loss: 利用L2范数来计算张量的误差值
		self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
		self.action_gradients = tf.gradients(self.q_value_output, self.action_input)

	def create_q_network(self, ob_length, action_space):
		layer1_size = self.layer1_size

		ob_input = tf.placeholder("float", [None, ob_length])
		action_input = tf.placeholder("float", [None, action_space])

		W1 = self.variable([ob_length, layer1_size], ob_length + action_space)
		W1_action = self.variable([action_space, layer1_size], ob_length + action_space)
		b1 = self.variable([layer1_size], ob_length + action_space)
		W2 = tf.Variable(tf.random_uniform([layer1_size, 1], -3e-3, 3e-3))
		b2 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))

		#Insert actions
		layer1 = tf.nn.relu(tf.matmul(ob_input, W1) + tf.matmul(action_input, W1_action) + b1)
		q_value_output = tf.identity(tf.matmul(layer1, W2) + b2)

		return ob_input, action_input, q_value_output, [W1, W1_action, b1, W2, b2]

	def create_target_q_network(self, ob_length, action_space, net):
		ob_input = tf.placeholder("float", [None, ob_length])
		action_input = tf.placeholder("float", [None, action_space])

		ema = tf.train.ExponentialMovingAverage(decay = 1 - self.tau)
		target_update = ema.apply(net)
		target_net = [ema.average(x) for x in net]

		layer1 = tf.nn.relu(tf.matmul(ob_input, target_net[0]) + tf.matmul(action_input, target_net[1]) + target_net[2])
		q_value_output = tf.identity(tf.matmul(layer1, target_net[3]) + target_net[4])

		return ob_input, action_input, q_value_output, target_update

	def update_target(self):
		self.sess.run(self.target_update)

	def train(self, y_batch, ob_batch, action_batch):
		#self.time_step += 1
		self.sess.run(self.optimizer, feed_dict = {
			self.y_input: y_batch,
			self.ob_input: ob_batch,
			self.action_input: action_batch
			})

	def gradients(self, ob_batch, action_batch):
		return self.sess.run(self.action_gradients, feed_dict = {
			self.ob_input: ob_batch,
			self.action_input: action_batch
			})[0]

	def target_q(self, ob_batch, action_batch):
		return self.sess.run(self.target_q_value_output, feed_dict = {
			self.target_ob_input: ob_batch,
			self.target_action_input: action_batch
			})

	def q_value(self, ob_batch, action_batch):
		return self.sess.run(self.q_value_output, feed_dict = {
			self.ob_input: ob_batch,
			self.action_input:action_batch
			})

	def variable(self, shape, f):
		return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))

	def load_q_network(self, dir = "model/ddpg_critic", step = 0):
		name = "ddpg_agent_{}.h5".format(step)
		model_name = os.path.join(dir, name)
		print("load from " + model_name)
		self.saver.restore(self.sess, model_name)

	def save_q_network(self, dir = "model/ddpg_critic", step = 0):
		name = "ddpg_agent_{}.h5".format(step)
		model_name = os.path.join(dir, name)
		self.saver.save(self.sess, model_name)


class Memory(object):
	def __init__(self, memory_size):
		self.memory_size = memory_size
		self.num_experiences = 0
		self.memory = deque()

	def get_batch(self, batch_size):
		if self.num_experiences < batch_size:
			return random.sample(self.memory, self.num_experiences)
		else:
			return random.sample(self.memory, batch_size)

	def size(self):
		return self.memory_size

	#Remember transitions
	def reme(self, ob, action, reward, next_ob, done):
		experience = (ob, action, reward, next_ob, done)
		if self.num_experiences < self.memory_size:
			self.memory.append(experience)
			self.num_experiences += 1
		else:
			self.memory.popleft()
			self.memory.append(experience)

	def count(self):
		return self.num_experiences

	def erase(self):
		self.memory = deque()
		self.num_experiences = 0

#Ornstein-Uhlenback Noise: 使用于时间离散粒度小，惯性系统
class OUNoise(object):
	def __init__(self, action_space, mu, theta, sigma):
		self.action_space = action_space
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.ob = np.ones(self.action_space) * self.mu
		self.reset()

	def reset(self):
		self.ob = np.ones(self.action_space) * self.mu

	def noise(self):
		x = self.ob
		dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(size = len(x))
		self.ob = x + dx
		return self.ob

#Based in brownian motion
'''class OUNoise(object):
	def __init__(self, delta, sigma, ou_a, ou_mu):
		self.delta = delta
		self.sigma = sigma
		self.ou_a = ou_a
		self.ou_mu = ou_mu

	def brownian_motion_log_returns(self):
		#Return a Wiener proess / Brownian motion
		sqrt_delta_sigma = np.sqrt(self.delta) * self.sigma
		return np.random.normal(loc = 0, scale = sqrt_delta_sigma, size = None)

	def ornstein_uhlenbeck_level(self, prev_ou_level):
		#Return the rate levels of a mean_reverting ornstein uhlenbeck process
		drift = self.ou_a * (self.ou_mu - prev_ou_level) * self.delta
		randomness = self.brownian_motion_log_returns()
		return prev_ou_level + drift + randomness'''


#action定义方法1: 定义为<60s的绿灯时间，若绿灯时间小于5秒则跳过该相位
#??action_dim在动作连续的情况下如何定义？？
class DDPGAgent():
	def __init__(self):
		self.now_phase = {}
		self.green_sec = 30
		self.red_sec = 5
		self.last_change_step = {}
		self.agent_list = []
		self.phase_passablelane = {}
		self.max_phase = 8

		self.memory_size = 20
		self.learning_start = 10
		self.update_model_freq = 1

		self.batch_size = 10
		self.gamma = 0.95

		#self.environment = env
		self.ob_length = 24 #env.observation_space.shape[0]
		#self.action_space = env.action_space.shape[0]
		self.action_space = 8

		#for actor network
		self.a_learning_rate = 0.0001
		self.a_tau = 0.001

		#for critic network
		self.c_learning_rate = 0.001
		self.c_tau = 0.001
		self.l2 = 0.01

		#Epsiodes with noise
		self.noise_max_ep = 15

		#for ou noise
		self.mu = 0.
		self.theta = 0.15
		self.sigma = 0.2

		#for Brownian Motion
		'''self.delta = 0.5 #the rate of change (time)
		self.sigma = 0.5 #volatility of the stochastic processes
		self.ou_a = 3. #the rate of mean reversion
		self.ou_mu = 0. #the long run average interest rate'''

		self.sess = tf.InteractiveSession()
		#self.sess.run(tf.global_variables_initializer())

		self.actor_network = ActorNetwork(self.sess, self.ob_length, self.action_space, self.a_learning_rate, self.a_tau)
		self.critic_network = CriticNetwork(self.sess, self.ob_length, self.action_space, self.c_learning_rate, self.l2, self.c_tau)

		#Initialize replay buffer
		self.memory = Memory(self.memory_size)

		#Initialize a random process the Ornstein-Uhlenbeck process for action exploration
		self.exploration_noise = OUNoise(self.action_space, self.mu, self.theta, self.sigma)

	#############################
	#don't modify this function
	def load_agent_list(self, agent_list):
		self.agent_list = agent_list
		self.now_phase = dict.fromkeys(self.agent_list, 1)
		self.last_change_step = dict.fromkeys(self.agent_list, 0)

	def load_roadnet(self, intersections, roads, agents):
		self.intersections = intersections
		self.roads = roads
		self.agents = agents
	##############################

	def replay(self):
		minibatch = self.memory.get_batch(self.batch_size)
		ob_batch = np.asarray([data[0] for data in minibatch])
		action_batch = np.asarray([data[1] for data in minibatch])
		reward_batch = np.asarray([data[2] for data in minibatch])
		next_ob_batch = np.asarray([data[3] for data in minibatch])
		done_batch = np.asarray([data[4] for data in minibatch])

		#print("Reward_batch is {}".format(reward_batch))

		# for action_space  = 1
		action_batch = np.resize(action_batch, [self.batch_size, self.action_space])

		#Calculate y_batch
		next_action_batch = self.actor_network.target_actions(next_ob_batch)
		q_value_batch = self.critic_network.target_q(next_ob_batch, next_action_batch)
		y_batch = []


		#To store q-values and losses
		num = 0
		q_sum = 0
		critic_loss_sum = 0

		for i in range(len(minibatch)):
			num += 1
			if done_batch[i]:
				target = reward_batch[i]
			else:
				target = reward_batch[i] + self.gamma * q_value_batch[i]
			y_batch.append(target)
			#To store Q value
			q_sum += target
		#To store critic loss
		target_f = self.critic_network.q_value(ob_batch, action_batch)
		critic_loss_sum = np.sum(np.square(y_batch - target_f))
		
		y_batch = np.resize(y_batch, [self.batch_size, 1])
		#Update critic by minimize the loss L
		self.critic_network.train(y_batch, ob_batch, action_batch)

		##
		#l= np.array(q_sum / num)
		#print("Type of l is {}".format(type(l)))
		avg_q = q_sum / num
		try:
			avg_q = str(avg_q[0])
		except:
			avg_q = str(avg_q)
		print("Average Q value is {}".format(avg_q))
		with open("q_ddpg.txt", 'a', encoding = 'utf-8') as f:
			f.write(avg_q + "\n")
		f.close()

		avg_critic_loss = str(critic_loss_sum / num)
		print("Loss is {}".format(avg_critic_loss))
		with open("loss_critic_ddpg.txt", 'a', encoding = 'utf-8') as x:
			x.write(avg_critic_loss + "\n")
		x.close()
		##

		#Update the actor policy using the sampled gradient
		action_batch_for_gradients = self.actor_network.actions(ob_batch)
		q_gradient_batch = self.critic_network.gradients(ob_batch, action_batch_for_gradients)

		self.actor_network.train(q_gradient_batch, ob_batch)

		#Update the target networks
		self.actor_network.update_target()
		self.critic_network.update_target()

	def remember(self, ob, action, reward, next_ob, done):
		self.memory.reme(ob, action, reward, next_ob, done)
		#print("Current memory size is {}".format(self.memory.count()))

#如果是连续动作，且动作为（相位序数，持续时间）这个形式，这里如何定义选取动作？
	def act_(self, observations_for_agent, episode):
		#Instead of override, we use another act_() function for training
		#while keep the original act() function for evaluation unchanged
		actions = {}
		if episode < self.noise_max_ep:
			for agent_id in self.agent_list:
				action = self.noise_action(observations_for_agent[agent_id]['lane'])
				actions[agent_id] = action
		else:
			for agent_id in self.agent_list:
				action = self.action(observations_for_agent[agent_id]['lane'])
				actions[agent_id] = action
		return actions

	def act(self, obs, episode):
		observations = obs['observations']
		info = obs['info']
		actions = {}

		#Get observation
		observations_for_agent = {}
		for key, val in observations.items():
			#
			#print("Key is {}".format(key))
			observations_agent_id = int(key.split('_')[0])
			observations_feature = key[key.find('_') + 1:]
			if (observations_agent_id not in observations_for_agent.keys()):
				observations_for_agent[observations_agent_id] = {}
			observations_for_agent[observations_agent_id][observations_feature] = val[1:]

		#Get actions
		#Previously, episilon = 0 means no exploration, thus here for ddpg, no noise added to action
		if episode < self.noise_max_ep:
			for agent in self.agent_list:
				actions[agent] = self.noise_action(observations_for_agent[agent]['lane_vehicle_num']) + 1
		else:
			for agent in self.agent_list:
				actions[agent] = self.action(observations_for_agent[agent]['lane_vehicle_num']) + 1

		return actions

	def noise_action(self, ob):
		#Select action according to the current policy and exploration noise
		action = self.actor_network.action(ob)
		a = action + self.exploration_noise.noise()
		#print("The noise action is {}".format(np.argmax(a)))
		#print("Noisy a are {}".format(a))
		return np.argmax(a)

	def action(self, ob):
		action = self.actor_network.action(ob)
		#print("The action is {}".format(np.argmax(action)))
		#print("A are {}".format(action))
		return np.argmax(action)

	def save_model(self, dirs, step):
		self.actor_network.save_network(dirs, step)
		self.critic_network.save_q_network(dirs, step)

	def load_model(self, dirs, step):
		self.actor_network.load_network(dirs, step)
		self.critic_network.load_q_network(dirs, step)

	#Should be added in train_dqn.py
	'''def perceive(self, ob, action, reward, next_ob, done):
		#Remember transition
		self.memory.remember(ob, action, reward, next_ob, done)

		if self.memory.count() > self.memory_size:
			self.replay()

		#to save network for actor and critic

		#Re-initialize the random process when an episode ends
		if done:
			self.exploration_noise.reset()'''

scenario_dirs = [
	"test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
	agent_specs[k] = DDPGAgent()





