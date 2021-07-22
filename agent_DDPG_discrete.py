#Compared to DDPG-v0
#Add delta time to OU noise
#Add param noise
#When do exploration, use random.choice to sample from action probabilities rather than argmax
#Use softmax activation rather than tanh
#Do hard update before training of actor and critic networks

import pickle
import os
path = os.path.split(os.path.realpath(__file__))[0]
import sys
sys.path.append(path)

import random
import math

import gym

from pathlib import Path

import tensorflow as tf
#tf.disable_v2_behavior()
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

from collections import deque
import numpy as np
import numpy.random as nr

#Can slows down training by a bit
#How to define class LayerNorm() in tensorlfow?

class ActorNetwork(object):
	def __init__(self, sess, layer1_size, ob_length, action_space, learning_rate, tau):
		self.sess = sess
		self.ob_length = ob_length
		self.action_space = action_space
		self.layer1_size = layer1_size
		self.learning_rate = learning_rate
		self.tau = tau

		#create actor network
		self.ob_input, self.action_output, self.net, self.is_training = self.create_network(ob_length, action_space)

		#To initialize parameter for target network, first use a soft update
		self.target_update, self.target_net = self.hard_update(self.net)

		#create target actor network
		self.target_ob_input, self.target_action_output, self.target_is_training = self.create_target_network(ob_length, action_space)

		#define training rules
		self.create_training_method()

		#initialization
		self.sess.run(tf.initialize_all_variables())

		self.saver = tf.train.Saver()

	def create_training_method(self):
		self.q_gradient_input = tf.placeholder(tf.float32, [None, self.action_space])
		self.parameters_gradients = tf.gradients(self.action_output, self.net, -self.q_gradient_input)
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.parameters_gradients, self.net))

	def create_network(self, ob_length, action_space):
		layer1_size = self.layer1_size

		ob_input = tf.placeholder(tf.float32, [None, ob_length])
		is_training = tf.placeholder(tf.bool)

		W1 = self.variable([ob_length, layer1_size], ob_length)
		b1 = self.variable([layer1_size], ob_length)
		#以下网络参数的初始化是否合理？
		W2 = tf.Variable(tf.random_uniform([layer1_size, action_space], -3e-3, 3e-3))
		b2 = tf.Variable(tf.random_uniform([action_space], -3e-3, 3e-3))

		#layer1 = tf.nn.relu(tf.matmul(ob_input, W1) + b1)
		layer0_bn = self.batch_norm_layer(ob_input, training_phase = is_training, scope_bn = 'batch_norm_0', activation = tf.identity)
		layer1 = tf.matmul(layer0_bn, W1) + b1
		layer1_bn = self.batch_norm_layer(layer1, training_phase = is_training, scope_bn = 'batch_norm_1', activation = tf.nn.relu)
		action_output = tf.nn.softmax(tf.matmul(layer1_bn, W2) + b2)

		return ob_input, action_output, [W1, b1, W2, b2], is_training

	def hard_update(self, net):
		ema = tf.train.ExponentialMovingAverage(decay = 0.0)
		self.target_update = ema.apply(net)
		self.target_net = [ema.average(x) for x in net]

	def soft_update(self, net):
		ema = tf.train.ExponentialMovingAverage(decay = 1 - self.tau)
		self.target_update = ema.apply(net)
		self.target_net = [ema.average(x) for x in net]

	def create_target_network(self, ob_length, action_space):
		ob_input = tf.placeholder(tf.float32, [None, ob_length])
		is_training = tf.placeholder(tf.bool)
		updated_net = self.target_net

		layer0_bn = self.batch_norm_layer(ob_input, training_phase = is_training, scope_bn = 'target_batch_norm_0', activation = tf.identity)
		layer1 = tf.matmul(layer0_bn, updated_net[0]) + updated_net[1]
		layer1_bn = self.batch_norm_layer(layer1, training_phase = is_training, scope_bn = 'target_batch_norm_1', activation = tf.nn.relu)
		action_output = tf.nn.softmax(tf.matmul(layer1_bn, updated_net[2]) + updated_net[3])

		return ob_input, action_output, is_training

	def update_target(self):
		self.target_update, self.target_net = self.soft_update(self.net)

	def train(self, q_gradient_batch, ob_batch):
		self.sess.run(self.optimizer, feed_dict = {
			self.q_gradient_input: q_gradient_batch,
			self.ob_input: ob_batch,
			self.is_training: True
			})

	def actions(self, ob_batch):
		return self.sess.run(self.action_output, feed_dict = {
			self.ob_input: ob_batch,
			self.is_training: True
			})

	def action(self, ob):
		return self.sess.run(self.action_output, feed_dict = {
			self.ob_input: [ob],
			self.is_training: False
			})[0]

	def target_actions(self, ob_batch):
		return self.sess.run(self.target_action_output, feed_dict = {
			self.target_ob_input: ob_batch,
			self.target_is_training:True
			})

	def variable(self, shape, f):
		return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))

	def batch_norm_layer(self, x, training_phase, scope_bn, activation = None):
		return tf.cond(training_phase,
			lambda: tf.contrib.layers.batch_norm(x, activation_fn = activation, center = True, scale = True,
				updates_collections = None, is_training = True, reuse = None, scope = scope_bn, decay = 0.9, epsilon = 1e-5),
			lambda: tf.contrib.layers.batch_norm(x, activation_fn = activation, center = True, scale = True,
				updates_collections = None, is_training = False, reuse = True, scope = scope_bn, decay = 0.9, epsilon = 1e-5))

	def load_network(self, dir = "model/ddpg_actor", step = 0):
		name = "ddpg_agent_{}.h5".format(step)
		model_name = os.path.join(dir, name)
		print("load from " + model_name)
		self.saver.restore(self.sess, model_name)

	def save_network(self, dir = "model/ddpg_actor", step = 0):
		name = "ddpg_agent_{}.h5".format(step)
		model_name = os.path.join(dir, name)
		self.saver.save(self.sess, model_name)

class CriticNetwork(object):
	def __init__(self, sess, layer1_size, ob_length, action_space, learning_rate, l2, tau):
		self.sess = sess
		self.layer1_size = layer1_size
		self.learning_rate = learning_rate
		self.tau = tau
		self.l2 = l2

		#create q network
		self.ob_input, self.action_input, self.q_value_output, self.net, self.is_training = self.create_q_network(ob_length, action_space)

		self.target_update, self.target_net = self.hard_update(self.net)

		#create target q network 
		self.target_ob_input, self.target_action_input, self.target_q_value_output, self.target_is_training = self.create_target_q_network(ob_length, action_space)

		self.create_training_method()

		#initialization
		self.sess.run(tf.initialize_all_variables())

		self.saver = tf.train.Saver()

	def create_training_method(self):
		self.y_input = tf.placeholder(tf.float32, [None, 1])
		weight_decay = tf.add_n([self.l2 * tf.nn.l2_loss(var) for var in self.net])

		self.cost = tf.reduce_mean(tf.square(self.y_input - self.q_value_output)) + weight_decay
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
		self.action_gradients = tf.gradients(self.q_value_output, self.action_input)

	def create_q_network(self, ob_length, action_space):
		layer1_size = self.layer1_size

		ob_input = tf.placeholder(tf.float32, [None, ob_length])
		action_input = tf.placeholder(tf.float32, [None, action_space])
		is_training = tf.placeholder(tf.bool)

		W1 = self.variable([ob_length, layer1_size], ob_length + action_space)
		W1_action = self.variable([action_space, layer1_size], ob_length + action_space)
		b1 = self.variable([layer1_size], ob_length + action_space)
		W2 = tf.Variable(tf.random_uniform([layer1_size, 1], -3e-3, 3e-3))
		b2 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))

		#Insert actions
		layer0_bn = self.batch_norm_layer(ob_input, training_phase = is_training, scope_bn = 'q_batch_norm_0', activation = tf.identity)
		layer1 = tf.nn.relu(tf.matmul(layer0_bn, W1) + tf.matmul(action_input, W1_action) + b1)
		q_value_output = tf.identity(tf.matmul(layer1, W2) + b2)

		return ob_input, action_input, q_value_output, [W1, W1_action, b1, W2, b2], is_training

	def hard_update(self, net):
		ema = tf.train.ExponentialMovingAverage(decay = 0.0)
		self.target_update = ema.apply(net)
		self.target_net = [ema.average(x) for x in net]

	def soft_update(self, net):
		ema = tf.train.ExponentialMovingAverage(decay = 1 - self.tau)
		self.target_update = ema.apply(net)
		self.target_net = [ema.average(x) for x in net]

	def create_target_q_network(self, ob_length, action_space):
		ob_input = tf.placeholder(tf.float32, [None, ob_length])
		action_input = tf.placeholder(tf.float32, [None, action_space])
		is_training = tf.placeholder(tf.bool)
		
		updated_net = self.target_net

		layer0_bn = self.batch_norm_layer(ob_input, training_phase = is_training, scope_bn = 'target_q_batch_norm_0', activation = tf.identity)
		layer1 = tf.nn.relu(tf.matmul(layer0_bn, updated_net[0]) + tf.matmul(action_input, updated_net[1]) + updated_net[2])
		q_value_output = tf.identity(tf.matmul(layer1, updated_net[3]) + updated_net[4])

		return ob_input, action_input, q_value_output, is_training

	def update_target(self):
		self.target_update, self.target_net = self.soft_update(self.net)

	def train(self, y_batch, ob_batch, action_batch):
		self.sess.run(self.optimizer, feed_dict = {
			self.y_input: y_batch,
			self.ob_input: ob_batch,
			self.action_input: action_batch,
			self.is_training: True
			})

	def gradients(self, ob_batch, action_batch):
		return self.sess.run(self.action_gradients, feed_dict = {
			self.ob_input: ob_batch,
			self.action_input: action_batch,
			self.is_training: False
			})[0]

	def target_q(self, ob_batch, action_batch):
		return self.sess.run(self.target_q_value_output, feed_dict = {
			self.target_ob_input: ob_batch,
			self.target_action_input: action_batch,
			self.target_is_training:False
			})

	def q_value(self, ob_batch, action_batch):
		return self.sess.run(self.q_value_output, feed_dict = {
			self.ob_input: ob_batch,
			self.action_input: action_batch,
			self.is_training: False
			})

	def variable(self, shape, f):
		return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))

	def batch_norm_layer(self, x, training_phase, scope_bn, activation = None):
		return tf.cond(training_phase,
			lambda: tf.contrib.layers.batch_norm(x, activation_fn = activation, center = True, scale = True,
				updates_collections = None, is_training = True, reuse = None, scope = scope_bn, decay = 0.9, epsilon = 1e-5),
			lambda: tf.contrib.layers.batch_norm(x, activation_fn = activation, center = True, scale = True,
				updates_collections = None, is_training = False, reuse = True, scope = scope_bn, decay = 0.9, epsilon = 1e-5))

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
		self.memory = list()

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
			del(self.memory[0])
			self.memory.append(experience)

	def count(self):
		return self.num_experiences

	def erase(self):
		self.memory = list()
		self.num_experiences = 0

class OUNoise(object):
	def __init__(self, action_space, mu, sigma, theta, dt, x0 = None):
		self.action_space = action_space
		self.mu = mu
		self.sigma = sigma
		self.theta = theta
		self.ob = np.ones(self.action_space) * self.mu
		self.dt = dt
		self.x0 = x0
		self.reset()

	def reset(self):
		self.ob = self.x0 if self.x0 is not None else np.ones(self.action_space) * self.mu

	def noise(self):
		x = self.ob
		dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size = self.action_space)
		self.ob = x + dx
		return ob

	#print out the value of mu, sigma and theta
	def repr(self):
		return 'OrnsteinUhlenbeckActionNoise(mu = {}, sigma = {}, theta = {})'.format(self.mu, self.sigma, self.theta)

#先定义扰动和非扰动动作空间策略的距离，然后根据这个距离是否高于指定阈值（desired_action_stddev）来调节参数噪声的大小（current_stddev）
class AdaptiveParamNoiseSpec(object):
	def __init__(self, initial_stddev, desired_action_stddev, adoption_coefficient):
		self.initial_stddev = initial_stddev
		self.desired_action_stddev = desired_action_stddev
		self.adoption_coefficient = adoption_coefficient

		self.current_stddev = initial_stddev

	def adapt(self, distance):
		if distance > self.desired_action_stddev:
			#Decrease stddev
			self.current_stddev /= self.adoption_coefficient
		else:
			#Increase srddev
			self.current_stddev *= self.adoption_coefficient

	def get_stats(self):
		stats = {'param_noise_stddev': self.current_stddev}
		return stats

	def repr(self):
		return 'AdaptiveParamNoiseSpec(initial_stddev = {}, desired_action_stddev = {}, adoption_coefficient = {})'.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)

class DDPGAgent(object):
	def __init__(self):
		self.now_phase = {}
		self.green_sec = 30
		self.red_sec = 5
		self.last_change_steo = {}
		self.agent_list = []
		self.phase_passablelane = {}
		self.max_phase = 8

		self.memory_size = 20
		self.learning_start = 20
		self.update_model_freq = 1
		#self.hard_update_target_model_freq = 20

		self.batch_size = 10
		self.gamma = 0.9

		self.ob_length = 24
		self.action_space = 8

		#For actor net
		self.a_layer1_size = 24
		self.a_learning_rate = 0.0001
		self.a_tau = 0.01

		#For critic net
		self.c_layer1_size = 24
		self.c_learning_rate = 0.001
		self.c_tau = 0.01
		self.l2 = 0.01

		#Episodes with ounoise
		self.ou_noise_max_ep = 15
		self.para_noise_max_ep = 15
		#Max episode to use sample from softmax rather than argmax
		self.explore_max_ep = 15

		#For OUNoise
		self.mu = 0.
		self.sigma = 0.2
		self.theta = 0.15
		self.dt = 0.01

		#For AdaptParamNoiseSpec
		self.initial_stddev = 0.1
		self.desired_action_stddev = 0.1
		self.adoption_coefficient = 1.01

		#Frequency of updating param noise's stddev
		#Update once an episode?
		self.update_param_fq = 5

		self.sess = tf.InteractiveSession()

		self.actor_network = ActorNetwork(self.sess, self.a_layer1_size, self.ob_length, self.action_space, self.a_learning_rate, self.a_tau)
		self.actor_perturbed_network = ActorNetwork(self.sess, self.a_layer1_size, self.ob_length, self.action_space, self.a_learning_rate, self.a_tau)
		self.critic_network = CriticNetwork(self.sess, self.c_layer1_size, self.ob_length, self.action_space, self.c_learning_rate, self.l2, self.c_tau)

		#Initialize Memory Buffer
		self.memory = Memory(self.memory_size)

		#Initialize a random process OUNoise for action exploration
		self.ou_explore = OUNoise(self.action_space, self.mu, self.sigma, self.theta, self.dt, None)

		#Initialize a random process AdaptiveParamNoiseSpec for action exploration
		self.ap_explore = AdaptiveParamNoiseSpec(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)

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

		action_batch = np.resize(action_batch, [self.batch_size, self.action_space])

		#Calculate y_batch
		next_action_batch = self.actor_network.target_actions(next_ob_batch)
		q_value_batch = self.critic_network.target_q(next_ob_batch, next_action_batch)
		y_batch = []

		#To store Q values and losses
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
			q_sum += target
		target_f = self.critic_network.q_value(ob_batch, action_batch)
		critic_loss_sum = np.sum(np.square(y_batch, target_fr))

		y_batch = np.resize(y_batch, [self.batch_size, 1])
		#Train critic network
		self.critic_network.train(y_batch, ob_batch, action_batch)

		##
		avg_q = q_sum / num
		try:
			avg_q = str(avg_q[0])
		except:
			avg_q = str(avg_q)
		print("Average Q value is {}".format(avg_q))
		with open("q_ddpg.txt", 'a', encoding = 'utf-8') as f:
			f.write(avg_q + "\n")
		f.close()

		avg_critic_loss = critic_loss_sum / num
		try:
			avg_critic_loss = str(avg_critic_loss[0])
		except:
			avg_critic_loss = str(avg_critic_loss)
		print("Average Critic loss is {}".format(avg_critic_loss))
		with open("critic_loss_ddpg.txt", 'a', encoding = 'utf-8') as x:
			x.write(avg_critic_loss + "\n")
		x.close()
		##

		#Train actor network
		action_batch_for_gradients = self.actor_network.actions(ob_batch)
		q_gradient_batch = self.critic_network.gradients(ob_batch, action_batch_for_gradients)

		self.actor_network.train(q_gradient_batch, ob_batch)

		#Soft update target networks
		self.actor_network.update_target()
		self.critic_network.update_target()

	#To add the parameter noise to actor network
	def perturb_actor_parameters(self, param_noise):
		#Apply parameter noise to actor model for exploration
		net = self.actor_network.net
		target_net = self.actor_network.target_net

		#Copy param of actor to both net and target net of actor_perturbed
		self.actor_perturbed_network.net = net
		self.actor_perturbed_network.target_net = target_net
		new_net = []
		for para in net:
			#Here, how to get the shape of parameter W and b? tf.shape()?
			para += tf.random_normal(tf.random_normal(tf.shape(para))) * self.ap_explore.current_stddev
			new_net.append(para)
		#Update actor_perturbed network by adding param noise
		self.actor_perturbed_network.net = new_net

	#To compute distance between action by actor_perturbed and action by actor
	#In order to adjust stddev of param noise
	def distance_metric(self, action1, action2):
		#Compute distance between actions taken by two policies at the sane states
		#Expects data with structure of numpy arrays
		diff = action1 - action2
		mean_diff = np.mean(np.square(diff), axis = 0)
		dist = sqrt(np.mean(mean_diff))
		return dist

	def remember(self, ob, action, reward, next_ob, done):
		self.memory.reme(ob, action, reward, next_ob, done)

	#For train() function in train_ddpg.py
	def act_(self, observations_for_agent, episode):
		actions = {}
		actions_probs = {}
		entropys = {}
		is_param = (episode < self.para_noise_max_ep)
		is_ou = (episode < self.ou_noise_max_ep)
		is_explore = (episode < self.explore_max_ep)
		for agent_id in self.agent_list:
			action, action_probs, entropy = self.get_action(observations_for_agent[agent_id]['lane'], is_param, is_ou, is_explore)
			actions[agent_id] = action
			actions_probs[agent_id] = action_space
			entropys[agent_id] = entropy

		return actions, actions_probs, entropys

	def act(self, obs, episode):
		observations = obs['observations']
		info = obs['info']
		actions = {}
		actions_probs = {}
		entropys = {}

		#Get observation
		observations_for_agent = {}
		for key, val in observations.items():
			observations_agent_id = int(key.split('_')[0])
			obsercations_feature = key[key.find('_') + 1:]
			if (observations_agent_id not in observations_for_agent.keys()):
				observations_for_agent[observations_agent_id] = {}
			observations_for_agent[observations_agent_id][obsercations_feature] = val[1:]

		#Get actions
		for agent in self.agent_list:
			actions[agent], actions_probs[agent], entropys[agent] = self.get_action(observations_for_agent[agent]['lane_vehicle_num']) + 1

		return actions, actions_probs, entropys

	def get_action(self, ob, is_param = False, is_ou = False, is_explore = True):
		#If need param noise
		if is_param:
			action = self.actor_perturbed_network.action(ob)
		else:
			action = self.actor_network.action(ob)

		#If need ou noise
		if is_ou:
			action += self.ou_explore.noise()

		entropy = -tf.reduce_mean(tf.reduce_sum(tf.log(action) * action))
		#If need sampling rather than simply argmax
		if is_explore:
			#归一化后取样
			action = action / np.sum(action)
			chooseFrom = np.arange(0, self.action_space)
			return np.random.choice(chooseFrom, size = 1, p = action), action, entropy
		else:
			return np.argmax(action), action, entropy

	def save_model(self, dirs, step):
		self.actor_network.save_network(dirs, step)
		self.critic_network.save_q_network(dirs, step)

	def load_model(self, dirs, step):
		self.actor_network.load_network(dirs, step)
		self.critic_network.load_q_network(dirs, step)			

scenario_dirs = [
	"test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
	agent_specs[k] = DDPGAgent()











