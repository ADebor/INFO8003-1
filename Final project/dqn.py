
import gym
import pybulletgym
import random
import math
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import pickle

import config

class ReplayMemory():
	"""
	Datastructure that contains the agent's experience
	"""
	def __init__(self):
		self.replay_memory = deque(maxlen = 100000)


	def store_experience(self, state, action, next_state, reward, done):
		"""
		Apppend a transition of the agent's trajectory
		"""
		self.replay_memory.append((state, action, next_state, reward, done))


	def sample_batch(self):
		"""
		Sample a batch of experiences in order to train the agent
		"""
		batch_size = config.BATCH_SIZE
		batch = random.sample(self.replay_memory, k=batch_size) if len(self.replay_memory)>batch_size \
			else self.replay_memory
		batch_state = []
		batch_action = []
		batch_next_state = []
		batch_reward = []
		batch_done = []

		for experience in batch:
			batch_state.append(experience[0])
			batch_action.append(experience[1])
			batch_next_state.append(experience[2])
			batch_reward.append(experience[3])
			batch_done.append(experience[4])

		return np.array(batch_state), np.array(batch_action), np.array(batch_next_state),\
		np.array(batch_reward), np.array(batch_done)


class DQN_network(nn.Module):
	"""
	The neural net model will predict the Q values for all actions given a state
	The input as the shape of the state space and the output has the shape of the action space
	"""
	def __init__(self):
		super(DQN_network, self).__init__()
		self.fc1 = nn.Linear(9, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, 21)


	def forward(self,x):
		x = self.fc1(x)
		x = F.relu(x)
		x = self.fc2(x)
		x = F.relu(x)
		output = self.fc3(x)
		return output


class DQN_Agent():

	def __init__(self):
		self.replay_buffer = ReplayMemory()
		self.env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
		self.q_net = DQN_network()
		self.q_target = DQN_network()
		self.discount = 0.99
		self.lr = 0.001
		#self.env.render()
		self.action_space = np.linspace(-1,1,num=21)
		self.optimizer = torch.optim.Adam(self.q_net.parameters(),lr = self.lr)
		self.epsilon = 1
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.0005
		self.tau = 0.01


	def policy(self, state):
		"""
		Given a state, sample an action following a epsilon greedy policy
		"""
		input_state = torch.FloatTensor(state)
		actions = self.q_target(input_state)
		actions = actions.detach().numpy()
		best_action = np.argmax(actions)
		return best_action

	def epsilon_policy(self,state):
		"""
		Epsilon greedy policy used during learning
		"""
		if np.random.random() < config.EPSILON:
			return random.randrange(config.OUTPUT_SZ)
		return self.policy(state)

	def trained_policy(self, state):
		"""
		Policy to use when the agent is trained, given by its Q network and not its target
		"""
		input_state = torch.FloatTensor(state)
		actions = self.q_net(input_state)
		actions = actions.detach().numpy()
		best_action = np.argmax(actions)
		return self.action_space[best_action]

	def get_experience(self):
		"""
		Collect experience and store it in the buffer
		"""
		state = self.env.reset()
		done = False
		while not done:
			action = self.epsilon_policy(state)
			next_state, reward, done, _ = self.env.step([self.action_space[action]])
			self.replay_buffer.store_experience(state, action, next_state, reward,done)
			state = next_state

	def train(self, batch):
		"""
		Given a batch of experience, train the agent to predict Q values
		"""
		batch_state,batch_action,batch_next_state,batch_reward,batch_done = batch
		q_current = self.q_net(torch.FloatTensor(batch_state)).detach().numpy()
		q_target = np.copy(q_current)
		q_next = self.q_target(torch.FloatTensor(batch_next_state)).detach().numpy()
		q_max_next = np.amax(q_next,axis=1)

		for i in range(batch_state.shape[0]):
			if batch_done[i]:
				q_target[i][batch_action[i]] = batch_reward[i]
			else:
				q_target[i][batch_action[i]] = batch_reward[i] + self.discount*q_max_next[i]

		#Here comes the training of one batch
		loss_fn = nn.MSELoss()
		X = self.q_net(torch.FloatTensor(batch_state))
		loss = loss_fn(X,torch.FloatTensor(q_target))
		#Backpropagation
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min\
			else self.epsilon_min

		return loss.item()

	def training_loop(self,max_epoch = 15000):
		"""
		That is where the neural net will be trained
		"""
		best = 0
		score = []
		mean = []
		for epoch in range(max_epoch):
			self.get_experience()
			batch = self.replay_buffer.sample_batch()
			loss = self.train(batch)
			expected_return = self.get_expected_return(self.env, self)
			print('Episode {}/{} and so far the performance is {} and loss is {}'.format(epoch,max_epoch,expected_return,loss))
			score.append(expected_return)
			mean.append(np.mean(score[-100:]))
			if best < expected_return:
				path = os.getcwd()
				torch.save(agent.q_net.state_dict(), path+"\\dqn_best.sav")
				torch.save(agent.q_target.state_dict(), path+"\\dqn_target_best.sav")
				best = expected_return 
			if epoch % config.TARGET_UPDATE == 0:
				self.q_target.load_state_dict(self.q_net.state_dict())

		#plt.figure()
		x = np.arange(0,max_epoch,1)
		plt.plot(score, 'r')
		plt.plot(mean,'b')
		plt.xlabel("Episode")
		plt.ylabel("Cumulated discounted reward")
		plt.savefig("Discounted_reward_DQN_update{}_{}epochs.pdf".format(10,15000))
		return

	def get_expected_return(self, env, agent):

		state = env.reset()
		done = False
		expected_return = 0
		i = 0

		while not done:
			action = agent.policy(state)
			next_state, reward, done, _ = env.step([agent.action_space[action]])
			expected_return += self.discount**i * reward
			state = next_state
			i = i+1

		return expected_return 


	def load_model(self, name1):
		path = os.getcwd()
		self.q_net.load_state_dict(torch.load(path+name1))
		return 


if __name__=="__main__":

	agent = DQN_Agent()
	agent.training_loop()
	path = os.getcwd()
	torch.save(agent.q_net.state_dict(), path+"\\dqn_final.sav")
	torch.save(agent.q_target.state_dict(), path+"\\dqn_target_final.sav")
	#agent.q_net.load_state_dict(torch.load(path+"\\tmp.sav"))

