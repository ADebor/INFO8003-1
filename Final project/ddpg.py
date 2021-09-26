import gym
import pybulletgym
import random
import math 
from collections import deque
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


class ReplayMemory():
	"""
	Datastructure that contains the agent's experience
	"""
	def __init__(self):
		self.replay_memory = deque(maxlen = 1000000)
		self.batch_size = 64

	def __len__(self):
		return(len(self.replay_memory))


	def store_experience(self, state, action, next_state, reward, done):
		"""
		Apppend a transition of the agent's trajectory
		"""
		self.replay_memory.append((state, action, next_state, reward, done))


	def sample_batch(self):
		"""
		Sample a batch of experiences in order to train the agent
		"""
		batch_size = 64
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


class OUActionNoise:
	"""
	#This class is implemented by https://keras.io/examples/rl/ddpg_pendulum/
	It is used to add exploration noise during training
	"""
	def __init__(self, mean, std=0.15, theta=0.15, dt=0.01, x_initial=None):
		self.theta = theta
		self.mean = mean
		self.std_dev = std
		self.dt = dt
		self.x_initial = x_initial
		self.reset()

	def __call__(self):
		x = (
		    self.x_prev
		    + self.theta * (self.mean - self.x_prev) * self.dt
		    + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
		)
		# Store x into x_prev
		# Makes next noise dependent on current one (Temporal correlation)
		self.x_prev = x
		return x

	def reset(self):
		if self.x_initial is not None:
		    self.x_prev = self.x_initial
		else:
		    self.x_prev = np.zeros_like(self.mean)


class Critic(nn.Module):
	def __init__(self,n_obs,n_actions, lr):
		super(Critic, self).__init__()
		self.input_dim = n_obs
		self.n_actions = n_actions
		#The number of neurons is set according to the DDPG paper
		self.fc1 = nn.Linear(self.input_dim, 400)
		#Now one can initialize the weights and bias to a restricted interval to speed up convergence 
		born = 1./np.sqrt(self.fc1.weight.data.size()[0])
		T.nn.init.uniform_(self.fc1.weight.data,-born,born)
		T.nn.init.uniform_(self.fc1.bias.data,-born,born)
		#Batch normalization layer
		self.bn1 = nn.LayerNorm(400)
		#The number of neurons is still set according to the DDPG paper (ADD a parameter in the config file?)
		self.fc2 = nn.Linear(400,300)
		born = 1./np.sqrt(self.fc2.weight.data.size()[0])
		T.nn.init.uniform_(self.fc2.weight.data,-born,born)
		T.nn.init.uniform_(self.fc2.bias.data,-born,born)
		self.bn2 = nn.LayerNorm(300)
		#Actions are not included before the second layer 
		self.fc_a = nn.Linear(self.n_actions,300)
		#The last layer is the Q value 
		self.q = nn.Linear(300,1)
		#According to the paper, the weights and bias are initialized randomly between [-.003,.003]
		T.nn.init.uniform_(self.q.weight.data, -.003, .003)
		T.nn.init.uniform_(self.q.bias.data, -.003, .003)
		self.optimizer = T.optim.Adam(self.parameters(), lr = lr) 

	def forward(self,state, action):
		#At the main input, only the state will go through
		state_val = self.fc1(state)
		state_val = self.bn1(state_val)
		state_val = F.relu(state_val)
		state_val = self.fc2(state_val)
		state_val = self.bn2(state_val)
		#Here we deal with the action value that pass through fc_a then to a relu activation layer
		action_val = F.relu(self.fc_a(action))
		#At the output layer both state_value and action value that passed through the net are added
		q_val = F.relu(T.add(state_val, action_val))
		q_val = self.q(q_val)

		return q_val

class Actor(nn.Module):
	def __init__(self,n_obs, n_actions, lr):
		super(Actor,self).__init__()
		self.input_dim = n_obs
		self.n_actions = n_actions 
		#Again the hidden layer sizes are chosen according to the DDPG paper 
		self.fc1 = nn.Linear(self.input_dim,400)
		#Basically the network is pretty much the same as the Critic but does not take an action as input
		#In the following we'll perform the same operations as in the previous class 
		born = 1./np.sqrt(self.fc1.weight.data.size()[0])
		T.nn.init.uniform_(self.fc1.weight.data,-born,born)
		T.nn.init.uniform_(self.fc1.bias.data,-born,born)
		self.bn1 = nn.LayerNorm(400)
		self.fc2 = nn.Linear(400,300)
		born = 1./np.sqrt(self.fc2.weight.data.size()[0])
		T.nn.init.uniform_(self.fc2.weight.data,-born,born)
		T.nn.init.uniform_(self.fc2.bias.data,-born,born)
		self.bn2 = nn.LayerNorm(300)
		#This network output an action according to the policy he has learned
		self.policy = nn.Linear(300, self.n_actions)
		T.nn.init.uniform_(self.policy.weight.data, -.003, .003)
		T.nn.init.uniform_(self.policy.bias.data, -.003, .003)
		self.optimizer = T.optim.Adam(self.parameters(), lr = lr)



	def forward(self, state):
		state_val = self.fc1(state)
		state_val = self.bn1(state_val)
		state_val = F.relu(state_val)
		state_val = self.fc2(state_val)
		state_val = self.bn2(state_val)
		state_val = F.relu(state_val)
		#The tanh function is convenient to constraint the action value to lies in between [-1,1]
		action_val = T.tanh(self.policy(state_val))
		return action_val


class DDPG_Agent():
	def __init__(self):
		self.replay_buffer = ReplayMemory()
		self.env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
		self.discount = 0.99
		self.lr_a = 0.000025
		self.lr_c = 0.00025
		self.tau = 0.001
		self.n_actions = self.env.action_space.shape[0]
		self.n_obs = self.env.observation_space.shape[0]

		self.actor = Actor(self.n_obs, self.n_actions, lr= self.lr_a)
		self.target_actor = Actor(self.n_obs, self.n_actions, lr = self.lr_a)

		self.critic = Critic(self.n_obs, self.n_actions,lr = self.lr_c)
		self.target_critic = Critic(self.n_obs, self.n_actions, lr = self.lr_c)

		self.noise = OUActionNoise(mean = np.zeros(self.n_actions))

		self.actor_loss = []
		self.critic_loss = []
		self.actor_grad = []
		self.critic_grad = []
		

		#self.env.render()


	def policy(self,state):
		#In order to speed up the computations
		self.actor.eval()
		input_state = T.FloatTensor(state)
		action = self.actor(input_state)
		noisy_action = action + T.FloatTensor(self.noise())
		#Put the actor in train mode back 
		self.actor.train()
		return noisy_action

	def trained_policy(self,state):
		input_state = T.FloatTensor(state)
		action = self.actor(input_state)
		return action.detach().numpy()


	def train(self, batch):
		"""
		Given a batch of experiences this function trains the agent
		"""

		batch_state,batch_action,batch_next_state,batch_reward,batch_done = batch

		#Put the different network in eval mode to save computational ressources
		self.target_actor.eval()
		self.target_critic.eval()
		self.critic.eval()

		target_action = self.target_actor(T.FloatTensor(batch_next_state))
		target_q = self.target_critic(T.FloatTensor(batch_next_state),target_action)
		q = self.critic(T.FloatTensor(batch_state),T.FloatTensor(batch_action))

		y = []

		for i in range(len(batch_state)):
			#The (1-done) factor allows the term to be 0 if it is a terminal state 
			y.append(batch_reward[i] + self.discount*target_q[i]*(1-batch_done[i]))

		#Now we can update the actor and critic parameters 
		self.critic.train()
		 
		self.critic.optimizer.zero_grad()
		y = T.FloatTensor(y).reshape(q.shape)
		critic_loss = F.mse_loss(y,q)
		critic_loss.backward()
		self.critic.optimizer.step()
		self.critic.eval()
		self.critic_loss.append(critic_loss.item())

		self.actor.optimizer.zero_grad()
		action = self.actor(T.FloatTensor(batch_state))
		self.actor.train()
		actor_loss = -self.critic(T.FloatTensor(batch_state),action)
		actor_loss = T.mean(actor_loss)
		actor_loss.backward()
		self.actor.optimizer.step()
		self.actor_loss.append(actor_loss)
		
		#Update the network parameters doing soft target update
		for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
					target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

		for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
			target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)



	def training_loop(self,max_episodes = 2500):
		"""
		That is where the neural net will be trained 
		"""
		expected_return = []
		mean_expected = []
		best = 0

		for episode in range(max_episodes):
			state = self.env.reset()
			self.noise.reset()
			done = False 

			while not done:
				action = self.policy(state).detach().numpy()
				next_state, reward, done, _ = self.env.step([action])
				#Store the current transistion in the replay buffer
				self.replay_buffer.store_experience(state,action,next_state, reward, done)
				state = next_state

				if len(self.replay_buffer) > self.replay_buffer.batch_size:
					batch = self.replay_buffer.sample_batch()
					self.train(batch)
				
			score = self.get_expected_return()
			if best < score:
				path = os.getcwd()
				T.save(agent.actor.state_dict(), path+"\\ddpg_actor_best.sav")
				T.save(agent.target_actor.state_dict(), path+"\\ddpg_target_actor_best.sav")
				best = score
			expected_return.append(score)
			mean_expected.append(np.mean(expected_return[-100:]))
			print('Episode {}/{} and so far the performance is {}'.format(episode+1,max_episodes,score))


		plt.plot(expected_return,'r')
		plt.plot(mean_expected, 'b')
		plt.xlabel("Episode")
		plt.ylabel("Cumulated discounted reward")
		plt.savefig("score_ddpg_2.pdf")


	def get_expected_return(self):
		"""
		The expected return is computed the current policy learned
		"""

		state = self.env.reset()
		done = False
		score = 0
		i = 0

		while not done:
			action = self.trained_policy(state)
			next_state, reward, done, _ = self.env.step([action])
			score += self.discount**i * reward
			i = i +1 
			state = next_state

		return score

	def load_model(self, name1):
		path = os.getcwd()
		self.actor.load_state_dict(T.load(path+name1))
		#self.critic.load_state_dict(T.load(path+name))
		return 

if __name__=="__main__":

	np.random.seed(0)

	agent = DDPG_Agent()

	agent.training_loop()

	path = os.getcwd()
	T.save(agent.actor.state_dict(), path+"\\ddpg_final.sav")
	T.save(agent.critic.state_dict(), path+"\\ddpg_critic_final.sav")










