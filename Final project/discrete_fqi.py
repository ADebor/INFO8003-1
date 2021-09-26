import gym
import pybulletgym
import random
import math 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import os 
import pickle


class FQI_discrete():

	def __init__(self, n_actions, dataset_episodes,model = None):
		self.env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
		self.n_actions = n_actions
		self.discrete_actions = np.linspace(-1,1,num=self.n_actions)
		self.num_episodes = dataset_episodes
		self.dataset = pd.read_pickle("dataset_1000.pkl")
		#self.dataset = self.generate_dataset(dataset_episodes)
		self.gamma = 0.99
		self.model = model if model == None else pickle.load(open(model, 'rb'))


	def trained_policy(self, state):
		#This method returns the action associated with the highest Q-value
		Q_array = np.zeros((1, len(self.discrete_actions)))
		for index, action in enumerate(self.discrete_actions):
			X = np.zeros((1, 10))
			X[:,0:9] = state[0],state[1],state[2],state[3],state[4],state[5],state[6],state[7],state[8]
			X[:,9] = action
			Q_array[:, index] = self.model.predict(X)

		return self.discrete_actions[np.argmax(Q_array, axis = 1)]


	def generate_dataset(self,n_episodes):
		#performs n_episodes, and return a dataframe with the one step transitions
		#

		print("Creating dataset...\n")

		num_observations = self.env.observation_space.shape[0]

		df = pd.DataFrame(columns = ['s1','s2','s3','s4','s5','s6','s7','s8','s9','action','reward','next_s1',\
			'next_s2','next_s3','next_s4','next_s5','next_s6','next_s7','next_s8',\
			'next_s9', 'done'], dtype = float)

		j = 0
		for i in range(n_episodes):
			state = self.env.reset()
			self.env.render()
			done = False

			while not done:
				action = self.discrete_actions[randrange(0,len(self.discrete_actions))]
				next_state, reward, done, _ = self.env.step([action])
				df.loc[j] = ([state[0],state[1],state[2],state[3],state[4],state[5],state[6],state[7],state[8], action, reward, next_state[0],\
				next_state[1],next_state[2],next_state[3],next_state[4],next_state[5],next_state[6],next_state[7],next_state[8], int(done)])
				j += 1
				state = next_state

		print("...Dataset created!\nSize = {}".format(len(df)))
	
		return df

	def learning(self):
		#This is were FQI is implemented
		#Iniatialize some arrays for the plots 
		expected_return = []
		mean_expected = []

		X = self.dataset[['s1','s2','s3','s4','s5','s6','s7','s8','s9','action']]
		y = self.dataset[['reward']]

		model  = ExtraTreesRegressor(n_estimators=50, n_jobs=-1).fit(X,np.ravel(y))

		N = 250

		j = 0

		while j < N:

			print("Updating the training set...\n")

			#The Q values are stored in a matrix, for every possible action of the discrete space 

			Q_array = np.zeros((len(self.dataset), len(self.discrete_actions)))

			for index, action in enumerate(self.discrete_actions):
				X_next = np.zeros((len(self.dataset), 10))
				X_next[:,0:9] = self.dataset[['next_s1','next_s2','next_s3','next_s4','next_s5','next_s6',\
				'next_s7','next_s8','next_s9']]
				X_next[:,9] = action
				Q_array[:, index] = model.predict(X_next)

			rewards = self.dataset['reward']
			#The term 1-done allows to not update the terminal state lines 
			y_new = rewards + self.gamma*(1-self.dataset['done'])*np.max(Q_array, axis = 1)

			print("Update finished!\n")

			new_model  = ExtraTreesRegressor(n_estimators=50, n_jobs=-1).fit(X,np.ravel(y_new))
			model = new_model
			self.model = model
			print("Iteration: {}\n".format(j))
			j += 1
			print("Computing the expected return...\n")
			exp = self.compute_expected_return()
			print("Current expected return is: {}\n".format(exp))
			expected_return.append(exp)
			mean_expected.append(np.mean(expected_return[-10:]))
		

		plt.plot(expected_return,'r')
		plt.plot(mean_expected,'b')
		plt.xlabel("Number of iterations")
		plt.ylabel("Cumulated discounted reward")
		plt.savefig("FQI_discrete_{}_iterations.pdf".format(N))
		plt.show()
		



	def compute_expected_return(self):

		score = 0
		state = self.env.reset()
		done = False
		j = 0
		while not done:
			action = self.trained_policy(state)
			next_state, reward, done, _ = self.env.step([action])
			score += self.gamma**j * reward
			j = j+1
			state = next_state
		return score



if __name__ == "__main__":

	agent = FQI_discrete(n_actions = 21, dataset_episodes = 1000)

	agent.learning()

	pickle.dump(agent.model, open('discrete_fqi.sav', 'wb'))