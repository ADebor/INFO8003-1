"""
In this file, the policy scores are computed
"""
import gym
import pybulletgym
from dqn import DQN_Agent
from ddpg import DDPG_Agent 
from discrete_fqi import FQI_discrete
from continuous_fqi import FQI_continuous 


def assess_performance(agent):

	agent.env.render()
	state = agent.env.reset()
	done = False
	score = 0
	discounted = 0
	j=0

	while not done:
		action = agent.trained_policy(state)
		next_state, reward, done, _ = agent.env.step([action])
		score += reward 
		discounted += 0.99**j * reward 
		state = next_state
		j = j+1

	return score , discounted


if __name__ == "__main__":
	
	#Load the models 
	fqi_d = FQI_discrete(n_actions = 21, dataset_episodes = 1000, model = "discrete_fqi.sav")
	fqi_c = FQI_continuous(n_actions = 41, dataset_episodes = 1000, model = "continuous_fqi.sav")


	dqn = DQN_Agent()
	dqn.load_model("\\dqn_best.sav")
	dqn.q_net.eval()

	ddpg = DDPG_Agent()
	ddpg.load_model("\\ddpg_actor_best.sav")
	ddpg.actor.eval()

	"""
	score , discounted = assess_performance(dqn)
	print("The DQN's agent cumulated score is {} and the discount score is {}".format(score,discounted))
	"""
	score , discounted = assess_performance(fqi_d)
	print("The FQI_discrete's agent cumulated score is {} and the discount score is {}".format(score,discounted))
	"""
	score , discounted = assess_performance(ddpg)
	print("The DDPG's agent cumulated score is {} and the discount score is {}".format(score,discounted))

	score , discounted = assess_performance(fqi_c)
	print("The FQI_continuous' agent cumulated score is {} and the discount score is {}".format(score,discounted))
	"""





		





