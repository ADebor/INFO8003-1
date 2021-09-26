import random
import argparse
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
plt.rc('text', usetex=True)


"""
INFO8003-1 : Optimal decision making for complex problems
2021 - Assignment 2
Pierre NAVEZ & Antoine DEBOR

SECTION 4 - Fitted-Q-iteration
"""


class Domain():

    def __init__(self):
        self.int_step = 0.001
        self.discount = 0.95
        self.time_step = 0.1
        return

    def get_actions(self):
        """
        Available actions in the domain

        Returns
        -------
        Value of the action in the domain

        """
        return -4,4

    def get_reward(self, p, s, p_next, s_next):
        """
        Compute the reward of the agent based on its action

        Parameters
        ----------
        p : initial position
        s : initial speed
        p_next : next position after the action
        s_next : next speed afyer the action

        Returns
        -------
        The reward signal from a given position, after performing an action

        """

        if p_next == p and s_next == s and (abs(p_next) > 1 or abs(s_next) > 3):
            return 0
        elif p_next < -1 or abs(s_next) > 3:
            return -1
        elif p_next > 1 and abs(s_next) <= 3:
            return 1
        else:
            return 0

    def dynamics(self, p, s, action):
        """
        Compute the next state after performing an action, based on the Euler
        integration method

        Parameters
        ----------
        p : The initial state
        s : The initial speed
        action : The action performed by the agent 

        Returns
        -------
        p : The next position after performing the action
        s : The next speed after performing the action 

        """
        #number of iteration used for the Euler integration method
        iterations = int(self.time_step/self.int_step)

        for i in range(iterations):
            #handle terminal state
            if abs(p) > 1 or abs(s) > 3:
                return p, s
            else:
                p_prime = p + self.int_step * s #p'=s
                s_prime = s + self.int_step * self.s_derivation(p, s, action)

            p = p_prime
            s = s_prime

        return p, s


    def s_derivation(self, p, s, action):
        """
        Compute the derivatives of the speed based on the hill shape

        Parameters
        ----------
        p : The current position of the agent
        s : The current speed of the agent 
        action : The action performed by the agent 

        Returns
        -------
        The derivative of the speed

        """
        m = 1
        g = 9.81

        first = action/(m*(1+hill_derivatives(p)[0]**2))
        second = g*hill_derivatives(p)[0]/(1+hill_derivatives(p)[0]**2)
        third = ((s**2)*hill_derivatives(p)[0]*hill_derivatives(p)[1])/(1+hill_derivatives(p)[0]**2)
        return first - second - third


class Agent():

    def __init__(self, domain, policy, random_Q):
        self.domain = domain
        self.policy = policy
        self.random_Q_estimate = random_Q

    def action_taken(self, state):
        """
        Action taken by the agent from its policy

        Parameters
        ----------
        state : a tuple (p,s)

        Returns
        -------
        An action derived from its policy

        """
        if self.policy == "random" or random.uniform(0,1) < 0.25:
            actions = self.domain.get_actions()
            index = random.randint(0, 1)
            return actions[index]
        #In that case, the policy is derived from the model 
        else:
            p, s = state
            a = self.random_Q_estimate.predict([[p, s, 4]])[0]
            b = self.random_Q_estimate.predict([[p, s, -4]])[0]
            if a > b:
                return 4
            else:
                return -4

    def domain_exploration(self, n_steps):
        """
        Compute the trajectory of the agent in the domain, following a certain
        policy

        Parameters
        ----------
        n_steps : number of steps in this trajectory

        Returns
        -------
        trajectory : A sequence of tuples defining the trajectory

        """

        trajectory = list()
        p, s = generate_initial_state()

        for i in range(n_steps):
            tup = list()
            action = self.policy()
            p_next, s_next = self.domain.dynamics(p,s,action)
            reward = self.domain.get_reward(p,s,p_next,s_next)
            tup.append((p,s))
            tup.append(action)
            tup.append(reward)
            tup.append((p_next, s_next))
            trajectory.append(tup) # append une copie ?
            p = p_next; s = s_next
        return trajectory

    def compute_q(self, df, SL_algo, stop_rule):
        """
        Fitted Q-iteration algorithm

        Parameters
        ----------
        df : The dataframe containing the one-step system transition
        SL_algo : The supervised learning algorithm used
        stop_rule : Rule for stopping the algorithm (distance, fixed number of
                                                     iterations)

        Returns
        -------
        Save the plots of the results

        """
        print("FQI...")
        #compute Q_1
        print("\nQ_1...")
        X = df[["p", "s", "u"]]
        y = df["r"]
        if SL_algo == "linear_regression":
            model = LinearRegression().fit(X,y)
        elif SL_algo == "extra_trees":
            model = ExtraTreesRegressor(n_estimators=50, n_jobs=-1).fit(X,y)
        elif SL_algo == "neural_net":
            model = MLPRegressor(hidden_layer_sizes=(100,5)).fit(X,y)
        else:
            exit("\nERROR : The SL algorithm name you entered is not valid")
        #stopping rules spec.
        min_dif = 0.00001
        #N = 176 - 1
        N = 49
        iterate = True
        k = 0
        aux = df[["p_next", "s_next"]].copy()
        aux["four"] = [4]*len(df)
        aux["minus_four"] = [-4]*len(df)
        while(iterate):
            print("\nTS updating...")
            #update of the TS
            y += self.domain.discount * np.maximum(model.predict(aux[["p_next", "s_next", "four"]]),
                                                    model.predict(aux[["p_next", "s_next", "minus_four"]]))
            k += 1
            print("\nQ_{}...".format(k))
            #compute Q_N (update of the model)
            if SL_algo == "linear_regression":
                new_model = LinearRegression().fit(X,y)
            elif SL_algo == "extra_trees":
                new_model = ExtraTreesRegressor(n_estimators=50, n_jobs=-1).fit(X,y)
            elif SL_algo == "neural_net":
                new_model = MLPRegressor(hidden_layer_sizes=(100,5), activation="tanh").fit(X,y)
            #stopping rule check
            if stop_rule == "distance":
                #Computation of the distance
                distance = np.inf
                
                p = df['p'].to_numpy()
                p=p.reshape(-1,1)
                s = df['s'].to_numpy()
                s=s.reshape(-1,1)
                action = df['u'].to_numpy()
                action=action.reshape(-1,1)
                X = np.concatenate([p,s,action], axis =1)
                d = sum(((new_model.predict(X)-model.predict(X))**2))
                distance = d/len(df)
                print(distance)
                if distance < min_dif or k == 49:
                    iterate = False

            elif stop_rule == "fixed_N":
                print("N = ", N)
                N -= 1
                if N == 0:
                    print("N = ", N)
                    iterate = False
            else:
                    exit("\nERROR : The stopping rule name you entered is not valid")
            model = new_model

        p_space = np.linspace(-1,1,201, endpoint = True)
        p_list = []
        s_space = np.linspace(3,-3,601, endpoint = True)
        s_list_i = s_space.tolist()
        s_list = []
        four_list = []
        minus_four_list = []
        for element in p_space:
            list = [element]*601
            p_list.extend(list.copy())
            s_list.extend(s_list_i.copy())
            four_list.extend([4]*601)
            minus_four_list.extend([-4]*601)
        grid_df = pd.DataFrame(columns=['p','s', 'four', 'minus_four'],dtype=float)
        grid_df['p'] = p_list
        grid_df['s'] = s_list
        grid_df['four'] = four_list
        grid_df['minus_four'] = minus_four_list

        a = model.predict(grid_df[['p', 's', 'four']])
        b = model.predict(grid_df[['p', 's', 'minus_four']])

        grid = np.zeros([len(s_space),len(p_space)])
        map = np.zeros([len(s_space),len(p_space), 2])

        print("\nBuilding the grids...")
        l = 0
        for column in range(len(p_space)):
            for row in range(len(s_space)):
                map[row, column, 0] = a[l]
                map[row, column, 1] = b[l]
                if a[l] > b[l]:
                    grid[row][column]=4
                elif a[l] < b[l]:
                    grid[row][column]=-4
                else:
                    #print(a[l]==b[l])
                    grid[row][column]=0
                l += 1
        print("\nGrids finished\n")
        if self.policy == "random":
            filename = '{}_random_model.sav'.format(SL_algo)
            pickle.dump(model, open(filename, 'wb'))

        #Make the color maps associated to each action
        CS_Q_pos = plt.contourf(p_space, s_space, map[:, :, 0],levels = 20, cmap="RdBu")
        plt.xlabel("$p$")
        plt.ylabel("$s$")
        plt.colorbar()
        plt.savefig("CS_Q_pos_tree_distance_random.pdf")
        plt.show()

        CS_Q_neg = plt.contourf(p_space, s_space, map[:, :, 1], levels = 20, cmap="RdBu")
        plt.xlabel("$p$")
        plt.ylabel("$s$")
        plt.colorbar()
        plt.savefig("CS_Q_neg_tree_distance_random.pdf")
        plt.show()
        #Make the policy color map
        CS_policy = plt.contourf(p_space, s_space, grid, levels=[-4,0,4], cmap = colors.ListedColormap(['red', 'white','blue']))
        plt.xlabel("$p$")
        plt.ylabel("$s$")
        plt.colorbar()
        plt.savefig("CS_policy_tree_distance_random.pdf")
        plt.show()

        return

    def generate_dataset(self):
        """


        Parameters
        ----------
        domain : Domain of the problem.
        agent : Intelligent agent that evolves in the domain.
        size : size of the generated dataset.

        Returns
        -------
        Dataframe containing the tuples ((x_k, u_k), r_k).

        """
        #create the Dataframe

        df = pd.DataFrame(columns=['p','s','u','r','p_next','s_next'],dtype=float)
        n_episodes = 1000
        j=0

        for i in range(n_episodes):
            print("\nepisode ", i)
            #draw an initial state at random
            p, s = -0.5,0
            iterate = True

            while(iterate):
                #select an action following the generation strategy
                action = self.action_taken((p, s))
                #compute the next state
                p_next, s_next = domain.dynamics(p, s, action)

                if abs(p_next) > 1 or abs(s_next) > 3:
                    iterate = False
                #get the reward for this particular action
                reward = domain.get_reward(p,s,p_next,s_next)
                #add a new line in the dataframe
                df.loc[j] = ([p,s,action,reward, p_next, s_next])
                j += 1
                p = p_next; s = s_next

        return df


def generate_initial_state():
    """ 

    Returns
    -------
    p_0 : The initial position, from a uniform distribution between [-0.1,0.1]
    s_0 : The initial speed

    """
    p_0 = random.uniform(-0.1, 0.1)
    s_0 = 0
    return p_0, s_0

def hill_derivatives(p):
    """

    Parameters
    ----------
    p : position of the car

    Returns
    -------
    (hill'(p), hill''(p))
     A tuple containing the first and second derivative of the function hill(p)
    """

    if p < 0 :
    #hill(p) = p^2 + p
        return (2*p+1,2)

    else:
    #hill(p) = p/sqrt(1+5p^2)
        return (1/(1+5*p**2)**(3/2), -15*p/(1+5*p**2)**(5/2))

def arguments_parsing():
    """
    Argument parser function
    ---
    parameters :

    None
    ---
    return :

    - args : Keyboard passed arguments
    """

    parser = argparse.ArgumentParser(description="ODMCP - A2 - Section 4")

    parser.add_argument("-SL", "--SL_algorithm", type=str, default="linear_regression",
                        help="SL algorithm used in FQI, in {linear_regression, extra_trees, neural_net}")

    parser.add_argument("-TS", "--trajectory_strategy", type=str, default="random",
                        help="Trajectory generation strategy used in FQI, in {random, epsilon_greedy}")

    parser.add_argument("-SR", "--stopping_rule", type=str, default="fixed_N",
                        help="Stopping rule used in FQI, in {fixed_N, distance}")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    random.seed(0)

    args = arguments_parsing()
    domain = Domain()
    if  args.trajectory_strategy == "random":
        random_opt_policy = None
    else:
        if args.SL_algorithm == "linear_regression":
            random_opt_policy = pickle.load(open("{}_random_model.sav".format(args.SL_algorithm), 'rb'))
        elif args.SL_algorithm == "extra_trees":
            random_opt_policy = pickle.load(open("{}_random_model.sav".format(args.SL_algorithm), 'rb'))
        elif args.SL_algorithm == "neural_net":
            random_opt_policy = pickle.load(open("{}_random_model.sav".format(args.SL_algorithm), 'rb'))

    agent = Agent(domain, args.trajectory_strategy, random_opt_policy)

    if args.trajectory_strategy == "random":
        df = agent.generate_dataset()
        df.to_pickle("df.pkl")
        df = pd.read_pickle("df.pkl")
    else:
        df = agent.generate_dataset()
    agent.compute_q(df, args.SL_algorithm, args.stopping_rule)
