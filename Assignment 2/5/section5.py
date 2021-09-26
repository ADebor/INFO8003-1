import random
import argparse
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor

import torch
import torch.nn as nn
import torch.optim as optim

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
plt.rc('text', usetex=True)


"""
INFO8003-1 : Optimal decision making for complex problems
2021 - Assignment 2
Pierre NAVEZ & Antoine DEBOR

SECTION 5 - Parametric-Q-iteration
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


    def expected_return_FQI(self, model):
    	"""
        Compute the expected return derived from a particular model of the FQI algorithm

        Parameters
        ----------
        model: the supervised learning algorithm used in FQI

        Returns
        -------
        The expected return which corresponds to the mean of the expected return over the discretized state space

        """
        val = 0
        for i in range(-8, 9):
            p = i * 0.125
            for j in range(-8, 9):
                s = j * 0.375

                p_next4, s_next4 = domain.dynamics(p, s, 4)
                p_next_4, s_next_4 = domain.dynamics(p,s, -4)
                reward4  = domain.get_reward(p, s, p_next4, s_next4)
                reward_4 = domain.get_reward(p, s, p_next_4, s_next_4)

                if reward4 == 1 or reward_4 == 1:
                    val += 1
                elif reward4 == -1 and reward_4 == 0:
                    val += model.predict([[p,s,-4]])
                elif reward_4 == -1 and reward4 == 0:
                    val += model.predict([[p,s,4]])
                else:
                    val += np.maximum(model.predict([[p,s,4]]), model.predict([[p,s,-4]]))

        return val/289



    def FQI(self, df, SL_algo, stop_rule, terminal_index_win, terminal_index_loose):
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
        Return the expected return derived from the FQI algorithm

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
        dist = []
        x_axis = []
        k = 0
        aux = df[["p_next", "s_next"]].copy()
        aux["four"] = [4]*len(df)
        aux["minus_four"] = [-4]*len(df)
        while(iterate):
            print("\nTS updating...")
            #update of the TS


            y = df['r'] + self.domain.discount * np.maximum(model.predict(aux[["p_next", "s_next", "four"]]),
                                                    model.predict(aux[["p_next", "s_next", "minus_four"]]))

            for index in terminal_index_win:
                y.loc[index] = 1
            for index in terminal_index_loose:
                y.loc[index] = -1


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
                dist.append(distance)
                print(distance)
                x_axis.append(k)
                if distance < min_dif or k == 49:
                    plt.plot(x_axis,dist)
                    plt.show()
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


        expected_r = self.expected_return_FQI(model)

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
                    grid[row][column]=0
                l += 1
        print("\nGrids finished\n")
        print(grid)
        if self.policy == "random":
            filename = '{}_random_model.sav'.format(SL_algo)
            pickle.dump(model, open(filename, 'wb'))

        #Make the color maps associated to each action
        CS_Q_pos = plt.contourf(p_space, s_space, map[:, :, 0], levels=20, cmap=plt.cm.jet_r)
        plt.xlabel("$p$")
        plt.ylabel("$s$")
        plt.colorbar()
        plt.savefig("CS_Q_pos_lin_distance_random.pdf")
        plt.show()

        CS_Q_neg = plt.contourf(p_space, s_space, map[:, :, 1], levels=20, cmap=plt.cm.jet_r)
        plt.xlabel("$p$")
        plt.ylabel("$s$")
        plt.colorbar()
        plt.savefig("CS_Q_neg_lin_distance_random.pdf")
        plt.show()
        #Make the policy color map
        vmin = -4
        vmax = 4
        CS_policy = plt.pcolormesh(p_space, s_space, grid, vmin=vmin, vmax=vmax, cmap = "bwr")
        plt.xlabel("$p$")
        plt.ylabel("$s$")
        plt.colorbar()
        plt.savefig("CS_policy_lin_distance_random.pdf")
        plt.show()

        return expected_r

    def PQL(self, df):
    	"""
        Parametric Q learning algorithm

        Parameters
        ----------
        df : The dataframe containing the one-step system transition

        Returns
        -------
        Save the plots of the results
        Return the expected return derived from the PQL algorithm

        """
    	
        self.net = nn.Sequential(nn.Linear(3, 5), nn.Tanh(),
                            nn.Linear(5, 5), nn.Tanh(),
                            nn.Linear(5, 1), nn.Sigmoid())

        alpha = .01 # learning rate

        for i in range(len(df)):
            p, s, u, r, p_next, s_next = df.values[i]
            input_t_u = torch.FloatTensor([p, s, u])
            self.net(input_t_u).backward()
            with torch.no_grad():
                for param in self.net.parameters():

                    norm_factor = torch.max(torch.norm(param.grad), torch.FloatTensor([10e-10]))
                    new_val =  param + alpha * self.delta(p, s, u, r, p_next, s_next) * param.grad / norm_factor
                    # Normalized alterniative :
                    #term = self.delta(p, s, u, r, p_next, s_next) * param.grad
                    #norm_factor_bis = norm_factor = torch.max(torch.norm(term), torch.FloatTensor([10e-10]))
                    #new_val = param + alpha * term / norm_factor_bis

                    param.copy_(new_val)


        val = 0
        for i in range(-8, 9):
            p = i * 0.125
            for j in range(-8, 9):
                s = j * 0.375

                #input_t_u = torch.FloatTensor([p, s, u])

                p_next4, s_next4 = domain.dynamics(p, s, 4)
                p_next_4, s_next_4 = domain.dynamics(p,s, -4)
                reward4  = domain.get_reward(p, s, p_next4, s_next4)
                reward_4 = domain.get_reward(p, s, p_next_4, s_next_4)

                if reward4 == 1 or reward_4 == 1:
                    val += 1
                elif reward4 == -1 and reward_4 == 0:
                    val += self.net(torch.FloatTensor([p, s, -4])).detach().numpy()
                elif reward_4 == -1 and reward4 == 0:
                    val += self.net(torch.FloatTensor([p, s, 4])).detach().numpy()
                else:
                    val += np.maximum(self.net(torch.FloatTensor([p, s, 4])).detach().numpy(), self.net(torch.FloatTensor([p, s, -4])).detach().numpy())

        expected_r = val/289


        p_space = np.linspace(-1,1,201, endpoint = True)
        p_list = []
        s_space = np.linspace(3,-3,601, endpoint = True)
        s_list_i = s_space.tolist()
        s_list = []
        for element in p_space:
            list = [element]*601
            p_list.extend(list.copy())
            s_list.extend(s_list_i.copy())
        #grid_df = pd.DataFrame(columns=['p','s', 'four', 'minus_four'],dtype=float)
        grid_df = pd.DataFrame(columns=['p','s'],dtype=float)
        grid_df['p'] = p_list
        grid_df['s'] = s_list
        #grid_df['four'] = four_list
        #grid_df['minus_four'] = minus_four_list

        a = []
        b = []
        for p, s in grid_df[['p', 's']].values:
            input_t_u_acc = torch.FloatTensor([p, s, 4])
            input_t_u_dec = torch.FloatTensor([p, s, -4])
            a.append(self.net(input_t_u_acc))
            b.append(self.net(input_t_u_dec))

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
                    grid[row][column]=0
                l += 1
        print("\nGrids finished\n")
        print(grid)


        #Make the color maps associated to each action
        CS_Q_pos = plt.contourf(p_space, s_space, map[:, :, 0], levels=20, cmap=plt.cm.jet_r)
        plt.xlabel("$p$")
        plt.ylabel("$s$")
        plt.colorbar()
        plt.savefig("CS_Q_pos_NN_parametric.pdf")
        plt.show()

        CS_Q_neg = plt.contourf(p_space, s_space, map[:, :, 1], levels=20, cmap=plt.cm.jet_r)
        plt.xlabel("$p$")
        plt.ylabel("$s$")
        plt.colorbar()
        plt.savefig("CS_Q_neg_NN_parametric.pdf")
        plt.show()

        #Make the policy color map
        CS_policy = plt.contourf(p_space, s_space, grid, levels=[-4, 0, 4], cmap = colors.ListedColormap(['red', 'white','blue']))
        plt.xlabel("$p$")
        plt.ylabel("$s$")
        plt.colorbar()
        plt.savefig("CS_policy_NN_parametric.pdf")
        plt.show()

        return expected_r

    def delta(self, p, s, u, r, p_next, s_next):
    	"""
		Compute the delta of the PQL algorithm

        Parameters
        ----------
        p,s,u,r,p_next, s_next a one step system transition

        Returns
        -------
        A tensor delta which goes into the update equation of the PQL

        """
        input_t_acc = torch.FloatTensor([p_next, s_next, 4]) # decelerate input tensor
        input_t_dec = torch.FloatTensor([p_next, s_next, -4]) # accelerate input tensor
        q_acc = self.net(input_t_acc)
        q_dec = self.net(input_t_dec)

        input_t_u = torch.FloatTensor([p, s, u])
        q_u = self.net(input_t_u).detach().numpy()

        return torch.FloatTensor(r + self.domain.discount * max(q_acc.detach().numpy(), q_dec.detach().numpy()) - q_u)

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

    parser = argparse.ArgumentParser(description="ODMCP - A2 - Section 5")

    parser.add_argument("-SL", "--SL_algorithm", type=str, default="linear_regression",
                        help="SL algorithm used in FQI, in {linear_regression, extra_trees, neural_net}")

    parser.add_argument("-TS", "--trajectory_strategy", type=str, default="random",
                        help="Trajectory generation strategy used in FQI, in {random, epsilon_greedy}")

    parser.add_argument("-SR", "--stopping_rule", type=str, default="fixed_N",
                        help="Stopping rule used in FQI, in {fixed_N, distance}")

    parser.add_argument("-comp", "--comparison", action='store_true',
                        help="Option string to be added to perform the FQI/PQL comparison protocol")

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
        #df = agent.generate_dataset()
        #df.to_pickle("df.pkl")
        df = pd.read_pickle("df.pkl")
        #Retrieve the index of the terminal states:
        temp = df.loc[lambda df: df['r'] > 0,:]
        terminal_index_win = temp.index
        temp = df.loc[lambda df: df['r'] < 0,:]
        terminal_index_loose = temp.index
    else:
        df = agent.generate_dataset()
        temp = df.loc[lambda df: df['r'] > 0,:]
        terminal_index_win = temp.index
        temp = df.loc[lambda df: df['r'] < 0,:]
        terminal_index_loose = temp.index

    e_r = agent.PQL(df)

    # FQI/Parametric Q-Learning comparison protocol
    if args.comparison == True:

        # Initialization
        N_min = 5000
        N_max = 50000
        N_step = 5000
        N = np.arange(N_min, N_max + N_step, N_step)
        expected_return_FQI = []
        expected_return_PQL = []

        # Comparison data generation
        i=1
        for N_i in N:
            print(i)
            aux_df = df.sample(N_i, random_state = 1)
            temp = aux_df.loc[lambda aux_df: aux_df['r'] > 0,:]
            terminal_index_win = temp.index
            temp = aux_df.loc[lambda aux_df: aux_df['r'] < 0,:]
            terminal_index_loose = temp.index
            expected_return_FQI.append(agent.FQI(aux_df, "extra_trees", "fixed_N", terminal_index_win, terminal_index_loose).copy())
            expected_return_PQL.append(agent.PQL(aux_df).copy())
            i+=1

        # Results display
        plt.figure()
        plt.plot(N, expected_return_FQI, label="FQI")
        plt.plot(N, expected_return_PQL, label="PQL")
        plt.legend()
        plt.xlabel("$N$")
        plt.ylabel("$J^{\\hat{\\mu}^{*}_N}_{\\infty}$")
        plt.savefig("FQI_PQL_comparison_1h_5n.pdf")
        plt.show()
