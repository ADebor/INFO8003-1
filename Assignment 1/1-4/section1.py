import argparse

import numpy as np
import pandas as pd
import random


"""
INFO8003-1 : Optimal decision making for complex problems
2021 - Assignement 1
Pierre NAVEZ & Antoine DEBOR

SECTION 1 - Implementation of the domain
"""

class Domain:
    def __init__(self, domain_matrix, domain_type, discount_factor, stoch_thresh):
        self.domain_matrix = domain_matrix
        self.domain_type = domain_type
        self.discount_factor = discount_factor
        self.stoch_thresh = stoch_thresh
        return

    def state_space(self):
        """
        Define dimensions of the considered domain
        ---
        parameters :

        None
        ---
        return :

        - height, width : dimensions of the considered domain
        """

        height = np.shape(self.domain_matrix)[0]
        width = np.shape(self.domain_matrix)[1]
        return height, width

    def action_space(self):
        """
        Define the action space of the considered domain
        ---
        parameters :

        None
        ---
        return :

        - tuple of possible actions
        """

        return ((1, 0), (0, 1), (-1, 0), (0, -1))

    def reward(self, state, action):
        """
        Compute the reward corresponding to a given action from a given state
        ---
        parameters :

        - state : current state
        - action : action performed from state state
        ---
        return :

        - reward corresponding to action action, from state state
        """

        x_prime, y_prime = self.dynamics(state, action)
        return self.domain_matrix[x_prime,y_prime]

    def dynamics(self, state, action):
        """
        Define the dynamics of the considered domain
        ---
        parameters :

        - state : current state
        - action : action performed from state state
        ---
        return :

        - reward corresponding to action action, from state state, according to the considered domain's dynamics
        """

        x, y = state
        i, j = action
        n, m = self.state_space()

        if self.domain_type=="Deterministic":
            return min(max(x+i, 0), n-1), min(max(y+j, 0), m-1)

        elif self.domain_type=="Stochastic":
            if random.random() <= self.stoch_thresh:
                return min(max(x+i, 0), n-1), min(max(y+j, 0), m-1)
            else:
                return 0,0


def policy(domain):
    """
    Defines the studied rule-based stationary policy
    ---
    parameters :

    - domain : Domain instance corresponding to the considered domain
    ---
    return :

    - tuple corresponding to the corresponding action
    """

    actions = domain.action_space()
    # choice of policy : always go right
    return actions[1]
    # alternative choice of policy : always go random (please comment the previous line and uncomment the next one for testing this policy)
    #return random.choice(actions)


def trajectory(domain, initial_state, n_steps):
    """
    Generates a trajectory of a given number of steps starting from a given state, following the policy defined in function policy()
    ---
    parameters :

    - domain : Domain instance corresponding to the considered domain
    ---
    return :

    - None
    """
    print("\nFollowed trajectory for {} steps ({} domain):".format(n_steps, domain.domain_type))
    current = initial_state
    for i in range(n_steps+1):
        u = policy(domain)
        x = domain.dynamics(current, u)
        r = domain.reward(current, u)
        print("(x_{0}, u_{0}, r_{0}, x_{1}) : ({2}, {3}, {4}, {5})".format(i, i+1, current, u, r, x))
        current = x
    return


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

    parser = argparse.ArgumentParser(description="ODMCP - A1 - Section 1")


    #parser.add_argument("-dt", "--domain_type", type=str, choices=["Deterministic", "Stochastic"], default="Deterministic",
    #                    help="Type of the domain in [\"Deterministic\", \"Stochastic\"], \"Deterministic\" by default")

    parser.add_argument("-stocha", "--stochastic", action='store_true',
                        help="Stochastic character of the domain, option string to be added for stochastic behaviour")

    parser.add_argument("-s_th", "--stochastic_threshold", type=float, default=0.5,
                        help="Stochastic threshold involved in the stochastic dynamics")

    parser.add_argument("-df", "--discount_factor", type=float, default=0.99,
                        help="Discount factor, 0.99 by default")

    parser.add_argument("-n", "--nb_steps", type=int, default=10,
                        help="Number of steps upon which to run the simulation, 10 by default")

    parser.add_argument("-s_0", "--initial_state", type=tuple, default=(3,0),
                        help="Initial state of the agent, (3,0) by default")

    parser.add_argument("-f", "--domain_instance_file", type=str, default='instance.csv',
                        help="Filename of the domain instance")

    args = parser.parse_args()

    if args.stochastic:
        print("\nStochastic domain chosen")
    else:
        print("\nDeterministic domain chosen (default)")

    print("\nInitial state of the agent : {}".format(args.initial_state))

    return args

if __name__ == "__main__":
    random.seed(1)

    args = arguments_parsing()
    # Domain instance file reading
    domain_matrix = pd.read_csv(args.domain_instance_file, delimiter=',', header=None).values

    print("\nInstance of the domain:\n{}".format(domain_matrix))
    # Domain instanciation
    domain = Domain(domain_matrix,
        "Stochastic" if args.stochastic else "Deterministic",
        args.discount_factor, args.stochastic_threshold)
    # Generation of a trajectory
    trajectory(domain, args.initial_state, args.nb_steps)
