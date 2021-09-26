import argparse
import numpy as np
import pandas as pd
import random


"""
INFO8003-1 : Optimal decision making for complex problems
2021 - Assignement 1
Pierre NAVEZ & Antoine DEBOR

SECTION 2 - Expected return from a policy
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

        state_prime = self.dynamics(state, action)
        return self.get_reward(state_prime)

    def get_reward(self, state):
        """
        Extract the reward corresponding to a given cell from the considered domain
        ---
        parameters :

        - state : considered cell
        ---
        return :

        - reward corresponding to cell state
        """

        x, y = state
        return self.domain_matrix[x,y]

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

def expected_return(domain, N):
    """
    Compute the estimated expected return of policy defined according to function policy(), for N iterations
    ---
    parameters :

    - domain : Domain instance corresponding to the considered domain
    - N : max iterate of the recurrent equation use to estimate the expected return
    ---
    return :

    - expected_mat : expected return matrix, for policy defined in function policy(), for each initial state of the domain
    """

    n, m = domain.state_space()
    expected_mat = np.zeros([n, m])
    i, j = policy(domain)
    for l in range(N):
        expected_mat_prime = np.zeros([n, m])
        for x in range(n):
            for y in range(m):
                state = x, y
                x_prime, y_prime =  min(max(x+i, 0), n-1), min(max(y+j, 0), m-1)
                if domain.domain_type == "Deterministic":
                    r = domain.reward(state, (i, j))
                    expected_mat_prime[x, y] = r + domain.discount_factor * expected_mat[x_prime, y_prime]
                else:
                    r_det = domain.get_reward((x_prime, y_prime))
                    r_stoch = domain.get_reward((0, 0))
                    r = domain.stoch_thresh * r_det + (1-domain.stoch_thresh) * r_stoch
                    expected_mat_prime[x, y] = r + domain.discount_factor * (domain.stoch_thresh * expected_mat[x_prime, y_prime] + (1-domain.stoch_thresh) * expected_mat[0, 0])
        expected_mat = expected_mat_prime
    return expected_mat

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

    parser.add_argument("-stocha", "--stochastic", action='store_true',
                        help="Stochastic character of the domain, option string to be added for stochastic behaviour")

    parser.add_argument("-s_th", "--stochastic_threshold", type=float, default=0.5,
                        help="Stochastic threshold involved in the stochastic dynamics")

    parser.add_argument("-df", "--discount_factor", type=float, default=0.99,
                        help="Discount factor, 0.99 by default")

    parser.add_argument("-f", "--domain_instance_file", type=str, default='instance.csv',
                        help="Filename of the domain instance")

    parser.add_argument("-n_i", "--nb_iterations", type=int, default=3000,
                        help="Number of iterations for the computation of the expected return's approximation")

    args = parser.parse_args()

    if args.stochastic:
        print("\nStochastic domain chosen")
    else:
        print("\nDeterministic domain chosen (default)")

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
    # Computation of the expected return
    expected_mat = expected_return(domain, args.nb_iterations)
    print("\nExpected returns for {} iterations".format(args.nb_iterations))
    print(expected_mat)
