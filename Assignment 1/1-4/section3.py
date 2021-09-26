import argparse
import numpy as np
import pandas as pd
import random


"""
INFO8003-1 : Optimal decision making for complex problems
2021 - Assignement 1
Pierre NAVEZ & Antoine DEBOR

SECTION 3 - Optimal policy
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

    def get_state_space_indices(self):
        """
        Build a matrix whose elements are indices of the considered domain's cells
        ---
        parameters :

        None
        ---
        return :

        - indices : matrix whose elements are indices of the considered domain's cells
        """

        n, m = self.state_space()
        indices = np.zeros([n, m], dtype=object)
        for i in range(n):
            for j in range(m):
                indices[i, j] = (i, j)
        return indices

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

    def MDP_proba(self, state, state_prime, action):
        """
        Compute probability p(x'|x, u) defining the structure of the equivalent MDP
        ---
        parameters :

        - state : current state x
        - state_prime : candidate state x'
        - action : performed action u
        ---
        return :

        -  p(x'|x, u)
        """

        x, y = state
        i, j = action
        n, m = self.state_space()
        if self.domain_type=="Deterministic":
            return (1 if state_prime==self.dynamics(state, action) else 0)
        else:
            prob = 0
            if state_prime==(min(max(x+i, 0), n-1), min(max(y+j, 0), m-1)):
                prob += self.stoch_thresh
            if state_prime==(0, 0):
                prob += 1-self.stoch_thresh
            return prob

    def MDP_reward(self, state, action):
        """
        Compute reward r(x, u) defining the structure of the equivalent MDP
        ---
        parameters :

        - state : current state x
        - action : performed action u
        ---
        return :

        -  r(x, u)
        """

        if self.domain_type=="Deterministic":
            return self.reward(state, action)
        else:
            x, y = state
            i, j = action
            n, m = self.state_space()
            state_prime = (min(max(x+i, 0), n-1), min(max(y+j, 0), m-1))
            return self.stoch_thresh * self.get_reward(state_prime) + (1 - self.stoch_thresh) * self.get_reward((0, 0))

def state_action_value_function(domain, N):
    """
    Compute the Q-function
    ---
    parameters :

    - domain : Domain instance corresponding to the considered domain
    - N : Maximum iterate of the recursive equation defining the state-action value functions
    ---
    return :

    - Q_mat : Q(x, u) matrix, for every initial state x and every possible action u
    """

    n, m = domain.state_space()
    actions = domain.action_space()
    state_space = domain.get_state_space_indices()
    state_space = state_space.reshape(np.size(state_space))
    Q_mat = np.zeros([n, m, len(actions)])
    for i in range(N):
        Q_mat_prime = np.zeros([n, m, len(domain.action_space())])
        for x in range(n):
            for y in range(m):
                state = x, y
                for k, action in enumerate(actions):
                    r = domain.MDP_reward(state, action)
                    sum = 0
                    for state_prime in state_space:
                        p = domain.MDP_proba(state, state_prime, action)
                        sum += p * max(Q_mat[state_prime])
                    Q_mat_prime[x, y, k] = r + domain.discount_factor * sum
        Q_mat = Q_mat_prime
    return Q_mat

def derive_best_policy(domain, Q):
    """
    Extract the optimal policy from a Q-function
    ---
    parameters :

    - domain : Domain instance corresponding to the considered domain
    - Q : The Q function upon which extracting the optimal policy
    ---
    return :

    - best_policy : Optimal policy µ*(x) matrix, for each initial state x, extracted from Q
    """

    n, m = domain.state_space()
    actions = domain.action_space()
    best_policy = np.zeros([n, m], dtype=object)
    for x in range(n):
        for y in range(m):
            best_action = actions[np.argmax(Q[x, y])]
            best_policy[x, y] = best_action
    return best_policy

def derive_best_expected_return(domain, Q):
    """
    Extract the optimal expected return from a Q-function
    ---
    parameters :

    - domain : Domain instance corresponding to the considered domain
    - Q : The Q function upon which extracting the optimal expected return
    ---
    return :

    - best_return : Optimal expected return J^µ*(x) matrix, for each initial state x, extracted from Q
    """

    n, m = domain.state_space()
    best_return = np.zeros([n, m])
    for x in range(n):
        for y in range(m):
            best_return[x, y] = max(Q[x, y])
    return best_return

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
    # Computation of the Q-function
    Q_mat = state_action_value_function(domain, args.nb_iterations)
    # Extraction of the optimal policy and expected return
    best_policy = derive_best_policy(domain, Q_mat)
    best_return = derive_best_expected_return(domain, Q_mat)
    print("\nOptimal policy:")
    print(best_policy)
    print("\nOptimal expected return:")
    print(best_return)
