import argparse
import itertools
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)


"""
INFO8003-1 : Optimal decision making for complex problems
2021 - Assignement 1
Pierre NAVEZ & Antoine DEBOR

SECTION 4 - System identification
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
    r = np.zeros([n, m, len(actions)])
    p = np.zeros([n, m, n, m, len(actions)])
    for i in range(N):
        Q_mat_prime = np.zeros([n, m, len(domain.action_space())])
        for x in range(n):
            for y in range(m):
                state = x, y
                for k, action in enumerate(actions):
                    r[x, y, k] = domain.MDP_reward(state, action)
                    sum = 0
                    for state_prime in state_space:
                        x_prime, y_prime = state_prime
                        p[x, y, x_prime, y_prime, k] = domain.MDP_proba(state, state_prime, action)
                        sum += p[x, y, x_prime, y_prime, k] * max(Q_mat[state_prime])
                    Q_mat_prime[x, y, k] = r[x, y, k] + domain.discount_factor * sum
        Q_mat = Q_mat_prime
    return Q_mat, r, p

def state_action_value_function_hat(domain, N, r_hat, p_hat):
    """
    Compute the estimated Q-function from estimated MDP parameters
    ---
    parameters :

    - domain : Domain instance corresponding to the considered domain
    - N : Maximum iterate of the recursive equation defining the state-action value functions
    - r_hat : estimated r(x,u) of the MDP
    - p_hat : estimated p(x'|x,u) of the MDP
    ---
    return :

    - Q_mat_hat : estimated Q(x, u) matrix, for every initial state x and every possible action u
    """

    n, m = domain.state_space()
    actions = domain.action_space()
    state_space = domain.get_state_space_indices()
    state_space = state_space.reshape(np.size(state_space))
    Q_mat_hat = np.zeros([n, m, len(actions)])
    for l in range(N):
        Q_mat_prime = np.zeros([n, m, len(domain.action_space())])
        for x in range(n):
            for y in range(m):
                for k, action in enumerate(actions):
                    sum = 0
                    for state_prime in state_space:
                        x_prime, y_prime = state_prime
                        sum += p_hat[x, y, x_prime, y_prime, k] * max(Q_mat_hat[state_prime])
                    Q_mat_prime[x, y, k] = r_hat[x, y, k] + domain.discount_factor * sum
        Q_mat_hat = Q_mat_prime
    return Q_mat_hat

def derive_best_policy(domain, Q):
    """
    Derives optimal policy from Q-function
    ---
    parameters :

    - domain : Domain instance corresponding to the considered domain
    - Q : Q-function from which to derive the expected return
    ---
    return :

    - best_policy : optimal policy derived from Q
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
    Derives optimal expected return from Q-function
    ---
    parameters :

    - domain : Domain instance corresponding to the considered domain
    - Q : Q-function from which to derive the expected return
    ---
    return :

    - best_return : optimal expected return derived from Q
    """

    n, m = domain.state_space()
    best_return = np.zeros([n, m])
    for x in range(n):
        for y in range(m):
            best_return[x, y] = max(Q[x, y])
    return best_return

def gen_trajectory(domain, traj_len):
    """
    Generates a random trajectory of a certain size in a certain domain
    ---
    parameters :

    - domain : Domain instance corresponding to the considered domain
    - traj_len : size of the trajectory to generate
    ---
    return :

    - liste corresponding to the generated trajectory
    """

    traj = list()
    n, m = domain.state_space()
    x_start = random.randint(0, n-1)
    y_start = random.randint(0, m-1)
    state = x_start, y_start
    for j in range(traj_len):
        actions = domain.action_space()
        action = actions[random.randint(0, 3)]
        state_prime = domain.dynamics(state, action)
        r = domain.get_reward(state_prime)
        traj.append((state, action, r))
        state = state_prime
    return traj

def MDP_hat(domain, traj):
    """
    Estimates the parameters of the MDP of a domain from a trajectory
    ---
    parameters :

    - domain : Domain instance corresponding to the considered domain
    - traj : generated trajectory
    ---
    return :

    - estimated r(x,u) and p(x'|x,u) of the MDP
    """

    n, m = domain.state_space()
    # -- Estimation of r_hat(x,u) --
    r_mat = np.zeros([n, m, len(domain.action_space()), 2])
    for state, action, r in traj:
        x, y = state
        action_idx = domain.action_space().index(action)
        r_mat[x, y, action_idx, 0] += r
        r_mat[x, y, action_idx, 1] += 1
    r_hat = r_mat[:, :, :, 0]/r_mat[:, :, :, 1]
    # Avoid NaN due to division by 0
    r_hat[np.isnan(r_hat)] = 0
    # -- Estimation of p_hat(x'|x, u) --
    p_mat = np.zeros([n, m, n, m, len(domain.action_space())])
    traj = list(pairwise(traj))
    for (state, action, r), (state_prime, _, _) in traj:
        x, y = state
        x_prime, y_prime = state_prime
        action_idx = domain.action_space().index(action)
        p_mat[x, y, x_prime, y_prime, action_idx] += 1
    for x in range(n):
        for y in range(m):
            for u in range(len(domain.action_space())):
                p_mat[x, y, :, :, u] = p_mat[x, y, :, :, u] / r_mat[x, y, u, 1]
    # Avoid NaN due to division by 0
    p_mat[np.isnan(p_mat)] = 0
    p_hat = p_mat
    return r_hat, p_hat

def pairwise(iterable):
    """
    Re-arrange an iterable pairwise
    ---
    parameters :

    - iterable : iterable to re-arrange
    ---
    return :

    - zip corresponding to the pairwise re-arrangement
    """

    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

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

    parser.add_argument("-n_i", "--nb_iterations", type=int, default=1000,
                        help="Number of iterations for the computation of the expected return's approximation")

    #parser.add_argument("-tl", "--trajectory_length", type=int, default=1,
                        #help="Length of the trajectory, 1 by default")

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
    Q_mat, r, p = state_action_value_function(domain, args.nb_iterations)
    # Extraction of the optimal policy and expected return
    best_policy = derive_best_policy(domain, Q_mat)
    best_return = derive_best_expected_return(domain, Q_mat)
    print("\nOptimal policy:")
    print(best_policy)
    print("\nOptimal expected return:")
    print(best_return)
    Q = Q_mat.reshape([100])
    r = r.reshape([100])
    p = p.reshape([2500])
    traj_len_range =  [i for i in np.arange(1,5000,100)]
    traj_amnt = 10
    Q_diff = np.zeros([len(traj_len_range)])
    r_diff = np.zeros([len(traj_len_range)])
    p_diff = np.zeros([len(traj_len_range)])
    # Estimation of the MDP
    for i, length in enumerate(traj_len_range):
        r_hat, p_hat = MDP_hat(domain, gen_trajectory(domain, length))
        Q_hat = state_action_value_function_hat(domain, args.nb_iterations, r_hat, p_hat)
        r_hat = r_hat.reshape([100])
        p_hat = p_hat.reshape([2500])
        Q_hat = Q_hat.reshape([100])
        r_diff[i] = np.linalg.norm((r_hat-r),ord=np.inf)
        p_diff[i] = np.linalg.norm((p_hat-p),ord=np.inf)
        Q_diff[i] = np.linalg.norm((Q_hat-Q),ord=np.inf)
    plt.plot(traj_len_range, r_diff)
    plt.xlabel("$T$", fontsize=18)
    plt.ylabel("$\\|\\hat{r}-r\\|_{\\infty}$", fontsize=18)
    plt.show()
    plt.plot(traj_len_range, p_diff)
    plt.xlabel("$T$", fontsize=18)
    plt.ylabel("$\\|\\hat{p}-p\\|_{\\infty}$", fontsize=18)
    plt.show()
    plt.plot(traj_len_range, Q_diff)
    plt.xlabel("$T$", fontsize=18)
    plt.ylabel("$\\|\\hat{Q}-Q\\|_{\\infty}$", fontsize=18)
    plt.show()
    mu_hat_opt = derive_best_policy(domain, Q_hat.reshape([5, 5, 4]))
    print("mu_hat_opt : \n{}".format(mu_hat_opt))
    J_mu_hat_opt = derive_best_expected_return(domain, Q_hat.reshape([5, 5, 4]))
    print("J_mu_hat_opt ; \n{}".format(J_mu_hat_opt))
