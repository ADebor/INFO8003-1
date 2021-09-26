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

SECTION 5 - Q-Learning in a batch setting
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

def gen_trajectory(domain, traj_len, start_state):
    """
    Generates a random trajectory of a certain size in a certain domain
    ---
    parameters :

    - domain : Domain instance corresponding to the considered domain
    - traj_len : size of the trajectory to generate
    - start_state : initial state
    ---
    return :

    - traj : list corresponding to the generated trajectory
    """

    traj = list()
    n, m = domain.state_space()
    state = start_state
    for j in range(traj_len):
        actions = domain.action_space()
        action = actions[random.randint(0, 3)]
        state_prime = domain.dynamics(state, action)
        r = domain.get_reward(state_prime)
        traj.append((state, action, r))
        state = state_prime
    return traj

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

    parser.add_argument("-lr", "--learning_rate", type=float, default=0.05,
                        help="Constant learning rate used in the Q-learning algorithm, 0.05 by default")

    args = parser.parse_args()

    if args.stochastic:
        print("\nStochastic domain chosen")
    else:
        print("\nDeterministic domain chosen (default)")

    return args

def offline_Q_learning(domain, trajectory, alpha):
    """
    Offline Q-Learning implementation
    ---
    parameters :

    - domain : Domain instance corresponding to the considered domain
    - trajectory : trajectory from which to perform the algorithm
    - alpha : learning rate
    ---
    return :

    - Q_hat : estimated Q-function computed with offline Q-Learning
    """

    n, m = domain.state_space()
    actions = domain.action_space()
    Q_hat = np.zeros([n, m, len(actions)])
    for k, (state, action, r) in enumerate(trajectory):
        if k == len(trajectory)-1:
            return Q_hat
        u = actions.index(action)
        x, y = state
        x_prime, y_prime = trajectory[k+1][0]
        Q_hat[x, y, u] = (1 - alpha) * Q_hat[x, y, u] + alpha * (r + domain.discount_factor * max(Q_hat[x_prime, y_prime]))

class Intelligent_agent:

    def __init__(self, domain, state_0):
        self.domain = domain
        self.state_0 = state_0
        self.policy = None

    def select_action(self, Q, state, epsilon):
        """
        Selects an action from a certain state, following an epsilon-greedy policy acc. to Q
        ---
        parameters :

        - Q : Q-function from which to derive the optimal policy
        - state : current state
        - epsilon : exploration rate
        ---
        return :

        - action to take
        """

        if random.random() < epsilon:
            actions = self.domain.action_space()
            action = actions[random.randint(0,3)]
            return action
        else:
            self.policy = self.derive_best_policy(Q)
            x, y = state
            return self.policy[x, y]

    def derive_best_policy(self, Q):
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

        n, m = self.domain.state_space()
        actions = self.domain.action_space()
        best_policy = np.zeros([n, m], dtype=object)
        for x in range(n):
            for y in range(m):
                best_action = actions[np.argmax(Q[x, y])]
                best_policy[x, y] = best_action
        return best_policy

    def gen_trajectory_epsilon_greedy(self, traj_len, start_state, epsilon, Q):
        """
        Generates a trajectory following an epsilon-greedy policy acc. to Q
        ---
        parameters :

        - traj_len : size of the trajectory to generate
        - start_state : initial state
        - epsilon : exploration rate
        - Q : Q-function from which to derive the expected return
        ---
        return :

        - traj : list corresponding to the generated trajectory
        """

        traj = list()
        n, m = self.domain.state_space()
        state = start_state
        for j in range(traj_len):
            # Choose action using epsilon-greedy policy derived from Q
            action = self.select_action(Q, state, epsilon)
            state_prime = self.domain.dynamics(state, action)
            r = self.domain.get_reward(state_prime)
            traj.append((state, action, r))
            state = state_prime
        return traj

    def online_Q_learning_first(self, alpha, epsilon, n_episodes, n_transitions):
        """
        Implements the first protocol
        ---
        parameters :

        - alpha : learning rate
        - epsilon_0 : initial exploration rate
        - n_episodes : number of episodes
        - n_transitions : number of transitions per episode
        ---
        return :

        - Q_vec : list of derived optimal expected returns (one for each episode)
        """

        # Online epsilon-greedy Q-learning algorithm - Protocol 1
        n, m = self.domain.state_space()
        actions = self.domain.action_space()
        Q_vec = []
        # Initialisation
        Q_hat = np.zeros([n, m, len(self.domain.action_space())])
        # Generate first random trajectory starting at (3,0) (no specific policy at this stage)
        trajectory = gen_trajectory(self.domain, n_transitions+1, self.state_0)
        # For each transition in each episode, update of Q_hat
        for e in range(n_episodes):
            for k, (state, action, reward) in enumerate(trajectory):
                if(k!=n_transitions):
                    x, y = state
                    u = actions.index(action)
                    x_prime, y_prime = trajectory[k+1][0]
                    Q_hat[x, y, u] = (1 - alpha) * Q_hat[x, y, u] + alpha * (reward + self.domain.discount_factor * max(Q_hat[x_prime, y_prime]))
            # Generate new trajectory starting at (3,0) following an epsilon-greedy policy
            trajectory = self.gen_trajectory_epsilon_greedy(n_transitions+1, self.state_0, epsilon, Q_hat)
            J_hat = derive_best_expected_return(self.domain, Q_hat)
            Q_vec.append(J_hat.copy())
        return Q_vec

    def online_Q_learning_second(self, alpha_0, epsilon, n_episodes, n_transitions):
        """
        Implements the second protocol
        ---
        parameters :

        - alpha_0 : initial learning rate
        - epsilon : exploration rate
        - n_episodes : number of episodes
        - n_transitions : number of transitions per episode
        ---
        return :

        - Q_vec : list of derived optimal expected returns (one for each episode)
        """

        # Online epsilon-greedy Q-learning algorithm - Protocol 1
        n, m = self.domain.state_space()
        actions = self.domain.action_space()
        Q_vec = []
        # Initialisation
        Q_hat = np.zeros([n, m, len(self.domain.action_space())])
        # Generate first random trajectory starting at (3,0) (no specific policy at this stage)
        trajectory = gen_trajectory(self.domain, n_transitions+1, self.state_0)
        # For each transition in each episode, update of Q_hat
        for e in range(n_episodes):
            alpha = alpha_0
            for k, (state, action, reward) in enumerate(trajectory):
                if(k!=n_transitions):
                    x, y = state
                    u = actions.index(action)
                    x_prime, y_prime = trajectory[k+1][0]
                    Q_hat[x, y, u] = (1 - alpha) * Q_hat[x, y, u] + alpha * (reward + self.domain.discount_factor * max(Q_hat[x_prime, y_prime]))
                    alpha = 0.8*alpha
            # Generate new trajectory starting at (3,0) following an epsilon-greedy policy
            trajectory = self.gen_trajectory_epsilon_greedy(n_transitions+1, self.state_0, epsilon, Q_hat)
            J_hat = derive_best_expected_return(self.domain, Q_hat)
            Q_vec.append(J_hat.copy())
        return Q_vec

    def online_Q_learning_third(self, alpha, epsilon, n_episodes, n_transitions, n_replay):
        """
        Implements the third protocol
        ---
        parameters :

        - alpha : learning rate
        - epsilon : exploration rate
        - n_episodes : number of episodes
        - n_transitions : number of transitions per episode
        - n_replay : number of experience replays per transition
        ---
        return :

        - Q_vec : list of derived optimal expected returns (one for each episode)
        """

        # Online epsilon-greedy Q-learning algorithm - Protocol 3
        n, m = self.domain.state_space()
        actions = self.domain.action_space()
        Q_vec = []
        # Initialisation
        Q_hat = np.zeros([n, m, len(self.domain.action_space())])
        # Generate first random trajectory starting at (3,0) (no specific policy at this stage)
        trajectory = gen_trajectory(self.domain, n_transitions+1, self.state_0)
        # For each transition in each episode, update of Q_hat
        for e in range(n_episodes):
            for t in range(n_transitions):
                # Experience replay
                for i in range(n_replay):
                    # Draw randomly an action from the buffer
                    traj_index = random.randint(0, n_transitions-1)
                    state_replay, action_replay, reward_replay = trajectory[traj_index]
                    state_prime_replay = trajectory[traj_index+1][0]
                    u = actions.index(action_replay)
                    x, y = state_replay
                    x_prime, y_prime = state_prime_replay
                    # Update Q_hat
                    Q_hat[x, y, u] = (1 - alpha) * Q_hat[x, y, u] + alpha * (reward_replay + self.domain.discount_factor * max(Q_hat[x_prime, y_prime]))
            # Generate new trajectory starting at (3,0) following an epsilon-greedy policy
            trajectory = self.gen_trajectory_epsilon_greedy(n_transitions+1, self.state_0, epsilon, Q_hat)
            J_hat = derive_best_expected_return(self.domain, Q_hat)
            Q_vec.append(J_hat.copy())
        return Q_vec


if __name__ == "__main__":
    random.seed(1)

    args = arguments_parsing()

    domain_matrix = pd.read_csv(args.domain_instance_file, delimiter=',', header=None).values

    print("\nInstance of the domain:\n{}".format(domain_matrix))
    domain = Domain(domain_matrix,
        "Stochastic" if args.stochastic else "Deterministic",
        args.discount_factor, args.stochastic_threshold)
    n, m = domain.state_space()

    # --OFFLINE Q-LEARNING--
    print("\nOFFLINE Q-LEARNING...")
    T = [pow(10, i) for i in [2, 3, 4, 5, 6]]
    T.append(5*10**6)
    J_diff = np.zeros([len(T)])
    alpha = args.learning_rate
    print("\nDeriving the optimal expected return...")

    Q, _, _ = state_action_value_function(domain, args.nb_iterations)
    mu_opt = derive_best_policy(domain, Q)
    J_mu_opt = derive_best_expected_return(domain, Q)
    print("\nOptimal expected return : \n{}".format(J_mu_opt))
    print("\nQ-learning algorithm running...")
    for i, t in enumerate(T):
        x_0 = random.randint(0, n-1)
        y_0 = random.randint(0, m-1)
        s_0 =x_0, y_0
        trajectory = gen_trajectory(domain, t, s_0)
        Q_hat = offline_Q_learning(domain, trajectory, alpha)
        mu_hat_opt = derive_best_policy(domain, Q_hat)
        J_mu_hat_opt = derive_best_expected_return(domain, Q_hat)
        diff = J_mu_hat_opt-J_mu_opt
        J_diff[i] = np.linalg.norm(diff.reshape(25),ord=np.inf)
    print(J_diff)
    plt.plot(T, J_diff)
    plt.xscale('log')
    plt.xlabel("$T$", fontsize=18)
    plt.ylabel("$\\|J^N_{\\hat{\\mu}^*}-J^N_{\\mu^*}\\|_{\\infty}$", fontsize=18)
    plt.savefig("J_mu_hat_opt_convergence_speed.pdf")
    plt.show()

    print("\nOptimal policy, for length = {} : \n{}".format(t, mu_hat_opt))
    print("\nOptimal return, for length = {} : \n{}".format(t, J_mu_hat_opt))

    # --ONLINE Q-LEARNING--
    print("\nONLINE Q-LEARNING...")
    alpha = 0.05
    epsilon = 0.25
    n_episodes = 100
    n_transitions = 1000
    n_replay = 10
    state_0 = (3,0)
    agent = Intelligent_agent(domain, state_0)

    # First protocol
    print("\n-- First protocol -- \n")
    Q = agent.online_Q_learning_first(alpha, epsilon, n_episodes, n_transitions)
    Q_diff_1 = np.zeros([n_episodes])
    for i, q in enumerate(Q):
        Q_diff_1[i] = np.linalg.norm(np.ravel(q-J_mu_opt),ord=np.inf)

    # Second protocol
    print("\n-- Second protocol -- \n")
    Q = agent.online_Q_learning_second(alpha, epsilon, n_episodes, n_transitions)
    Q_diff_2 = np.zeros([n_episodes])
    for i, q in enumerate(Q):
        Q_diff_2[i] = np.linalg.norm(np.ravel(q-J_mu_opt),ord=np.inf)

    # Third protocol
    print("\n-- Third protocol -- \n")
    Q = agent.online_Q_learning_third(alpha, epsilon, n_episodes, n_transitions, n_replay)
    Q_diff_3 = np.zeros([n_episodes])
    for i, q in enumerate(Q):
        Q_diff_3[i] = np.linalg.norm(np.ravel(q-J_mu_opt),ord=np.inf)

    # Comparison
    plt.plot(range(1, n_episodes+1), Q_diff_1, label="First protocol")
    plt.plot(range(1, n_episodes+1), Q_diff_2, label="Second protocol")
    plt.plot(range(1, n_episodes+1), Q_diff_3, label="Third protocol")
    plt.legend()
    plt.xlabel("Number of episodes", fontsize=18)
    plt.ylabel("$\\|\\hat{Q}-J^N_{\\mu^*}\\|_{\\infty}$", fontsize=18)
    plt.show()
