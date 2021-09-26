
import random
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)


"""
INFO8003-1 : Optimal decision making for complex problems
2021 - Assignment 2
Pierre NAVEZ & Antoine DEBOR

SECTION 1 - Implementation of the domain
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
    def __init__(self, domain):
        self.domain = domain


    def policy(self):
        """
        The policy followed by the agent in order to choose its actions

        Returns
        -------
        An action from the possible action values in the domain

        """
        actions = self.domain.get_actions()
        # Random policy (please uncomment the following line to test it, and comment the "always accelerate" policy)
        #index = random.randint(0, 1)
        # "Always accelerate" policy
        index = 1
        return actions[index]

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




if __name__ == "__main__":
    random.seed(0)

    domain = Domain()
    agent = Agent(domain)

    # Display of trajectory following "always accelerate" policy
    trajectory = agent.domain_exploration(10+1)
    print("\nTrajectory for policy \"always accelerate\" : \n")
    for i, ((p,s), u, r, (p_next, s_next)) in enumerate(trajectory):
        print("(x_{iter}, u_{iter}, r_{iter}, x_{iterbis}) : ({},{},{},{})".format((p,s), u, r, (p_next, s_next), iter=i, iterbis=i+1))

    # Display of trajectory following "always accelerate" policy, for a large number of iterations
    trajectory = agent.domain_exploration(5000)
    p_vec = []
    s_vec = []
    for (p,s),_,_,_ in trajectory:
        p_vec.append(p)
        s_vec.append(s)
    plt.plot(p_vec, s_vec, label="Trajectory")
    plt.plot(trajectory[0][0][0], trajectory[0][0][1], 'cp', label="Initial state")
    plt.plot(trajectory[4999][0][0], trajectory[4999][0][1], 'gp', label="Terminal state")
    plt.xlabel('$p$')
    plt.ylabel('$s$')
    plt.legend()
    plt.savefig("Trajectory_always_accelerate.pdf")
    plt.show()

    

    
    
    