""""""  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		 	   		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		 	   		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		 	   		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		 	   		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		 	   		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		 	   		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		 	   		  		  		    	 		 		   		 		  
or edited.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		 	   		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		 	   		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		 	   		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		 	   		  		  		    	 		 		   		 		  

Student Name: Tucker Balch (replace with your name)
GT User ID: tb34 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""

import random as rand

import numpy as np

class QLearner(object):
    """
    This is a Q learner object.

    :param num_states: The number of states to consider.
    :type num_states: int
    :param num_actions: The number of actions available.
    :type num_actions: int
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    :type alpha: float
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    :type gamma: float
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    :type rar: float
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    :type radr: float
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    :type dyna: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """
    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=False,
    ):
        """
        Constructor method
        """
        ## Init key Q-Learning params
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

        ## Init Q-table for all pairs
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.s = 0                  # Current State of the agent
        self.a = 0                  # Current Action of the agent
        self.experience = []        # Experience Buffer for Dyna-Q

        if self.verbose:
            print("Initializing Q-Learner")
            print("num_states: ", self.num_states)
            print("num_actions: ", self.num_actions)
            print("alpha: ", self.alpha)
            print("gamma: ", self.gamma)
            print("rar: ", self.rar)
            print("radr: ", self.radr)
            print("dyna: ", self.dyna)



    def querysetstate(self, s):
        """
        Update the state without updating the Q-table

        :param s: The new state
        :type s: int
        :return: The selected action
        :rtype: int
        """

        # Update state
        self.s = s

        # Update action
        if rand.uniform(0, 1) < self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s, :])

        self.a = action

        if self.verbose:
            print(f"QLearner Query set state: s = {s}, a = {action}")
        return action

    def query(self, s_prime, r):
        """
        Update the Q table and return an action

        :param s_prime: The new state
        :type s_prime: int
        :param r: The immediate reward
        :type r: float
        :return: The selected action
        :rtype: int
        """
        # Update Q-value using the Q-learning formula
        self.Q[self.s, self.a] = (1 - self.alpha) * self.Q[self.s, self.a] + \
                                 self.alpha * (r + self.gamma * np.max(self.Q[s_prime, :]))

        # If Dyna-Q is enabled, add the experience to memory and hallucinate
        if self.dyna > 0:
            # Store the learning instance, or experience tuple (s, a, s_prime, r)
            self.experience.append((self.s, self.a, s_prime, r))
            # Simulate `self.dyna` additional experiences by sampling from the experience buffer
            for _ in range(self.dyna):
                # Randomly sample an experience from memory
                s, a, s_next, reward = self.experience[rand.randint(0, len(self.experience) - 1)]
                # Update Q-table based on this simulated experience
                self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + \
                               self.alpha * (reward + self.gamma * np.max(self.Q[s_next, :]))

        # Exploration rate decay
        self.rar *= self.radr

        # Choose next action based on exploration-exploitation trade-off
        if rand.uniform(0.0, 1.0) < self.rar:
            action = rand.randint(0, self.num_actions - 1)  # Random action
        else:
            action = np.argmax(self.Q[s_prime, :])  # Best action from Q-table

        # Update the current state and action
        self.s = s_prime
        self.a = action

        if self.verbose:
            print(f"Q-Learner Query: s = {s_prime}, a = {action}, r={r}")

        return self.a

    def author(self):
        """
        :return: The GT username of the student
        :rtype: str
        """
        return "akommi3"  # replace tb34 with your Georgia Tech username.

    def study_group(self):
        """
        :return: A comma separated string of GT_Name of each member of your study group
        :rtype: str
        # Example:"gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone
        """
        return "akommi3"

    def gtid(self):
        """
        :return: The GT ID of the student
        :rtype: int
        """
        return 903135337  # replace with your GT ID number

if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
