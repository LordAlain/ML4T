""""""
from numpy.ma.core import shape

"""Assess a betting strategy.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
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
  		  	   		 	   		  		  		    	 		 		   		 		  
Student Name: Aditya Kommi (replace with your name)  		  	   		 	   		  		  		    	 		 		   		 		  
GT User ID: akommi3 (replace with your User ID)  		  	   		 	   		  		  		    	 		 		   		 		  
GT ID: 903135337 (replace with your GT ID)  		  	   		 	   		  		  		    	 		 		   		 		  
"""  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
import numpy as np
import matplotlib.pyplot as plt
import sys
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def author():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    return "akommi3"  # replace tb34 with your Georgia Tech username.


def study_group():
    """
    :return: A comma separated string of GT_Name of each member of your study group
    :rtype: str
    # Example:"gburdell3, jdoe77, tbalch7" or "gburdell3" if a single individual working alone
    """
    return "akommi3"

def gtid():  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    return 903135337  # replace with your GT ID number
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def get_spin_result(win_prob):  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		 	   		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		 	   		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		 	   		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    result = False  		  	   		 	   		  		  		    	 		 		   		 		  
    if np.random.random() <= win_prob:  		  	   		 	   		  		  		    	 		 		   		 		  
        result = True  		  	   		 	   		  		  		    	 		 		   		 		  
    return result  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
  		  	   		 	   		  		  		    	 		 		   		 		  
def test_code(verbose=False):
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		 	   		  		  		    	 		 		   		 		  
    """  		  	   		 	   		  		  		    	 		 		   		 		  
    win_prob = 9.0/19  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once

    if verbose:
        print("win_prob:", win_prob, file=open('p1_test.txt', 'w'))
        print("get_spin_result(win_prob):", get_spin_result(win_prob), file=open('p1_test.txt', 'a'))  # test the roulette spin

    # add your code here to implement the experiments
    exp1(verbose)
    exp2(verbose)
  		  	   		 	   		  		  		    	 		 		   		 		  

def sim():
    """
    Method to run sim based on provided pseudocode
    """
    spin_count = 0
    episode_results = np.full(1001,80)
    win_prob = 9.0/19  # set appropriately to the probability of a win
    winnings = 0
    won_count = 0.0
    while winnings < 80 and spin_count < 1001:
        won = False
        bet_amount = 1
        while not won:
            # wager bet_amount on black
            episode_results[spin_count] = winnings
            spin_count += 1
            won = get_spin_result(win_prob) # won = result of roulette wheel spin
            if won:
                winnings = winnings + bet_amount
                won_count = 1.0
            else:
                winnings = winnings - bet_amount
                bet_amount = bet_amount * 2
                won_count = 0.0

    return episode_results, spin_count, won_count

def sim2(verbose=False):
    """
    Method to run sim based on provided pseudocode
    """
    spin_count = 0
    episode_results = np.full(1001,80)
    win_prob = 9.0/19  # set appropriately to the probability of a win
    winnings = 0
    bankroll = 256
    won_count = 0.0
    while winnings < 80 and spin_count < 1001:
        bet_amount = 1
        won = False
        while not won:
            # wager bet_amount on black
            episode_results[spin_count] = winnings
            spin_count += 1
            won = get_spin_result(win_prob) # won = result of roulette wheel spin
            if won:
                winnings = winnings + bet_amount
                won_count = 1.0
            else:
                winnings = winnings - bet_amount
                bet_amount = bet_amount * 2
                won_count = 0.0

                # bet amount more than remaining funds, Bet remaining
                if bet_amount >= bankroll + winnings:
                    bet_amount = bankroll + winnings
                    print("Bet Remaining:", bet_amount, ":: spin_count:", spin_count, file=open('p1_exp2.txt', 'a'))

                # Lost bankroll, fill forward
                if winnings <= -bankroll:
                    if verbose: print("Lost Bankroll - spin_count:", spin_count, ":: winnings:", winnings, file=open('p1_exp2.txt', 'a'))
                    episode_results[spin_count:] = winnings
                    return episode_results, spin_count, won_count


    return episode_results, spin_count, won_count

def exp1(verbose=False):
    
    ### Fig 1
    # plot winning from 10 episodes
    # horizontal axis from 0 to 300, vertical axis from -256 to +100

    plt.axis([0, 300, -256, 100])
    plt.title("Figure 1 - 10 Episodes")
    plt.xlabel("Spins")
    plt.ylabel("Winnings")

    if verbose:
        print("Experiment 1:", file=open('p1_exp1.txt', 'w'))
        print("Figure 1:", file=open('p1_exp1.txt', 'a'))
    won_count = 0.0
    for index in range(10):
        episode, spin_count, won = sim()
        won_count = won_count + won
        if verbose:
            # check when sim ended
            print("episode:", index+1, ":: spin_count:", spin_count,":: won_count:", won_count, file=open('p1_exp1.txt', 'a'))
        plt.plot(episode, label="Episode[{0}]".format(index))

    plt.legend()
    plt.savefig("./images/fig1.png")
    plt.clf()

    ### Fig 2
    # plot the (mean, mean + std, mean - std) of winnings from 1000 episodes
    # horizontal axis from 0 to 300, vertical axis from -256 to +100

    if verbose:
        print("\nFigure 2 & 3:", file=open('p1_exp1.txt', 'a'))
    results = np.zeros(shape=(1000, 1001))
    spin_count = np.zeros(shape=1000)
    won_count = 0.0
    for index in range(1000):
        results[index], spin_count[index], won = sim()
        won_count = won_count + won
        if verbose:
            # check when sim ended
            print("episode:", index+1, ":: spin_count:", spin_count[index],":: won_count:", won_count, file=open('p1_exp1.txt', 'a'))

    means = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    if verbose:
        # check when sim ende
        print("spin_count mean:", np.mean(spin_count), file=open('p1_exp1.txt', 'a'))
        print("spin_count median:", np.median(spin_count), file=open('p1_exp1.txt', 'a'))
        print("spin_count std:", np.std(spin_count), file=open('p1_exp1.txt', 'a'))
        print("Mean after 1000 spins:", means[999],file=open('p1_exp1.txt', 'a'))
        print("StDev after 1000 spins:", std[999],file=open('p1_exp1.txt', 'a'))

    plt.axis([0, 300, -256, 100])
    plt.title("Figure 2 - Means (+-stdev)")
    plt.xlabel("Spins")
    plt.ylabel("Winnings")

    plt.plot(means, label="mean")
    plt.plot(means + std, label="mean+stdev")
    plt.plot(means - std, label="mean-stdev")

    plt.legend()
    plt.savefig("./images/fig2.png")
    plt.clf()

    ### Fig 3
    # plot the medians of winnings from 1000 episodes
    # horizontal axis from 0 to 300, vertical axis from -256 to +100

    medians = np.median(results, axis=0)

    plt.axis([0, 300, -256, 100])
    plt.title("Figure 3 - Medians (+-stdev)")
    plt.xlabel("Spins")
    plt.ylabel("Winnings")

    plt.plot(medians, label="median")
    plt.plot(medians + std, label="median+stdev")
    plt.plot(medians - std, label="median-stdev")

    plt.legend()
    plt.savefig("./images/fig3.png")
    plt.clf()


def exp2(verbose=False):

    if verbose:
        print("Experiment 2:", file=open('p1_exp2.txt', 'w'))
        print("Figure 4 & 5:", file=open('p1_exp2.txt', 'a'))

    ### Fig 4
    # plot the (mean, mean + std, mean - std) of winnings from 1000 episodes
    # horizontal axis from 0 to 300, vertical axis from -256 to +100

    results = np.zeros(shape=(1000, 1001))
    spin_count = np.zeros(shape=1000)
    won_count = 0.0
    for index in range(1000):
        #results[index], spin_count[index] = sim2(verbose)
        results[index], spin_count[index], won = sim2(verbose)
        won_count = won_count + won
        if verbose:
            # check when sim ended
            #print("episode:", index+1, ":: spin_count:", spin_count[index], file=open('p1_exp2.txt.txt', 'a'))
            print("episode:", index+1, ":: spin_count:", spin_count[index], ":: won_count:", won_count, file=open('p1_exp2.txt', 'a'))
        
    means = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    if verbose:
        # check when sim ende
        print("spin_count mean:", np.mean(spin_count), file=open('p1_exp2.txt', 'a'))
        print("spin_count median:", np.median(spin_count), file=open('p1_exp2.txt', 'a'))
        print("spin_count std:", np.std(spin_count), file=open('p1_exp2.txt', 'a'))
        print("Mean after 1000 spins:", means[999],file=open('p1_exp2.txt', 'a'))
        print("StDev after 1000 spins:", std[999],file=open('p1_exp2.txt', 'a'))

    plt.axis([0, 300, -256, 100])
    plt.title("Figure 4 - Means (+-stdev)")
    plt.xlabel("Spins")
    plt.ylabel("Winnings")

    plt.plot(means, label="mean")
    plt.plot(means + std, label="mean+stdev")
    plt.plot(means - std, label="mean-stdev")


    plt.legend()
    plt.savefig("./images/fig4.png")
    plt.clf()


    ### Fig 5
    # plot the medians of winnings from 1000 episodes
    # horizontal axis from 0 to 300, vertical axis from -256 to +100

    medians = np.median(results, axis=0)

    plt.axis([0, 300, -256, 100])
    plt.title("Figure 5 - Medians (+-stdev)")
    plt.xlabel("Spins")
    plt.ylabel("Winnings")

    plt.plot(medians, label="median")
    plt.plot(medians + std, label="median+stdev")
    plt.plot(medians - std, label="median-stdev")

    plt.legend()
    plt.savefig("./images/fig5.png")
    plt.clf()



if __name__ == "__main__":
    # If last command line argument is "-debug", then set verbose = True to enable local printing
    # Gradescope will never invoke this code with the "-debug" argument.
    if sys.argv[len(sys.argv) - 1] == "-debug":
        test_code(verbose=True)
    else:
        test_code()