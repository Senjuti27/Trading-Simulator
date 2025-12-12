""""""  		  	   		 	 	 		  		  		    	 		 		   		 		  
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
  		  	   		 	 	 		  		  		    	 		 		   		 		  
Student Name: Senjuti Twisha 		  	   		 	 	 		  		  		    	 		 		   		 		  
GT User ID: stwisha3  		  	   		 	 	 		  		  		    	 		 		   		 		  
GT ID: 904080731 		  	   		 	 	 		  		  		    	 		 		   		 		  
"""  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
import numpy as np
import matplotlib.pyplot as plt
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
def author():  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return "stwisha3"  # replace tb34 with your Georgia Tech username.

def study_group():
    """
    :return: A comma-separated string of GT usernames of your study group members.
             If working alone, just return your username.
    :rtype: str
    """
    return "stwisha3"  # replace/add other usernames if you have a study group

def gtid():
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		 	 	 		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    return 904080731  # replace with your GT ID number
  		  	   		 	 	 		  		  		    	 		 		   		 		  
  		  	   		 	 	 		  		  		    	 		 		   		 		  
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


def simulator(max_spins, target_win, win_prob):
        spin_count = 1
        curr_money = 0
        bet = 1
        money_history = []
        money_history.append(curr_money)
        while curr_money <= target_win and spin_count < max_spins:
            # print(get_spin_result(win_prob))
            if get_spin_result(win_prob):
                curr_money += bet
                bet = 1
            else:
                curr_money -= bet
                bet *= 2
            money_history.append(curr_money)
            spin_count+= 1

        while len(money_history) < max_spins:
            money_history.append(curr_money)

        return money_history


def realistic_simulator(max_spins, target_win, max_loss, win_prob):
    spin_count = 1
    curr_money = 0
    bet = 1
    money_history = []
    money_history.append(curr_money)

    while curr_money <= target_win and spin_count < max_spins:
        if curr_money <= -max_loss:
            curr_money = -max_loss
            break

        # Ensure you don't bet more than your current bankroll
        bet = min(bet, max_loss + curr_money)

        # print(get_spin_result(win_prob))
        if get_spin_result(win_prob):
            curr_money += bet
            bet = 1
        else:
            curr_money -= bet
            bet *= 2
        money_history.append(curr_money)
        spin_count += 1

    while len(money_history) < max_spins:
        money_history.append(curr_money)

    return money_history

def fig1(episode_10):
    # Plot all 10 episodes
    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.plot(episode_10[i], label="Episode " + str(i + 1))
    plt.title("Figure 1: 10 Episodes of Gambling Strategy")
    plt.xlabel("Spin Number")  # X-axis: spin count
    plt.ylabel("Winnings ($)") # Y-axis: current winnings
    # Set axis limits
    plt.xlim(0, 300)        # Horizontal axis from 0 to 300 spins
    plt.ylim(-256, 100)     # Vertical axis from -256 to +100 dollars
    plt.legend()            # Show legend

    plt.savefig("fig1.png")
    # plt.show()              # Show the plot

def fig2(episode_1000):
    plt.figure(figsize=(10, 6))
    # Calculate mean and standard deviation for each spin
    mean_winnings = np.mean(episode_1000, axis=0)
    std_winnings = np.std(episode_1000, axis=0)
    # Plot the mean line
    plt.plot(mean_winnings, label="Mean Winnings")
    # Plot mean + std and mean - std
    plt.plot(mean_winnings + std_winnings, label="Mean + STD", linestyle="--")
    plt.plot(mean_winnings - std_winnings, label="Mean - STD", linestyle="--")

    plt.xlim(0, 300)
    plt.ylim(-256, 100)

    plt.title("Figure 2: Mean Winnings per Spin (1000 Episodes)")
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings ($)")
    plt.legend()
    plt.savefig("fig2.png")
    # plt.show()

def fig3(episode_1000):
    plt.figure(figsize=(10, 6))
    # Calculate mean and standard deviation for each spin
    median_winnings = np.median(episode_1000, axis=0)
    std_winnings = np.std(episode_1000, axis=0)

    # Plot the mean line
    plt.plot(median_winnings, label="Median Winnings")

    # Plot mean + std and mean - std
    plt.plot(median_winnings + std_winnings, label="Median + STD", linestyle="--")
    plt.plot(median_winnings - std_winnings, label="Median - STD", linestyle="--")

    plt.xlim(0, 300)
    plt.ylim(-256, 100)

    plt.title("Figure 3: Median Winnings per Spin (1000 Episodes)")
    plt.xlabel("Spin Number")
    plt.ylabel("Winnings ($)")
    plt.legend()
    plt.savefig("fig3.png")
    # plt.show()

def fig4(episode_1000_2):
    plt.figure(figsize=(10, 6))
    # Mean winnings at each spin across all episodes
    mean_winnings = np.mean(episode_1000_2, axis=0)

    # Standard deviation at each spin across all episodes
    std_winnings = np.std(episode_1000_2, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_winnings, label='Mean Winnings', color='blue')
    plt.plot(mean_winnings + std_winnings, label='Mean + Std', color='green', linestyle='--')
    plt.plot(mean_winnings - std_winnings, label='Mean - Std', color='red', linestyle='--')

    plt.title('Figure 4: Realistic Simulator – Mean and Std of 1000 Episodes')
    plt.xlabel('Spin Number')
    plt.ylabel('Winnings ($)')
    plt.xlim(0, 1000)
    plt.ylim(-256, 300)  # similar to Figure 1 bounds
    plt.legend()
    plt.savefig("fig4.png")
    # plt.show()

def fig5(episode_1000_2):
    plt.figure(figsize=(10, 6))
    # Median winnings at each spin across all episodes
    median_winnings = np.median(episode_1000_2, axis=0)

    # Standard deviation at each spin across all episodes (same as before)
    std_winnings = np.std(episode_1000_2, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(median_winnings, label='Median Winnings', color='blue')
    plt.plot(median_winnings + std_winnings, label='Median + Std', color='green', linestyle='--')
    plt.plot(median_winnings - std_winnings, label='Median - Std', color='red', linestyle='--')

    plt.title('Figure 5: Realistic Simulator – Median and Std of 1000 Episodes')
    plt.xlabel('Spin Number')
    plt.ylabel('Winnings ($)')
    plt.xlim(0, 1000)
    plt.ylim(-256, 300)  # same axis bounds as Figure 4
    plt.legend()
    plt.savefig("fig5.png")
    # plt.show()


def test_code():
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		 	 	 		  		  		    	 		 		   		 		  
    """  		  	   		 	 	 		  		  		    	 		 		   		 		  
    win_prob = 0.4737  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once  		  	   		 	 	 		  		  		    	 		 		   		 		  
    # print(get_spin_result(win_prob))  # test the roulette spin
    # add your code here to implement the experiments

    """
    Experiment 1
    """
    episode_10 = []
    for count in range(10):
        res1 = simulator(1000, 80, win_prob)
        episode_10.append(res1)
    episode_10 = np.array(episode_10)
    # print(episode_10[0])
    fig1(episode_10)

    episode_1000 = []
    for count in range(1000):
        res2 = simulator(1000, 80, win_prob)
        episode_1000.append(res2)
    episode_1000 = np.array(episode_1000)
    # print(episode_1000[0])
    fig2(episode_1000)
    fig3(episode_1000)

    # --- Compute probability of hitting $80 and expected value ---
    final_money = episode_1000[:, -1]  # last spin of each episode
    prob_hit_80 = np.sum(final_money >= 80) / 1000  # probability of reaching target
    expected_money = np.mean(final_money)  # average money after 1000 episodes
    #
    # print(f"Experiment 1:")
    # print(f"Probability of reaching the $80 target = {prob_hit_80:.4f} (fraction of episodes hitting target)")
    # print(f"Expected winnings after 1000 spins = ${expected_money:.2f} (average final money across episodes)")

    """
    Experiment 2
    """
    episode_1000_2 = []
    for count in range(1000):
        res3 = realistic_simulator(1000,80,256,win_prob)
        episode_1000_2.append(res3)
    episode_1000_2 = np.array(episode_1000_2)
    # print(episode_1000[0]_2)
    fig4(episode_1000_2)
    fig5(episode_1000_2)

    # --- Compute probability and expected value for realistic simulator ---
    final_money2 = episode_1000_2[:, -1]
    prob_hit_80_2 = np.sum(final_money2 >= 80) / 1000
    expected_money2 = np.mean(final_money2)
    #
    # print(f"\nExperiment 2 (Realistic Simulator):")
    # print(f"Probability of reaching the $80 target = {prob_hit_80_2:.4f}")
    # print(f"Expected winnings after 1000 spins = ${expected_money2:.2f}")


if __name__ == "__main__":
    test_code()  		  	   		 	 	 		  		  		    	 		 		   		 		  
