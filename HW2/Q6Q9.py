import numpy as np
import random
import matplotlib.pyplot as plt

def fictitious_play(payoff_p1, payoff_p2, iterations=500, init_p1=None, init_p2=None):
    n_actions_p1 = payoff_p1.shape[0]
    n_actions_p2 = payoff_p1.shape[1]

    # Use custom priors if given
    if init_p1 is None:
        count_p1 = np.ones(n_actions_p1)
    else:
        count_p1 = np.array(init_p1, dtype=float)

    if init_p2 is None:
        count_p2 = np.ones(n_actions_p2)
    else:
        count_p2 = np.array(init_p2, dtype=float)

    history = []

    for t in range(iterations):
        belief_p1 = count_p2 / np.sum(count_p2)
        belief_p2 = count_p1 / np.sum(count_p1)

        exp_payoff_p1 = payoff_p1 @ belief_p1
        exp_payoff_p2 = payoff_p2.T @ belief_p2

        best_p1 = np.argwhere(exp_payoff_p1 == np.max(exp_payoff_p1)).flatten()
        best_p2 = np.argwhere(exp_payoff_p2 == np.max(exp_payoff_p2)).flatten()

        action_p1 = random.choice(best_p1)
        action_p2 = random.choice(best_p2)

        count_p1[action_p1] += 1
        count_p2[action_p2] += 1

        history.append((int(action_p1), int(action_p2)))

    return history, count_p1, count_p2

def get_distribution(count):
    return count / np.sum(count)

def track_convergence(history, n_actions=2):
    counts = np.ones(n_actions)
    distributions = []

    for action, _ in history:
        counts[action] += 1
        distributions.append(counts / np.sum(counts))

    return np.array(distributions)

def plot_convergence(distributions, title="Convergence"):
    for i in range(distributions.shape[1]):
        plt.plot(distributions[:, i], label=f"Action {i}")

    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid()
    plt.show()

def run_game(payoff_p1, payoff_p2, name="Game", iterations=500, init_p1=None, init_p2=None):
    print(f"\n===== {name} =====")

    history, count_p1, count_p2 = fictitious_play(
        payoff_p1, payoff_p2, iterations, init_p1, init_p2
    )

    dist_p1 = count_p1 / np.sum(count_p1)
    dist_p2 = count_p2 / np.sum(count_p2)

    print("Final strategy Player 1:", dist_p1)
    print("Final strategy Player 2:", dist_p2)
    print("Last 10 actions:", history[-10:])

    # Plot convergence
    dist_track = track_convergence(history)
    plot_convergence(dist_track, title=name)

    return history, dist_p1, dist_p2

if __name__ == "__main__":

    # Q6, answer = NE depends on where init starts, if start from [10,1] [10,1] then wil converge to r1c1, other way around converge to r2c2
    # if neutral then either r1c1 or r2c2, depending on runs.
    # p1_q6 = np.array([
    #     [10, 0],
    #     [0, 10]
    # ])

    # p2_q6 = np.array([
    #     [10, 0],
    #     [0, 10]
    # ])
    # run_game(p1_q6, p2_q6, name="Q6 - Case 1",
    #      init_p1=[10,1], init_p2=[10,1])
    # run_game(p1_q6, p2_q6, name="Q6 - Case 2",
    #      init_p1=[1,10], init_p2=[1,10])
    # run_game(p1_q6, p2_q6, name="Q6 - Neutral")

    # # Q7, answer = players end with different strategies doesnt matter how they're initialized (either r1c2 or r2c1)
    # p1_q7 = np.array([
    #     [0, 1],
    #     [1, 0]
    # ])

    # p2_q7 = np.array([
    #     [0, 1],
    #     [1, 0]
    # ])
    # run_game(p1_q7, p2_q7, name="Q7 - Case 1",
    #      init_p1=[10,1], init_p2=[10,1])
    # run_game(p1_q7, p2_q7, name="Q7 - Case 2",
    #      init_p1=[1,10], init_p2=[1,10])
    # run_game(p1_q7, p2_q7, name="Q7 - Neutral")

    # #Q8, answer = converge according to how its initalizes similar to Q6, but when neutral is 50-50 for both for all runs, slight differ with Q6
    # p1_q8 = np.array([
    #     [3, 0],
    #     [0, 2]
    # ])

    # p2_q8 = np.array([
    #     [2, 0],
    #     [0, 3]
    # ])
    # run_game(p1_q8, p2_q8, name="Q8 - Case 1",
    #      init_p1=[10,1], init_p2=[10,1])
    # run_game(p1_q8, p2_q8, name="Q8 - Case 2",
    #      init_p1=[1,10], init_p2=[1,10])
    # run_game(p1_q8, p2_q8, name="Q8 - Neutral")

    #Q9
    p1_q9 = np.array([
        [3, 0],
        [2, 1]
    ])

    p2_q9 = np.array([
        [3, 2],
        [0, 1]
    ])
    run_game(p1_q9, p2_q9, name="Q9 - Case 1",
         init_p1=[10,1], init_p2=[10,1])
    run_game(p1_q9, p2_q9, name="Q9 - Case 2",
         init_p1=[1,10], init_p2=[1,10])
    run_game(p1_q9, p2_q9, name="Q9 - Neutral")



    