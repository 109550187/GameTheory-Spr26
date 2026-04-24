import numpy as np
import random
import matplotlib.pyplot as plt

def fictitious_play(payoff_p1, payoff_p2, iterations=500):
    n_actions_p1 = payoff_p1.shape[0]
    n_actions_p2 = payoff_p1.shape[1]

    # Initialized states
    count_p1 = np.ones(n_actions_p1)
    count_p2 = np.ones(n_actions_p2)

    history = []

    for t in range(iterations):
        # Beliefs
        belief_p1 = count_p2 / np.sum(count_p2)
        belief_p2 = count_p1 / np.sum(count_p1)

        # Expected payoffs
        exp_payoff_p1 = payoff_p1 @ belief_p1
        exp_payoff_p2 = payoff_p2.T @ belief_p2

        # Best responses
        best_p1 = np.argwhere(exp_payoff_p1 == np.max(exp_payoff_p1)).flatten()
        best_p2 = np.argwhere(exp_payoff_p2 == np.max(exp_payoff_p2)).flatten()

        action_p1 = random.choice(best_p1)
        action_p2 = random.choice(best_p2)

        # Update counts
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

def run_game(payoff_p1, payoff_p2, name="Game", iterations=500):
    print(f"\n===== {name} =====")

    history, count_p1, count_p2 = fictitious_play(
        payoff_p1, payoff_p2, iterations
    )

    dist_p1 = get_distribution(count_p1)
    dist_p2 = get_distribution(count_p2)

    print("Final strategy Player 1:", dist_p1)
    print("Final strategy Player 2:", dist_p2)
    print("Last 10 actions:", history[-10:])

    # Plot convergence
    dist_track = track_convergence(history)
    plot_convergence(dist_track, title=name)

    return history, dist_p1, dist_p2

if __name__ == "__main__":
    # Q1 payoff matrices , answer = always r2, c2
    p1_q1 = np.array([
        [-1, 1],
        [0, 3]
    ])

    p2_q1 = np.array([
        [-1, 0],
        [1, 3]
    ])

    # Q2, answer = sometimes r1,c1 sometimes r2,c2
    p1_q2 = np.array([
        [2, 1],
        [0, 3]
    ])

    p2_q2 = np.array([
        [2, 0],
        [1, 3]
    ])

    # Q3, answer = code will always converge to r1, c1 due to players getting better payoff instead of the indifference when NE is r2,c2
    p1_q3 = np.array([
        [1, 0],
        [0, 0]
    ])

    p2_q3 = np.array([
        [1, 0],
        [0, 0]
    ])

    # Q4, answer = no specific pure NE but the answer follows the assignment's given mixed NE r1=4/5, r2=1/5, c1=1/2, c2=1/2.
    p1_q4 = np.array([
        [0, 2],
        [2, 0]
    ])

    p2_q4 = np.array([
        [1, 0],
        [0, 4]
    ])

    # Q5, answer = no convergence, ping pongs between different answers. almost 50-50 for all strategies of players
    p1_q5 = np.array([
        [0, 1],
        [1, 0]
    ])

    p2_q5 = np.array([
        [1, 0],
        [0, 1]
    ])

    # Run Games
    # run_game(p1_q1, p2_q1, name="Q1", iterations=200)
    # run_game(p1_q2, p2_q2, name="Q2", iterations=200)
    # run_game(p1_q3, p2_q3, name="Q3", iterations=200)
    # run_game(p1_q4, p2_q4, name="Q4", iterations=200)
    run_game(p1_q5, p2_q5, name="Q5", iterations=200)