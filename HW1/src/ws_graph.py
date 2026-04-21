import random
import matplotlib.pyplot as plt
NEG_INF = -10**9

######################################
# Graph helper
######################################

def matrix_to_list(adj_matrix):
    n = len(adj_matrix)
    adj_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                adj_list[i].append(j)
    return adj_list

def edge_count(adj_matrix):
    n = len(adj_matrix)
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] == 1:
                total += 1
    return total

def graph_string(adj_matrix):
    n = len(adj_matrix)
    rows = ["".join(str(x) for x in row) for row in adj_matrix]
    return str(n) + " " + " ".join(rows)

######################################
# Check generated graph
######################################

def check_graph(adj_matrix, expected_n=None, expected_k=None):
    n = len(adj_matrix)

    if expected_n is not None and n != expected_n:
        return False, f"wrong number of nodes"

    for row in adj_matrix:
        if len(row) != n:
            return False, f"matrix is not square"

    for i in range(n):
        if adj_matrix[i][i] != 0:
            return False, f"self-loop found"
        for j in range(n):
            if adj_matrix[i][j] not in (0, 1):
                return False, f"matrix must contain only 0 or 1"
            if adj_matrix[i][j] != adj_matrix[j][i]:
                return False, f"graph must be undirected"

    degrees = [sum(adj_matrix[i]) for i in range(n)]
    if min(degrees) <= 0:
        return False, f"isolated node found"

    if expected_k is not None:
        expected_edges = expected_n * expected_k // 2
        if edge_count(adj_matrix) != expected_edges:
            return False, f"wrong number of edges"

    return True, {
        "n": n,
        "min_degree": min(degrees),
        "max_degree": max(degrees),
        "edges": edge_count(adj_matrix)
    }

######################################
# Generate WS graph
######################################

def ws_graph(n, k, p, seed=None):
    if k % 2 != 0:
        raise ValueError("k must be even")
    if k <= 0 or k >= n:
        raise ValueError("k must satisfy 0 < k < n")
    if p < 0 or p > 1:
        raise ValueError("p must be between 0 and 1")

    rng = random.Random(seed)
    adj_matrix = [[0] * n for _ in range(n)]
    half = k // 2

    # initial ring lattice
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1

    # rewire clockwise edges
    for i in range(n):
        for d in range(1, half + 1):
            j = (i + d) % n

            # only process each undirected edge once
            if i < j or (i + d) >= n:
                if rng.random() < p:
                    adj_matrix[i][j] = 0
                    adj_matrix[j][i] = 0

                    candidates = []
                    for m in range(n):
                        if m != i and adj_matrix[i][m] == 0:
                            candidates.append(m)

                    if len(candidates) == 0:
                        adj_matrix[i][j] = 1
                        adj_matrix[j][i] = 1
                    else:
                        m = rng.choice(candidates)
                        adj_matrix[i][m] = 1
                        adj_matrix[m][i] = 1

    ok, info = check_graph(adj_matrix, expected_n=n, expected_k=k)
    if not ok:
        raise RuntimeError(f"invalid WS graph: {info}")

    return adj_matrix

######################################
# Experimental best response
######################################

def random_state(length, rng):
    return [rng.randint(0, 1) for _ in range(length)]

def random_best_response(num_players, utility_fn, initial_state, rng, max_steps=100000):
    state = initial_state[:]
    move_count = 0

    for _ in range(max_steps):
        waiting = []

        for i in range(num_players):
            current_action = state[i]
            current_u = utility_fn(i, current_action, state)

            flipped_action = 1 - current_action
            flipped_u = utility_fn(i, flipped_action, state)

            if flipped_u > current_u:
                waiting.append((i, flipped_action))

        if len(waiting) == 0:
            break

        player, new_action = rng.choice(waiting)
        state[player] = new_action
        move_count += 1

    return state, move_count

######################################
# Symmetric MIS experiment
######################################

def exp_mis(n, adj_list, rng):
    def utility(i, action, state):
        selected_neighbors = sum(state[j] for j in adj_list[i])

        if action == 1:
            if selected_neighbors > 0:
                return NEG_INF
            return 1
        else:
            if selected_neighbors > 0:
                return 0
            return -1

    initial_state = random_state(n, rng)
    final_state, move_count = random_best_response(n, utility, initial_state, rng)

    selected_nodes = [i for i in range(n) if final_state[i] == 1]
    return selected_nodes, move_count

######################################
# Symmetric MDS-based IDS experiment
######################################

def dominated_neighbor(v, selected_set, adj_list):
    for u in adj_list[v]:
        if u in selected_set:
            return True
    return False

def exp_ids(n, adj_list, rng):
    selection_cost = 1.5

    def utility(i, action, state):
        selected_others = {u for u in range(n) if state[u] == 1 and u != i}

        if action == 1:
            for j in adj_list[i]:
                if j in selected_others:
                    return NEG_INF

            closed_neighborhood = [i] + adj_list[i]
            newly_covered = 0
            for v in closed_neighborhood:
                if v in selected_others:
                    continue

                dominated = False
                for u in adj_list[v]:
                    if u in selected_others:
                        dominated = True
                        break

                if not dominated:
                    newly_covered += 1

            return newly_covered - selection_cost
        else:
            if dominated_neighbor(i, selected_others, adj_list):
                return 0
            return -2

    initial_state = random_state(n, rng)
    final_state, move_count = random_best_response(n, utility, initial_state, rng)

    selected_nodes = [i for i in range(n) if final_state[i] == 1]
    return selected_nodes, move_count

######################################
# Matching experiment
######################################

def build_edges(n, adj_matrix):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] == 1:
                edges.append((i, j))
    return edges

def build_adjacent_graph(edges):
    m = len(edges)
    edge_adj = [[] for _ in range(m)]

    for a in range(m):
        u1, v1 = edges[a]
        endpoints_a = {u1, v1}

        for b in range(a + 1, m):
            u2, v2 = edges[b]
            endpoints_b = {u2, v2}

            if endpoints_a & endpoints_b:
                edge_adj[a].append(b)
                edge_adj[b].append(a)

    return edge_adj

def exp_matching(n, adj_matrix, rng):
    edges = build_edges(n, adj_matrix)
    m = len(edges)

    if m == 0:
        return [], 0

    edge_adj = build_adjacent_graph(edges)

    def utility(e_idx, action, state):
        selected_adjacent = sum(state[x] for x in edge_adj[e_idx])

        if action == 1:
            if selected_adjacent > 0:
                return NEG_INF
            return 1
        else:
            if selected_adjacent > 0:
                return 0
            return -1

    initial_state = random_state(m, rng)
    final_state, move_count = random_best_response(m, utility, initial_state, rng)

    selected_edges = [edges[i] for i in range(m) if final_state[i] == 1]
    return selected_edges, move_count

######################################
# Run experiment
######################################

def avg(values):
    if len(values) == 0:
        return 0
    return sum(values) / len(values)

def run_experiment(n=20, k=4, p_values=None, trials=10):
    if p_values is None:
        p_values = [0.0, 0.2, 0.4, 0.6, 0.8]

    result = {
        "p": [],
        "mis_card": [],
        "ids_card": [],
        "match_card": [],
        "mis_moves": [],
        "ids_moves": [],
        "match_moves": []
    }

    for p in p_values:
        mis_card_vals = []
        ids_card_vals = []
        match_card_vals = []

        mis_move_vals = []
        ids_move_vals = []
        match_move_vals = []

        for t in range(trials):
            adj_matrix = ws_graph(n, k, p, seed=1000 + t)
            adj_list = matrix_to_list(adj_matrix)

            rng1 = random.Random(2000 + t)
            rng2 = random.Random(3000 + t)
            rng3 = random.Random(4000 + t)

            mis_nodes, mis_moves = exp_mis(n, adj_list, rng1)
            ids_nodes, ids_moves = exp_ids(n, adj_list, rng2)
            match_edges, match_moves = exp_matching(n, adj_matrix, rng3)

            mis_card_vals.append(len(mis_nodes))
            ids_card_vals.append(len(ids_nodes))
            match_card_vals.append(len(match_edges))

            mis_move_vals.append(mis_moves)
            ids_move_vals.append(ids_moves)
            match_move_vals.append(match_moves)

        result["p"].append(p)
        result["mis_card"].append(avg(mis_card_vals))
        result["ids_card"].append(avg(ids_card_vals))
        result["match_card"].append(avg(match_card_vals))

        # average moves per node
        result["mis_moves"].append(avg(mis_move_vals) / n)
        result["ids_moves"].append(avg(ids_move_vals) / n)
        result["match_moves"].append(avg(match_move_vals) / n)

    return result

######################################
# Plot
######################################

def plot_cardinality(result):
    plt.figure()
    plt.plot(result["p"], result["mis_card"], marker="o", label="Symmetric MIS")
    plt.plot(result["p"], result["ids_card"], marker="s", label="Symmetric MDS-based IDS")
    plt.plot(result["p"], result["match_card"], marker="^", label="Matching")
    plt.xlabel("WS rewiring probability")
    plt.ylabel("Average cardinality")
    plt.title("Game cardinality comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_moves(result):
    plt.figure()
    plt.plot(result["p"], result["mis_moves"], marker="o", label="Symmetric MIS")
    plt.plot(result["p"], result["ids_moves"], marker="s", label="Symmetric MDS-based IDS")
    plt.plot(result["p"], result["match_moves"], marker="^", label="Matching")
    plt.xlabel("WS rewiring probability")
    plt.ylabel("Average movement")
    plt.title("Game movement comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

######################################
# Print one sample graph
######################################

def show_sample_graph(n=10, k=4, p=0.4, seed=123):
    adj_matrix = ws_graph(n, k, p, seed=seed)
    ok, info = check_graph(adj_matrix, expected_n=n, expected_k=k)
    print("Validation:", ok, info)
    print("Input format:")
    print(graph_string(adj_matrix))

if __name__ == "__main__":
    show_sample_graph(n=10, k=4, p=0.4, seed=123)

    result = run_experiment(
        n=20,
        k=4,
        p_values=[0.0, 0.2, 0.4, 0.6, 0.8],
        trials=10
    )

    print(result)
    plot_cardinality(result)
    plot_moves(result)