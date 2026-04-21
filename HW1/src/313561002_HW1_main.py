import sys
import random
NEG_INF = -10**9

def parse_graph(argv):
    n = int(argv[1])
    bit_rows = argv[2:]

    if len(argv) < 2 or len(argv) != n + 2:
        raise ValueError(f"please follow correct command-line argument")
    for row in bit_rows:
        if len(row) != n:
            raise ValueError(f"please follow correct command-line argument")
        if any(ch not in ('0', '1') for ch in row):
            raise ValueError("please follow correct command-line argument")

    # Build list
    adj_matrix = [[int(ch) for ch in row] for row in bit_rows]
    adj_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                adj_list[i].append(j)

    return n, adj_matrix, adj_list

######################################
# Player's best response for given state
######################################

def best_response(num_players, utility_fn, initial_state): 
    state = initial_state[:]
    move_count = 0

    while True:
        best_gain = 0
        best_player = None
        best_new_action = None
        
        for i in range(num_players):
            current_action = state[i]
            current_u = utility_fn(i, current_action, state)

            flipped_action = 1 - current_action
            flipped_u = utility_fn(i, flipped_action, state)

            gain = flipped_u - current_u
            if gain > best_gain:
                best_gain = gain
                best_player = i
                best_new_action = flipped_action

        if best_player is None:
            break

        state[best_player] = best_new_action
        move_count += 1

    return state, move_count

######################################
# Problem set 1: Symmetric MIS game
######################################

def symmetric_mis(n, adj_list):
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

    initial_state = [0] * n
    final_state, move_count = best_response(n, utility, initial_state)
    selected_nodes = [i for i in range(n) if final_state[i] == 1]
    return selected_nodes, move_count

######################################
# Problem set 2: Symmetric MDS-based IDS game
######################################

def dominated_neighbor(v, selected_set, adj_list):
    for u in adj_list[v]:
        if u in selected_set:
            return True
    return False

def symmetric_mds_ids(n, adj_list):
    selection_cost = 1.5
 
    def utility(i, action, state):
        selected_others = {u for u in range(n) if state[u] == 1 and u != i}
        if action == 1: #hold independence
            for j in adj_list[i]:
                if j in selected_others:
                    return NEG_INF
            #count nodes in closed neighborhood of i 
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

        #else if action == 0, node i should be dominated by some selected neighbor
        else:
            if dominated_neighbor(i, selected_others, adj_list):
                return 0
            return -2

    initial_state = [0] * n
    final_state, move_count = best_response(n, utility, initial_state)
    
    selected_nodes = [i for i in range(n) if final_state[i] == 1]
    return selected_nodes, move_count

######################################
# Problem set 3: Matching Game
######################################

def build_edges(n, adj_matrix):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i][j] == 1:
                edges.append((i, j))
    return edges

def build_adjacent_graph(edges):
    #two players are adjacent in the graph if the two share an endpoint
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

def matching_game(n, adj_matrix):
    #action = 1 is bad if any adjacent edge is already selected, good otherwise.
    #action = 0 is bad if no adjacent edge is selected, okay otherwise
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
 
    initial_state = [0] * m
    final_state, move_count = best_response(m, utility, initial_state)

    selected_edges = [edges[i] for i in range(m) if final_state[i] == 1]
    return selected_edges, move_count

if __name__ == "__main__":
    n, adj_matrix, adj_list = parse_graph(sys.argv)

    # Problem set 1
    mis_nodes, mis_moves = symmetric_mis(n, adj_list)
    print("Requirement 1-1")
    print( "the cardinality of Symmetric Maximal Independent Set (MIS) Game is", len(mis_nodes))

    # Problem set 2
    ids_nodes, ids_moves = symmetric_mds_ids(n, adj_list)
    print("Requirement 1-2")
    print("the cardinality of Symmetric MDS-based IDS Game is",len(ids_nodes))

    # Problem set 3
    matching_edges, matching_moves = matching_game(n, adj_matrix)
    print("Requirement 2")
    print("the cardinality of Matching Game is",len(matching_edges))