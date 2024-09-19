import random
import numpy as np
import networkx as nx
from py2opt.routefinder import RouteFinder


# #method to generate graphs
def make_graph(node_amount):
    graph = nx.Graph()
    graph.add_nodes_from(range(node_amount))

    for i in range(node_amount):
        for j in range(i + 1, node_amount):
            distance = random.randint(1, 100)
            graph.add_edge(i, j, weight=distance)
    
    matrix = nx.to_numpy_array(graph)

    return matrix

# Nearest Neighbors algorithm
# code based on the below stackoverflow post
# https://stackoverflow.com/questions/17493494/nearest-neighbour-algorithm
def nearest_neighbor(adj_matrix, start):
    path = [start]
    cost = 0
    N = adj_matrix.shape[0]
    mask = np.ones(N, dtype=bool)  # boolean values indicating which 
                                   # locations have not been visited
    mask[start] = False

    for i in range(N-1):
        last = path[-1]
        next_ind = np.argmin(adj_matrix[last][mask]) 
        next_loc = np.arange(N)[mask][next_ind] 
        path.append(int(next_loc))
        mask[next_loc] = False
        cost += adj_matrix[last, next_loc]

    return path, cost

def nearest_neighbor_2opt(adj_matrix):
    path, cost = nearest_neighbor(adj_matrix, 0)
    optimized_route = two_opt(path, adj_matrix)


# edited from the below stackoverflow post
# https://stackoverflow.com/questions/53275314/2-opt-algorithm-to-solve-the-travelling-salesman-problem-in-python
def cost_change(cost_mat, n1, n2, n3, n4):
    return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]

# edited from the same stackoverflow post as above
# https://stackoverflow.com/questions/53275314/2-opt-algorithm-to-solve-the-travelling-salesman-problem-in-python
def two_opt(route, cost_mat):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                if cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
        route = best
    return best
