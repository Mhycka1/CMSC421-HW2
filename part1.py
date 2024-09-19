import random
import numpy as np
import networkx as nx


# #method to generate graphs
def make_graph(node_amount):
    graph = nx.Graph()
    graph.add_nodes_from(range(node_amount))

    for i in range(node_amount):
        for j in range(i + 1, node_amount):
            distance = random.randint(1, 100)
            graph.add_edge(i, j, weight=distance)
    
    matrix = nx.to_numpy_array(graph)
    print(matrix)

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
        next_ind = np.argmin(adj_matrix[last][mask]) # find minimum of remaining locations
        next_loc = np.arange(N)[mask][next_ind] # convert to original location
        path.append(next_loc)
        mask[next_loc] = False
        cost += adj_matrix[last, next_loc]

    return path, cost


