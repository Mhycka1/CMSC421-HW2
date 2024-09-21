import random
import numpy as np
import networkx as nx
from py2opt.routefinder import RouteFinder
import time
import csv
import os
import psutil  # for CPU time


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
def nearest_neighbor(adj_matrix, start, track_time):

    if track_time:
        real_start_time = time.time()  # Wall clock time
        cpu_start_time = psutil.Process(os.getpid()).cpu_times().user  


    path = [start]
    cost = 0
    N = adj_matrix.shape[0]
    mask = np.ones(N, dtype=bool)  # boolean values indicating which 
                                   # locations have not been visited
    mask[start] = False

    nodes_expanded = 0

    for i in range(N-1):
        last = path[-1]
        next_ind = np.argmin(adj_matrix[last][mask]) 
        next_loc = np.arange(N)[mask][next_ind] 
        path.append(int(next_loc))
        mask[next_loc] = False
        cost += adj_matrix[last, next_loc]
        nodes_expanded += 1

    # this basically prevents the file from being written if the method is being run by 2-opt method
    if track_time:
    
        real_end_time = time.time()  
        cpu_end_time = psutil.Process(os.getpid()).cpu_times().user 

        # Calculate CPU run time and real-world (wall clock) run time
        cpu_run_time = cpu_end_time - cpu_start_time
        real_run_time = real_end_time - real_start_time

        with open('nearest_neighbor.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"Total cost: {cost}, Nodes expanded: {nodes_expanded}, CPU Run Time: {cpu_run_time:.6f} seconds, Real-World Run Time: {real_run_time:.6f} seconds"])


    return path, cost





def nearest_neighbor_2opt(adj_matrix):

    real_start_time = time.time()  
    cpu_start_time = psutil.Process(os.getpid()).cpu_times().user 

    path, cost = nearest_neighbor(adj_matrix, 0, False)
    optimized_route, nodes_expanded = two_opt(path, adj_matrix)

    real_end_time = time.time()  
    cpu_end_time = psutil.Process(os.getpid()).cpu_times().user  
    cpu_run_time = cpu_end_time - cpu_start_time
    real_run_time = real_end_time - real_start_time

    with open('nearest_neighbor_2opt.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"Total cost: {cost}, Nodes expanded: {nodes_expanded}, CPU Run Time: {cpu_run_time:.6f} seconds, Real-World Run Time: {real_run_time:.6f} seconds"])

    return optimized_route


# edited from the below stackoverflow post
# https://stackoverflow.com/questions/53275314/2-opt-algorithm-to-solve-the-travelling-salesman-problem-in-python
def cost_change(cost_mat, n1, n2, n3, n4):
    return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]

# edited from the same stackoverflow post as above
# https://stackoverflow.com/questions/53275314/2-opt-algorithm-to-solve-the-travelling-salesman-problem-in-python
def two_opt(route, cost_mat):
    best = route
    improved = True
    nodes_expanded = 0
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1: continue
                if cost_change(cost_mat, best[i - 1], best[i], best[j - 1], best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
                    nodes_expanded += 1
        route = best
    return best, nodes_expanded
