import time
import psutil
import os
import random
import numpy as np
import csv


def value(adj_matrix, path):
    """Calculate the total cost of a given path."""
    total_cost = 0
    for i in range(len(path) - 1):
        total_cost += adj_matrix[path[i]][path[i + 1]]
    # Add cost to return to the starting city
    total_cost += adj_matrix[path[-1]][path[0]]
    return total_cost

def get_neighbors(path):
    """Generate neighboring paths by swapping two cities."""
    neighbors = []
    N = len(path)
    for i in range(1, N - 1):  # Avoid swapping the starting city
        for j in range(i + 1, N):
            new_path = path[:]
            new_path[i], new_path[j] = new_path[j], new_path[i]
            neighbors.append(new_path)
    return neighbors

def hillClimbing(adj_matrix, make_file, restarts=1):
    best_path = None
    best_cost = float('inf')
    total_nodes_expanded = 0
    
    # Start the overall timing
    real_start_time = time.time()  # Wall clock time
    cpu_start_time = psutil.Process(os.getpid()).cpu_times().user  # CPU time

    for _ in range(restarts):
        N = adj_matrix.shape[0]
        nodes_expanded = 0
        
        # Start from an initial random path (starting city remains fixed)
        current_path = [0] + random.sample(range(1, N), N - 1)

        while True:
            neighbors = get_neighbors(current_path)
            nodes_expanded += len(neighbors)  # Count nodes expanded (neighbors generated)

            if not neighbors:
                break

            # Choose the best neighbor (minimize cost)
            best_neighbor = min(neighbors, key=lambda path: value(adj_matrix, path))
            if value(adj_matrix, best_neighbor) >= value(adj_matrix, current_path):
                break  # Stop if no better neighbor is found

            current_path = best_neighbor

        # Check if the current path is better than the best one found so far
        current_cost = value(adj_matrix, current_path)
        if current_cost < best_cost:
            best_cost = current_cost
            best_path = current_path
        
        total_nodes_expanded += nodes_expanded

    # Calculate overall timing
    real_end_time = time.time()  # Wall clock time
    cpu_end_time = psutil.Process(os.getpid()).cpu_times().user  # CPU time
    cpu_run_time = cpu_end_time - cpu_start_time
    real_run_time = real_end_time - real_start_time

    # If make_file is True, write results to CSV
    if make_file:
        with open('hillClimbing.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"Total cost: {best_cost}, Nodes expanded: {total_nodes_expanded}, "
                             f"CPU Run Time: {cpu_run_time:.6f} seconds, Real-World Run Time: {real_run_time:.6f} seconds"])

    return best_path, best_cost, total_nodes_expanded, cpu_run_time, real_run_time


