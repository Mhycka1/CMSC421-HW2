import time
import psutil
import os
import random
import numpy as np
import csv
import math


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
    
    # Start the overall timing with more precise real-world timing
    real_start_time = time.perf_counter()  # Precise real-world time
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
    real_end_time = time.perf_counter()  # Precise real-world time
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


def simuAnnealing(adj_matrix, make_file, restarts=1, initial_temperature=1000, alpha=0.95):
    best_path = None
    best_cost = float('inf')
    total_nodes_expanded = 0
    
    # Start overall timing
    real_start_time = time.perf_counter()  # Precise real-world time
    cpu_start_time = psutil.Process(os.getpid()).cpu_times().user  # CPU time

    for _ in range(restarts):
        N = adj_matrix.shape[0]
        nodes_expanded = 0
        
        # Initial random path (starting city remains fixed)
        current_path = [0] + random.sample(range(1, N), N - 1)
        current_cost = value(adj_matrix, current_path)
        
        # Initial temperature
        T = initial_temperature

        while T > 1e-8:  # Stop when temperature is close to zero
            neighbors = get_neighbors(current_path)
            nodes_expanded += len(neighbors)  # Count nodes expanded

            # Randomly select a neighbor
            next_path = random.choice(neighbors)
            next_cost = value(adj_matrix, next_path)

            # Calculate the cost difference
            delta_cost = next_cost - current_cost

            # Accept the new solution if it's better, or with some probability if it's worse
            if delta_cost < 0 or random.uniform(0, 1) < math.exp(-delta_cost / T):
                current_path = next_path
                current_cost = next_cost

            # Cool down the temperature
            T *= alpha

        # If this run produces a better result, update the best path and cost
        if current_cost < best_cost:
            best_cost = current_cost
            best_path = current_path
        
        total_nodes_expanded += nodes_expanded

    # Calculate overall timing
    real_end_time = time.perf_counter()  # Precise real-world time
    cpu_end_time = psutil.Process(os.getpid()).cpu_times().user  # CPU time
    cpu_run_time = cpu_end_time - cpu_start_time
    real_run_time = real_end_time - real_start_time

    # If make_file is True, write results to CSV
    if make_file:
        with open('simuAnnealing.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"Total cost: {best_cost}, Nodes expanded: {total_nodes_expanded}, "
                             f"CPU Run Time: {cpu_run_time:.6f} seconds, Real-World Run Time: {real_run_time:.6f} seconds"])

    return best_path, best_cost, total_nodes_expanded, cpu_run_time, real_run_time
