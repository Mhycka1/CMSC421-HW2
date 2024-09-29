import time
import psutil
import os
import random
import numpy as np
import csv
import math
from part2 import calculate_stats, A_MST
import matplotlib.pyplot as plt

#helper method based on chatgpt generated method
def value(adj_matrix, path):
    total_cost = 0
    for i in range(len(path) - 1):
        total_cost += adj_matrix[path[i]][path[i + 1]]
    # Add cost to return to the starting city
    total_cost += adj_matrix[path[-1]][path[0]]
    return total_cost

#helper method based on chatgpt generated method
def get_neighbors(path):
    neighbors = []
    N = len(path)
    for i in range(1, N - 1):  # Avoid swapping the starting city
        for j in range(i + 1, N):
            new_path = path[:]
            new_path[i], new_path[j] = new_path[j], new_path[i]
            neighbors.append(new_path)
    return neighbors

#method adapted from chatgpt prompt
def hillClimbing(adj_matrix, make_file, restarts=1):
    best_path = None
    best_cost = float('inf')
    total_nodes_expanded = 0
    
    real_start_time = time.perf_counter() 
    cpu_start_time = psutil.Process(os.getpid()).cpu_times().user  

    for _ in range(restarts):
        N = adj_matrix.shape[0]
        nodes_expanded = 0
        
        # Start from an initial random path
        current_path = [0] + random.sample(range(1, N), N - 1)

        while True:
            neighbors = get_neighbors(current_path)
            nodes_expanded += len(neighbors)  

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

    real_end_time = time.perf_counter()  
    cpu_end_time = psutil.Process(os.getpid()).cpu_times().user 
    cpu_run_time = cpu_end_time - cpu_start_time
    real_run_time = real_end_time - real_start_time

    if make_file:
        with open('hillClimbing.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"Total cost: {best_cost}, Nodes expanded: {total_nodes_expanded}, "
                             f"CPU Run Time: {cpu_run_time:.6f} seconds, Real-World Run Time: {real_run_time:.6f} seconds"])

    return best_path, float(best_cost), total_nodes_expanded, cpu_run_time, real_run_time


def simuAnnealing(adj_matrix, make_file, restarts=1, initial_temperature=1000, alpha=0.95):
    best_path = None
    best_cost = float('inf')
    total_nodes_expanded = 0
    
    real_start_time = time.perf_counter()  
    cpu_start_time = psutil.Process(os.getpid()).cpu_times().user  

    for _ in range(restarts):
        N = adj_matrix.shape[0]
        nodes_expanded = 0
        
        # Initial random path (starting city remains fixed)
        current_path = [0] + random.sample(range(1, N), N - 1)
        current_cost = value(adj_matrix, current_path)
        

        T = initial_temperature

        while T > 1e-8:  # Stop when temperature is close to zero
            neighbors = get_neighbors(current_path)
            nodes_expanded += len(neighbors)  

            # Randomly select a neighbor
            next_path = random.choice(neighbors)
            next_cost = value(adj_matrix, next_path)

           
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


    real_end_time = time.perf_counter() 
    cpu_end_time = psutil.Process(os.getpid()).cpu_times().user  
    cpu_run_time = cpu_end_time - cpu_start_time
    real_run_time = real_end_time - real_start_time

    if make_file:
        with open('simuAnnealing.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"Total cost: {best_cost}, Nodes expanded: {total_nodes_expanded}, "
                             f"CPU Run Time: {cpu_run_time:.6f} seconds, Real-World Run Time: {real_run_time:.6f} seconds"])

    return best_path, float(best_cost), total_nodes_expanded, cpu_run_time, real_run_time




#helper method based on chatgpt generated method
def selection(population, fitnesses, selection_type):
    if selection_type == "roulette":
        # Roulette wheel selection
        total_fitness = sum(fitnesses)
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fitness in enumerate(fitnesses):
            current += fitness
            if current > pick:
                return population[i]
    elif selection_type == "tournament":
        # Tournament selection
        tournament_size = 3
        selected = random.sample(list(zip(population, fitnesses)), tournament_size)
        selected.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness (maximize)
        return selected[0][0]

# Crossover function: Performs ordered crossover
def crossover(parent1, parent2, length):
    N = len(parent1)
    start = random.randint(0, N - length)
    child = [-1] * N
    
    # Copy a slice from parent1
    child[start:start + length] = parent1[start:start + length]

    # Fill in the rest from parent2
    idx = 0
    for city in parent2:
        if city not in child:
            while child[idx] != -1:
                idx += 1
            child[idx] = city
    
    return child

#helper method based on chatgpt generated method
def mutate(path, mutation_rate):
    if random.uniform(0, 1) < mutation_rate:
        i, j = random.sample(range(1, len(path)), 2)  # Avoid starting city
        path[i], path[j] = path[j], path[i]  # Swap two cities

#method based on chatgpt prompt
def genetic(adj_matrix, make_file, generations=100, selection_type="roulette", 
                     crossover_prob=0.8, crossover_length=3, mutation_rate=0.01, population_size=100):
    N = adj_matrix.shape[0]
    population = []

    # Create the initial population (random paths)
    for _ in range(population_size):
        path = [0] + random.sample(range(1, N), N - 1)  
        population.append(path)
    
    # Start overall timing
    real_start_time = time.perf_counter()  
    cpu_start_time = psutil.Process(os.getpid()).cpu_times().user  

    best_path = None
    best_cost = float('inf')
    total_nodes_expanded = 0

    for generation in range(generations):
        fitnesses = [1 / value(adj_matrix, path) for path in population]  # Higher fitness for lower cost

        # New generation
        new_population = []
        
        while len(new_population) < population_size:
            # Selection
            parent1 = selection(population, fitnesses, selection_type)
            parent2 = selection(population, fitnesses, selection_type)
            
            # Crossover
            if random.uniform(0, 1) < crossover_prob:
                child1 = crossover(parent1, parent2, crossover_length)
                child2 = crossover(parent2, parent1, crossover_length)
            else:
                child1, child2 = parent1[:], parent2[:]
            
            # Mutation
            mutate(child1, mutation_rate)
            mutate(child2, mutation_rate)
            
            new_population.append(child1)
            new_population.append(child2)

        # Update population and total nodes expanded
        population = new_population[:population_size]  # Ensure correct population size
        total_nodes_expanded += population_size  

        # Evaluate new population and update best solution
        for path in population:
            current_cost = value(adj_matrix, path)
            if current_cost < best_cost:
                best_cost = current_cost
                best_path = path

    real_end_time = time.perf_counter() 
    cpu_end_time = psutil.Process(os.getpid()).cpu_times().user  
    cpu_run_time = cpu_end_time - cpu_start_time
    real_run_time = real_end_time - real_start_time

    if make_file:
        with open('genetic.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"Total cost: {best_cost}, Nodes expanded: {total_nodes_expanded}, "
                             f"CPU Run Time: {cpu_run_time:.6f} seconds, Real-World Run Time: {real_run_time:.6f} seconds"])

    return best_path, float(best_cost), total_nodes_expanded, cpu_run_time, real_run_time


def run_hill_climbing(size_graphs, restarts=1):
    results = [] 

    for graph in size_graphs:
        path, cost, expanded_val, cpu_val, real_val = hillClimbing(graph, False, restarts)
        results.append((cost, expanded_val, cpu_val, real_val))

    costs, expanded, cpu, real = zip(*results) 
    stats = {
        'hillClimbing': {
            'costs': calculate_stats(costs),
            'expanded': calculate_stats(expanded),
        }
    }

    return results

def run_simuAnnealing(size_graphs, restarts=1, initial_temp=1000, alpha=0.95):
    results = []  

    for graph in size_graphs:
        path, cost, expanded_val, cpu_val, real_val = simuAnnealing(graph, False, restarts, initial_temp, alpha)
        results.append((cost, expanded_val, cpu_val, real_val))

    costs, expanded, cpu, real = zip(*results)  
    stats = {
        'simuAnnealing': {
            'costs': calculate_stats(costs),
            'expanded': calculate_stats(expanded),
        }
    }

    return results

def run_genetic(size_graphs, generations=100, approach="roulette", cross_prob=0.8, cross_length=3, mutation_rate=0.01):
    results = []  

    for graph in size_graphs:
        path, cost, expanded_val, cpu_val, real_val = genetic(graph, False, generations, approach, cross_prob, cross_length, mutation_rate, 100)
        results.append((cost, expanded_val, cpu_val, real_val))

    costs, expanded, cpu, real = zip(*results)  
    stats = {
        'simuAnnealing': {
            'costs': calculate_stats(costs),
            'expanded': calculate_stats(expanded),
        }
    }

    return results

def run_Astar_part3(size_graphs):
    results = []  

    for graph in size_graphs:
        path, cost, expanded_val, cpu_val, real_val = A_MST(graph)
        results.append((cost, expanded_val, cpu_val, real_val))

    costs, expanded, cpu, real = zip(*results)  
    stats = {
        'A_star': {
            'costs': calculate_stats(costs),
            'expanded': calculate_stats(expanded),
        }
    }

    return results


def plot_algorithm_performance(algorithm_name, algorithm_results, astar_results, output_dir):
    sizes = ['5', '6', '7']
    avg_diffs, min_diffs, max_diffs = [], [], []
    cpu_times = []

    for size in sizes:
        astar_cost = astar_results[size][0] 
        algorithm_costs = algorithm_results[size]
        
        # Compute differences in cost
        diffs = [cost_data[0] - astar_cost[0] for cost_data in algorithm_costs]
        cpu_times_for_size = [cost_data[2] for cost_data in algorithm_costs]  

        avg_diffs.append(sum(diffs) / len(diffs))
        min_diffs.append(min(diffs))
        max_diffs.append(max(diffs))
        cpu_times.append(sum(cpu_times_for_size) / len(cpu_times_for_size)) 
    

    plt.figure(figsize=(10, 6))
    plt.plot(cpu_times, avg_diffs, marker='o', color='b', label="AVG", linestyle='-')
    plt.fill_between(cpu_times, min_diffs, max_diffs, color='b', alpha=0.2, label="MIN/MAX")
    plt.title(f"{algorithm_name} Performance vs A*")
    plt.xlabel("CPU Runtime")
    plt.ylabel("Cost Difference (Algorithm - A*)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{algorithm_name.lower().replace(' ', '_')}_performance.png"))
    plt.close()