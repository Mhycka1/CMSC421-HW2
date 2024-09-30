import heapq
import time
import numpy as np
import csv
import networkx as nx 
from part1 import make_graph, calculate_stats, NN, NN2O, RNN
import os
import matplotlib.pyplot as plt

def mst_heuristic(adj_matrix, unvisited):
    if len(unvisited) <= 1:
        return 0  
    
    subgraph = adj_matrix[np.ix_(unvisited, unvisited)]
    graph = nx.Graph()
    
    for i in range(len(unvisited)):
        for j in range(i + 1, len(unvisited)):
            if subgraph[i, j] > 0:
                graph.add_edge(unvisited[i], unvisited[j], weight=subgraph[i, j])
    
    mst = nx.minimum_spanning_tree(graph)
    
    return mst.size(weight='weight')

# A* with MST heuristic
# adapted from code on below site
#https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2
def A_MST(adj_matrix, make_file=False):
    N = adj_matrix.shape[0]
    start_city = 0
    
    # Priority queue with elements as (cost, current_city, visited_cities, path)
    pq = []
    initial_state = (0, start_city, [start_city], [start_city])
    heapq.heappush(pq, (0, initial_state))
    
    visited_states = set()  
    nodes_expanded = 0  
   
    start_cpu_time = time.process_time() 
    start_real_time = time.time() 

    while pq:
        # Pop the state with the smallest f(n) = g(n) + h(n)
        current_cost, (g_n, current_city, visited, path) = heapq.heappop(pq)

        state_tuple = (current_city, tuple(visited))
        if state_tuple in visited_states:
            continue
        visited_states.add(state_tuple)
        
        nodes_expanded += 1

        # you've hit the goal if all cities are visited and you're at start
        if len(visited) == N:
            return_to_start_cost = adj_matrix[current_city][start_city]
            total_cost = g_n  + return_to_start_cost
            
            cpu_runtime = time.process_time() - start_cpu_time
            real_runtime = time.time() - start_real_time

            if make_file:
                with open('A_star', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([f"Best Total cost: {total_cost}, Total Nodes expanded: {nodes_expanded}, CPU Run Time: {cpu_runtime:.6f} seconds, Real-World Run Time: {real_runtime:.6f} seconds"])
            
            return path, total_cost, nodes_expanded, cpu_runtime, real_runtime
        
        for next_city in range(N):
            if next_city not in visited:
                # Calculate g(n) (current path cost)
                new_g_n = g_n + adj_matrix[current_city][next_city]
                
                # h(n): Calculate the MST heuristic for the remaining unvisited cities
                unvisited = [city for city in range(N) if city not in visited]
                mst_cost = mst_heuristic(adj_matrix, unvisited) if unvisited else 0
                f_n = new_g_n + mst_cost
                
                new_state = (new_g_n, next_city, visited + [next_city], path + [next_city])
                heapq.heappush(pq, (f_n, new_state))

    cpu_runtime = time.process_time() - start_cpu_time
    real_runtime = time.time() - start_real_time

    return None, float('inf'), nodes_expanded, cpu_runtime, real_runtime

def run_A_star(size_graphs):
    results = []  # Store (cost, expanded, cpu, real) for each graph

    for graph in size_graphs:
        path, cost, expanded_val, cpu_val, real_val = A_MST(graph)
        results.append((cost, expanded_val))

    costs, expanded = zip(*results)
    stats = {
        'A_star': {
            'costs': calculate_stats(costs),
            'expanded': calculate_stats(expanded),
        }
    }

    return results



def run_nn(size_graphs):

    results = [] 

    for graph in size_graphs:
        path, cost, expanded_val, cpu_val, real_val = NN(graph, 0, False)
        results.append((cost, expanded_val))

    costs, expanded= zip(*results) 
    stats = {
        'nn': {
            'costs': calculate_stats(costs),
            'expanded': calculate_stats(expanded),
        }
    }

    return results

def run_nn2o(size_graphs):
    results = [] 

    for graph in size_graphs:
        path, cost, expanded_val, cpu_val, real_val = NN2O(graph, False)
        results.append((cost, expanded_val))

    costs, expanded = zip(*results)  
    stats = {
        'nn': {
            'costs': calculate_stats(costs),
            'expanded': calculate_stats(expanded),
        }
    }

    return results

def run_rnn(size_graphs):
    results = []  
    for graph in size_graphs:
        path, cost, expanded_val, cpu_val, real_val = RNN(adj_matrix=graph, make_file=False)
        results.append((cost, expanded_val))


    costs, expanded = zip(*results)
    stats = {
        'nn': {
            'costs': calculate_stats(costs),
            'expanded': calculate_stats(expanded),
        }
    }

    return results

def compute_differences(a_results, other_results):
    cost_diffs = []
    expanded_diffs = []
    
    for a_result, other_result in zip(a_results, other_results):
        a_cost, a_expanded = a_result
        other_cost, other_expanded = other_result
        
        cost_diffs.append(a_cost - other_cost) 
        expanded_diffs.append(a_expanded - other_expanded) 
    
    return cost_diffs, expanded_diffs


def make_part2_graphs(statistics):
    output_dir = 'part2_result_graphs'
    os.makedirs(output_dir, exist_ok=True)

    sizes = ['5', '6', '7', '8', '9', '10']
    algorithms = ['nn', 'nn2o', 'rnn']  

    plt.figure(figsize=(12, 6))

    colors = {'nn': 'blue', 'nn2o': 'green', 'rnn': 'red'}
    styles = {'nn': '-', 'nn2o': '--', 'rnn': '-.'}
    markers = {'nn': 'o', 'nn2o': 's', 'rnn': '^'}

    for algorithm in algorithms:
        avg_costs = [statistics[size][algorithm]['cost_stats']['avg'] for size in sizes]
        min_costs = [statistics[size][algorithm]['cost_stats']['min'] for size in sizes]
        max_costs = [statistics[size][algorithm]['cost_stats']['max'] for size in sizes]
        
        # Plot the average line
        plt.plot(sizes, avg_costs, marker=markers[algorithm], label=f'{algorithm} - Avg', linestyle=styles[algorithm], color=colors[algorithm])

        # Add a small offset for nn2o to prevent exact overlap 
        offset = 0 if algorithm != 'nn2o' else 0.5
        
        # Fill the min/max area with transparency
        plt.fill_between(sizes, [min_c - offset for min_c in min_costs], [max_c - offset for max_c in max_costs], 
                        alpha=0.3, label=f'{algorithm} - Min/Max', color=colors[algorithm], edgecolor='black', linewidth=0.5)

    plt.title('Performance Differences in Total Cost')
    plt.xlabel('Graph Size')
    plt.ylabel('Difference in Cost')
    plt.xticks(sizes)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'total_cost_differences.png'))
    plt.close() 
    plt.figure(figsize=(12, 6))

    for algorithm in algorithms:
        avg_expanded = [statistics[size][algorithm]['expanded_stats']['avg'] for size in sizes]
        min_expanded = [statistics[size][algorithm]['expanded_stats']['min'] for size in sizes]
        max_expanded = [statistics[size][algorithm]['expanded_stats']['max'] for size in sizes]
        
        # Plot the average line
        plt.plot(sizes, avg_expanded, marker=markers[algorithm], label=f'{algorithm} - Avg', linestyle=styles[algorithm], color=colors[algorithm])
        
        # Fill the min/max area with transparency
        plt.fill_between(sizes, min_expanded, max_expanded, alpha=0.3, label=f'{algorithm} - Min/Max', color=colors[algorithm], edgecolor='black', linewidth=0.5)

    plt.title('Performance Differences in Nodes Expanded')
    plt.xlabel('Graph Size')
    plt.ylabel('Difference in Nodes Expanded')
    plt.xticks(sizes)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, 'nodes_expanded_differences.png'))
    plt.close() 