import heapq
import time
import numpy as np
import csv
import networkx as nx 
from part1 import make_graph, calculate_stats


# help adapted from chatgpt prompt
def mst_heuristic(adj_matrix, unvisited):
    """Calculate the Minimum Spanning Tree (MST) cost of the unvisited cities."""
    if len(unvisited) <= 1:
        return 0  # No MST if only one or no city is left to visit
    
    subgraph = adj_matrix[np.ix_(unvisited, unvisited)]
    graph = nx.Graph()
    
    # Add edges and weights to the graph
    for i in range(len(unvisited)):
        for j in range(i + 1, len(unvisited)):
            if subgraph[i, j] > 0:
                graph.add_edge(unvisited[i], unvisited[j], weight=subgraph[i, j])
    
    # Calculate the MST 
    mst = nx.minimum_spanning_tree(graph)
    
    return mst.size(weight='weight')

# A* with MST heuristic
def A_MST(adj_matrix):
    N = adj_matrix.shape[0]
    start_city = 0
    
    # Priority queue with elements as (cost, current_city, visited_cities, path)
    pq = []
    initial_state = (0, start_city, [start_city], [start_city])
    heapq.heappush(pq, (0, initial_state))
    
    visited_states = set()  # To track visited states
    nodes_expanded = 0  # Track the number of nodes expanded

    # Track CPU and real-world time
    start_cpu_time = time.process_time()  # CPU time
    start_real_time = time.time()  # Real-world time

    while pq:
        # Pop the state with the smallest f(n) = g(n) + h(n)
        current_cost, (g_n, current_city, visited, path) = heapq.heappop(pq)

        state_tuple = (current_city, tuple(visited))
        if state_tuple in visited_states:
            continue
        visited_states.add(state_tuple)
        
        # Increment the node expansion counter
        nodes_expanded += 1

        # If all cities are visited and we are back at the start, goal state is achieved
        if len(visited) == N:
            # Add the cost to return to the start city to complete the tour
            return_to_start_cost = adj_matrix[current_city][start_city]
            total_cost = g_n  # We're not adding return_to_start_cost
            
            # Calculate CPU and real-world runtime
            cpu_runtime = time.process_time() - start_cpu_time
            real_runtime = time.time() - start_real_time
            
            return path, total_cost, nodes_expanded, cpu_runtime, real_runtime
        
        # Expand successors (visit next city)
        for next_city in range(N):
            if next_city not in visited:
                # Calculate g(n) (current path cost)
                new_g_n = g_n + adj_matrix[current_city][next_city]
                
                # h(n): Calculate the MST heuristic for the remaining unvisited cities
                unvisited = [city for city in range(N) if city not in visited]
                mst_cost = mst_heuristic(adj_matrix, unvisited) if unvisited else 0
                
                # Calculate f(n) = g(n) + h(n)
                f_n = new_g_n + mst_cost
                
                # Create new state with the next city visited
                new_state = (new_g_n, next_city, visited + [next_city], path + [next_city])
                
                # Push the new state into the priority queue
                heapq.heappush(pq, (f_n, new_state))
    
    # If no solution found, return large values for cost and zero for times
    cpu_runtime = time.process_time() - start_cpu_time
    real_runtime = time.time() - start_real_time
    return None, float('inf'), nodes_expanded, cpu_runtime, real_runtime

def run_A_star(size_graphs, size_label):
    # Initialize lists for storing results
    costs, expanded, cpu, real = [], [], [], []

    # Loop through each graph in the family
    for graph in size_graphs:
        # Run A* algorithm and append results
        path, cost, expanded_val, cpu_val, real_val = A_MST(graph)
        
        # Append results for each graph
        costs.append(cost)
        expanded.append(expanded_val)
        cpu.append(cpu_val)
        real.append(real_val)

    # Calculate statistics for A* algorithm for this size
    stats = {
        'A_star': {
            'costs': calculate_stats(costs),
            'expanded': calculate_stats(expanded),
            'cpu': calculate_stats(cpu),
            'real': calculate_stats(real)
        }
    }

    # Output the stats to a CSV file (optional)
    with open(f'{size_label}_A*_stats.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

        # Define a custom dialect with a comma delimiter
        class CommaDialect(csv.Dialect):
            delimiter = ','  # Comma as the delimiter
            quoting = csv.QUOTE_MINIMAL
            quotechar = '"'  # Add a quote character
            lineterminator = '\n'

        writer = csv.writer(file, dialect=CommaDialect)
        writer.writerow(["Statistic", "Average", "Minimum", "Maximum"])

        # Writing A* stats
        writer.writerow(["A* Cost", stats['A_star']['costs']['avg'], stats['A_star']['costs']['min'], stats['A_star']['costs']['max']])
        writer.writerow(["A* Nodes Expanded", stats['A_star']['expanded']['avg'], stats['A_star']['expanded']['min'], stats['A_star']['expanded']['max']])
        writer.writerow(["A* CPU Time", stats['A_star']['cpu']['avg'], stats['A_star']['cpu']['min'], stats['A_star']['cpu']['max']])
        writer.writerow(["A* Real Time", stats['A_star']['real']['avg'], stats['A_star']['real']['min'], stats['A_star']['real']['max']])

    return stats