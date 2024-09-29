import random
import numpy as np
import networkx as nx
from py2opt.routefinder import RouteFinder
import time
import csv
import os
import psutil  # for CPU time
import matplotlib.pyplot as plt



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
def NN(adj_matrix, start, make_file):
    real_start_time = time.time()  # Wall clock time
    cpu_start_time = psutil.Process(os.getpid()).cpu_times().user  

    path = [start]
    cost = 0
    N = adj_matrix.shape[0]
    mask = np.ones(N, dtype=bool)  # Boolean values indicating which locations have not been visited
    mask[start] = False

    nodes_expanded = 0

    for i in range(N - 1):
        last = path[-1]
        unvisited_nodes = np.arange(N)[mask]
        unvisited_distances = adj_matrix[last][mask]

        next_ind = np.argmin(unvisited_distances)  # Get the index of the nearest unvisited node
        next_loc = unvisited_nodes[next_ind]  # Get the node number
        path.append(int(next_loc))
        mask[next_loc] = False
        cost += adj_matrix[last, next_loc]
        nodes_expanded += 1

    # Return to the starting node to complete the tour
    cost += adj_matrix[path[-1], start]
    path.append(start)

    real_end_time = time.time()  
    cpu_end_time = psutil.Process(os.getpid()).cpu_times().user 

    # Calculate CPU run time and real-world (wall clock) run time
    cpu_run_time = cpu_end_time - cpu_start_time
    real_run_time = real_end_time - real_start_time

    if make_file:
        with open('NN.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"Total cost: {cost}, Nodes expanded: {nodes_expanded}, CPU Run Time: {cpu_run_time:.6f} seconds, Real-World Run Time: {real_run_time:.6f} seconds"])

    return path, cost, nodes_expanded, cpu_run_time, real_run_time




# might need to keep track of the cost from 2-opt and add it, idk
def NN2O(adj_matrix, make_file):

    real_start_time = time.time()  
    cpu_start_time = psutil.Process(os.getpid()).cpu_times().user 

    path, cost, nn_expanded, nn_cpu_runtime, nn_real_runtime = NN(adj_matrix, 0, False)
    optimized_route, two_opt_expanded = two_opt(path, adj_matrix)


    real_end_time = time.time()  
    cpu_end_time = psutil.Process(os.getpid()).cpu_times().user  
    cpu_run_time = cpu_end_time - cpu_start_time
    real_run_time = real_end_time - real_start_time

    if make_file:
        with open('NN2O.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([f"Total cost: {cost}, Nodes expanded: {two_opt_expanded + nn_expanded}, CPU Run Time: {cpu_run_time:.6f} seconds, Real-World Run Time: {real_run_time:.6f} seconds"])

    return optimized_route, cost, two_opt_expanded + nn_expanded, cpu_run_time, real_run_time


# edited from the below stackoverflow post
# https://stackoverflow.com/questions/53275314/2-opt-algorithm-to-solve-the-travelling-salesman-problem-in-python
def cost_change(cost_mat, n1, n2, n3, n4):
    return cost_mat[n1][n3] + cost_mat[n2][n4] - cost_mat[n1][n2] - cost_mat[n3][n4]

# edited from the same stackoverflow post as above
# https://stackoverflow.com/questions/53275314/2-opt-algorithm-to-solve-the-travelling-salesman-problem-in-python
def two_opt(route, cost_mat):
    def total_cost(route, cost_mat):
        return sum(cost_mat[route[i - 1], route[i]] for i in range(1, len(route)))

    best = route[:]
    best_cost = total_cost(best, cost_mat)
    improved = True
    nodes_expanded = 0

    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best)):
                if j - i == 1:
                    continue

                # Calculate cost change if we reverse the segment best[i:j]
                current_cost = (
                    cost_mat[best[i - 1], best[i]] +
                    cost_mat[best[j - 1], best[j]]
                )
                new_cost = (
                    cost_mat[best[i - 1], best[j - 1]] +
                    cost_mat[best[i], best[j]]
                )
                
                if new_cost < current_cost:
                    # Reverse the segment and set improved to True
                    best[i:j] = best[j - 1:i - 1:-1]
                    best_cost = best_cost - current_cost + new_cost
                    improved = True
                    nodes_expanded += 1

                    # Break out of the loop early to restart after finding an improvement
                    break
            if improved:
                break

    return best, nodes_expanded


# adapted from a chatgpt prompt asking to adapt my above NN and NN2O code 
# into an RNN algorithm
def RNN(adj_matrix, iterations=10, n=3, make_file=True):
    best_path = None
    best_cost = float('inf')
    total_nodes_expanded = 0

    real_start_time = time.time()  
    cpu_start_time = psutil.Process(os.getpid()).cpu_times().user  # CPU time

    N = adj_matrix.shape[0]

    for start_node in range(iterations):
        path = [start_node % N]
        cost = 0
        mask = np.ones(N, dtype=bool) 
        mask[start_node % N] = False

        path_nodes_expanded = 0
        opt_nodes_expanded = 0

        # Construct an initial path using a randomized nearest neighbor approach
        for i in range(N - 1):
            last = path[-1]
            unvisited_nodes = np.arange(N)[mask]
            unvisited_distances = adj_matrix[last][mask]

            # Get 'n' nearest nodes randomly
            nearest_n_indices = np.argsort(unvisited_distances)[:n]
            next_ind = random.choice(nearest_n_indices)  # Randomly choose one of the nearest nodes
            next_loc = unvisited_nodes[next_ind]

            path.append(int(next_loc))
            mask[next_loc] = False
            cost += adj_matrix[last, next_loc]
            path_nodes_expanded += 1

        # Complete the tour by returning to the starting node
        cost += adj_matrix[path[-1], path[0]]
        path.append(path[0])

        # 2-Opt Optimization
        max_iterations = 1000  # Limit the total number of iterations for the 2-Opt process
        iteration_counter = 0
        best_route = path[:]
        improved = True

        while improved and iteration_counter < max_iterations:
            improved = False
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route)):
                    if j - i == 1:
                        continue
                    if cost_change(adj_matrix, best_route[i - 1], best_route[i], best_route[j - 1], best_route[j]) < 0:
                        best_route[i:j] = best_route[j - 1:i - 1:-1]
                        improved = True
                        opt_nodes_expanded += 1
                        iteration_counter += 1

                    if iteration_counter >= max_iterations:
                        break
                if iteration_counter >= max_iterations:
                    break

        total_cost = sum(adj_matrix[best_route[i], best_route[i + 1]] for i in range(len(best_route) - 1))

        if total_cost < best_cost:
            best_cost = total_cost
            best_path = best_route

    real_end_time = time.time()
    cpu_end_time = psutil.Process(os.getpid()).cpu_times().user
    cpu_run_time = cpu_end_time - cpu_start_time
    real_run_time = real_end_time - real_start_time

    total_nodes_expanded = path_nodes_expanded + opt_nodes_expanded

    if make_file:
        with open('RNN.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([f"Best Total cost: {best_cost}, Total Nodes expanded: {total_nodes_expanded}, CPU Run Time: {cpu_run_time:.6f} seconds, Real-World Run Time: {real_run_time:.6f} seconds"])

    return best_path, best_cost, total_nodes_expanded, cpu_run_time, real_run_time

def calculate_stats(data):
    return {
        'avg': round(float(np.mean(data)), 6),
        'min': round(float(np.min(data)), 6),
        'max': round(float(np.max(data)), 6)
    }




def process_graph_family(size_graphs, size_label):
    # Initialize lists for storing results
    nn_costs, nn_expanded, nn_cpu, nn_real = [], [], [], []
    nn2o_costs, nn2o_expanded, nn2o_cpu, nn2o_real = [], [], [], []
    rnn_costs, rnn_expanded, rnn_cpu, rnn_real = [], [], [], []
    rnn2_costs, rnn2_expanded, rnn2_cpu, rnn2_real = [], [], [], []
    rnn4_costs, rnn4_expanded, rnn4_cpu, rnn4_real = [], [], [], []

    # Loop through each graph in the family
    for graph in size_graphs:  # Iterate directly over size_graphs
        # Run algorithms and append results
        nn_path, nn_cost, nn_expanded_val, nn_cpu_val, nn_real_val = NN(graph, 0, False)
        nn2o_path, nn2o_cost, nn2o_expanded_val, nn2o_cpu_val, nn2o_real_val = NN2O(graph, False)
        rnn_path, rnn_cost, rnn_expanded_val, rnn_cpu_val, rnn_real_val = RNN(graph, n=3, make_file=False)
        rnn2_path, rnn2_cost, rnn2_expanded_val, rnn2_cpu_val, rnn2_real_val = RNN(graph, n=2, make_file=False)
        rnn4_path, rnn4_cost, rnn4_expanded_val, rnn4_cpu_val, rnn4_real_val = RNN(graph, n=4, make_file=False)

        # Append results for size X graphs
        nn_costs.append(nn_cost)
        nn_expanded.append(nn_expanded_val)
        nn_cpu.append(nn_cpu_val)
        nn_real.append(nn_real_val)

        nn2o_costs.append(nn2o_cost)
        nn2o_expanded.append(nn2o_expanded_val)
        nn2o_cpu.append(nn2o_cpu_val)
        nn2o_real.append(nn2o_real_val)

        rnn_costs.append(rnn_cost)
        rnn_expanded.append(rnn_expanded_val)
        rnn_cpu.append(rnn_cpu_val)
        rnn_real.append(rnn_real_val)

        rnn2_costs.append(rnn2_cost)
        rnn2_expanded.append(rnn2_expanded_val)
        rnn2_cpu.append(rnn2_cpu_val)
        rnn2_real.append(rnn2_real_val)

        rnn4_costs.append(rnn4_cost)
        rnn4_expanded.append(rnn4_expanded_val)
        rnn4_cpu.append(rnn4_cpu_val)
        rnn4_real.append(rnn4_real_val)

    # Calculate statistics for each algorithm for this size
    stats = {
        'nn': {
            'costs': calculate_stats(nn_costs),
            'expanded': calculate_stats(nn_expanded),
            'cpu': calculate_stats(nn_cpu),
            'real': calculate_stats(nn_real)
        },
        'nn2o': {
            'costs': calculate_stats(nn2o_costs),
            'expanded': calculate_stats(nn2o_expanded),
            'cpu': calculate_stats(nn2o_cpu),
            'real': calculate_stats(nn2o_real)
        },
        'rnn': {
            'costs': calculate_stats(rnn_costs),
            'expanded': calculate_stats(rnn_expanded),
            'cpu': calculate_stats(rnn_cpu),
            'real': calculate_stats(rnn_real)
        },
        'rnn2': {
            'costs': calculate_stats(rnn2_costs),
            'expanded': calculate_stats(rnn2_expanded),
            'cpu': calculate_stats(rnn2_cpu),
            'real': calculate_stats(rnn2_real)
        },
        'rnn4': {
            'costs': calculate_stats(rnn4_costs),
            'expanded': calculate_stats(rnn4_expanded),
            'cpu': calculate_stats(rnn4_cpu),
            'real': calculate_stats(rnn4_real)
        }
    }

    # Output stats to CSV file 
    # below file writer bit is from chatgpt since i couldn't figure out how to do it
    with open(f'{size_label}_stats.csv', mode='w', newline='') as file:
        writer = csv.writer(file)

  
        class CommaDialect(csv.Dialect):
            delimiter = ',' 
            quoting = csv.QUOTE_MINIMAL
            quotechar = '"' 
            lineterminator = '\n'

        writer = csv.writer(file, dialect=CommaDialect)
        writer.writerow(["Statistic", "Average", "Minimum", "Maximum"])

        # Writing all algorithm stats in a loop for cleanliness
        for algorithm in stats:
            writer.writerow([f"{algorithm.upper()} Cost", stats[algorithm]['costs']['avg'], stats[algorithm]['costs']['min'], stats[algorithm]['costs']['max']])
            writer.writerow([f"{algorithm.upper()} Nodes Expanded", stats[algorithm]['expanded']['avg'], stats[algorithm]['expanded']['min'], stats[algorithm]['expanded']['max']])
            writer.writerow([f"{algorithm.upper()} CPU Time", stats[algorithm]['cpu']['avg'], stats[algorithm]['cpu']['min'], stats[algorithm]['cpu']['max']])
            writer.writerow([f"{algorithm.upper()} Real Time", stats[algorithm]['real']['avg'], stats[algorithm]['real']['min'], stats[algorithm]['real']['max']])

    return stats

def read_stats(file_name):
    costs, nodes_expanded, cpu_time, real_time = {}, {}, {}, {}
    
    with open(file_name, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Check if necessary fields are present
            if 'Statistic' not in row or 'Average' not in row or 'Minimum' not in row or 'Maximum' not in row:
                print(f"Skipping malformed row: {row}")
                continue
            
            algorithm = row['Statistic'].split()[0]  

            if "Cost" in row['Statistic']:
                costs[algorithm] = [float(row['Average']), float(row['Minimum']), float(row['Maximum'])]
            elif "Nodes Expanded" in row['Statistic']:
                nodes_expanded[algorithm] = [float(row['Average']), float(row['Minimum']), float(row['Maximum'])]
            elif "CPU Time" in row['Statistic']:
                cpu_time[algorithm] = [float(row['Average']), float(row['Minimum']), float(row['Maximum'])]
            elif "Real Time" in row['Statistic']:
                real_time[algorithm] = [float(row['Average']), float(row['Minimum']), float(row['Maximum'])]

    return costs, nodes_expanded, cpu_time, real_time



def make_part1_graphs():
    sizes = [5, 10, 15, 20, 25, 30]
    all_costs, all_nodes, all_cpu, all_real = {}, {}, {}, {}

    for size in sizes:
        file_name = f'Size_{size}_stats.csv'
        costs, nodes, cpu, real = read_stats(file_name)

        all_costs[size] = costs
        all_nodes[size] = nodes
        all_cpu[size] = cpu
        all_real[size] = real

    output_dir = 'part1_result_graphs'
    os.makedirs(output_dir, exist_ok=True)

    color_map = {
        'NN': 'tab:blue',
        'NN2O': 'tab:orange',
        'RNN': 'tab:green',
        'RNN2': 'tab:red',
        'RNN4': 'tab:purple',
    }

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('Graph Size')
    ax1.set_ylabel('Total Cost')

    # Plot costs
    for algorithm in all_costs[5]:  
        color = color_map.get(algorithm.split()[0], 'black')  
        ax1.plot(sizes, [all_costs[size][algorithm][0] for size in sizes], 
                    label=f'{algorithm} Cost', marker='o', color=color)

    ax1.tick_params(axis='y')

    # Create second y-axis with the same values as the x-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Graph Size')


    ax2.plot(sizes, sizes, linestyle='--', color='gray', label='Graph Size (Reference)')
    ax2.tick_params(axis='y')
    fig.tight_layout()
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Total Cost and Graph Size for Different Algorithms')
    plt.savefig(os.path.join(output_dir, 'cost_and_nodes.png'))
    plt.close()  

    # Plot for CPU and Real-World Runtime
    fig, ax3 = plt.subplots()

    ax3.set_xlabel('Graph Size')
    ax3.set_ylabel('CPU Time')

    for algorithm in all_cpu[5]:
        color = color_map.get(algorithm.split()[0], 'black') 
        ax3.plot(sizes, [all_cpu[size][algorithm][0] for size in sizes], 
                    label=f'{algorithm} CPU Time', marker='o', color=color)

    ax3.tick_params(axis='y')

    ax4 = ax3.twinx()
    ax4.set_ylabel('Real Time')

    for algorithm in all_real[5]:
        color = color_map.get(algorithm.split()[0], 'black')  
        ax4.plot(sizes, [all_real[size][algorithm][0] for size in sizes], 
                    label=f'{algorithm} Real Time', linestyle='--', color=color)

    ax4.tick_params(axis='y')
    fig.tight_layout()
    ax3.legend(loc='upper left')
    ax4.legend(loc='upper right')
    plt.title('CPU Time and Real-World Runtime for Different Algorithms')
    plt.savefig(os.path.join(output_dir, 'cpu_and_real_time.png'))
    plt.close()  

