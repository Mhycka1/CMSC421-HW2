import sys
import numpy as np
from part1 import make_graph, nearest_neighbor, nearest_neighbor_2opt, repeated_randomized_nearest_neighbor_2opt
from part2 import A_MST
from part3 import hillClimbing, simuAnnealing, genetic


def main():
    # reads the size number in infile
    size = sys.stdin.readline().strip()
    size = int(size)
    matrix = []
    
    # Read the actual matrix
    for i in range(size):
        line = sys.stdin.readline().strip()
        matrix.append([int(x) for x in line.split()])  
    adjacency_matrix = np.array(matrix)

    nn_path, nn_cost, nn_nodes_expanded, nn_cpu_run_time, nn_real_run_time = nearest_neighbor(adjacency_matrix, 0, True)
    nn2_path, nn2_cost, nn2_nodes_expanded, nn2_cpu_run_time, nn2_real_run_time = nearest_neighbor_2opt(adjacency_matrix, True)
    rnn_path, rnn_cost, rnn_nodes_expanded, rnn_cpu_run_time, rnn_real_run_time = repeated_randomized_nearest_neighbor_2opt(adjacency_matrix, 10, 3, True)

    
    print(f'NN nodes expanded: {nn2_nodes_expanded}, NN2O nodes expanded: {nn2_nodes_expanded}, RNN nodes expanded: {rnn_nodes_expanded}')
   
        

if __name__ == '__main__':
    main()