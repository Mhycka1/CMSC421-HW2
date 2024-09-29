import sys
import numpy as np
from part1 import make_graph, NN, NN2O, RNN
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
        matrix.append([float(x) for x in line.split()])  
    adjacency_matrix = np.array(matrix)

    print("Running Part 1 algorithms")
    nn_path, nn_cost, nn_nodes_expanded, nn_cpu_run_time, nn_real_run_time = NN(adjacency_matrix, 0, True)
    nn2_path, nn2_cost, nn2_nodes_expanded, nn2_cpu_run_time, nn2_real_run_time = NN2O(adjacency_matrix, True)
    rnn_path, rnn_cost, rnn_nodes_expanded, rnn_cpu_run_time, rnn_real_run_time = RNN(adjacency_matrix, 10, 3, True)

    print("Running Part 2 algorithms")
    a_path, a_cost, a_nodes_expanded, a_cpu, a_real = A_MST(adjacency_matrix, True)

    print("Running Part 3 algorithms")
    h_path, h_cost, h_nodes_expanded, h_cpu, h_real = hillClimbing(adjacency_matrix, True)
    s_path, s_cost, s_nodes_expanded, s_cpu, s_real = simuAnnealing(adjacency_matrix, True)
    g_path, g_cost, g_nodes_expanded, g_cpu, g_real = genetic(adjacency_matrix, True)
        

if __name__ == '__main__':
    main()