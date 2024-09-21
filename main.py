import sys
import numpy as np
from part1 import make_graph, nearest_neighbor, nearest_neighbor_2opt, repeated_randomized_nearest_neighbor_2opt
from part2 import A_MST
from part3 import hillClimbing, simuAnnealing


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

    # path, cost = nearest_neighbor(adjacency_matrix, 0, True)
    # best = nearest_neighbor_2opt(adjacency_matrix)
    # test_path, test_cost = repeated_randomized_nearest_neighbor_2opt(adjacency_matrix, 10, 3)

    # path, cost = A_MST(adjacency_matrix)
    lol = simuAnnealing(adjacency_matrix, True)

    #part 1 of the assignment
    # size_5_graphs = [] 
    # size_10_graphs = []
    # size_15_graphs = []
    # size_20_graphs = []
    # size_25_graphs = []
    # size_30_graphs = []

    # for i in range(30):
    #     size_5_graphs.append(make_graph(5))
    #     size_10_graphs.append(make_graph(10))
    #     size_15_graphs.append(make_graph(15))
    #     size_20_graphs.append(make_graph(20))
    #     size_25_graphs.append(make_graph(25))
    #     size_30_graphs.append(make_graph(30))
        

if __name__ == '__main__':
    main()