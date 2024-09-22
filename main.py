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

    # path, cost = nearest_neighbor(adjacency_matrix, 0, True)
    # best = nearest_neighbor_2opt(adjacency_matrix)
    # test_path, test_cost = repeated_randomized_nearest_neighbor_2opt(adjacency_matrix, 10, 3)

    # path, cost = A_MST(adjacency_matrix)
    #lol = genetic(adjacency_matrix, True)

    #part 1 of the assignment
   
        

if __name__ == '__main__':
    main()