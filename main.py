import sys
import numpy as np
from part1 import make_graph, nearest_neighbor


def main():
    # reads the size number in infile
    size = sys.stdin.readline().strip()
    size = int(size)
    matrix = []
    
    # # Read the actual matrix
    for i in range(size):
        line = sys.stdin.readline().strip()
        matrix.append([int(x) for x in line.split()])  
    adjacency_matrix = np.array(matrix)
    print(adjacency_matrix)
    path, cost = nearest_neighbor(adjacency_matrix, 0)
    print("Visited path:", path)
    print("Total cost:", cost)

    #part 1 of the assignment
    size_5_graphs = [] 
    size_10_graphs = []
    size_15_graphs = []
    size_20_graphs = []
    size_25_graphs = []
    size_30_graphs = []

    for i in range(30):
        size_5_graphs.append(make_graph(5))
        size_10_graphs.append(make_graph(10))
        size_15_graphs.append(make_graph(15))
        size_20_graphs.append(make_graph(20))
        size_25_graphs.append(make_graph(25))
        size_30_graphs.append(make_graph(30))
        

if __name__ == '__main__':
    main()