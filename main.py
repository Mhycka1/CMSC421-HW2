import sys
import numpy as np


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
    

if __name__ == '__main__':
    main()