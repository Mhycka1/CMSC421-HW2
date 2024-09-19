import random
import numpy as np
import networkx as nx


# #method to generate graphs
def make_graph(node_amount):
    graph = nx.Graph()
    graph.add_nodes_from(range(node_amount))

    for i in range(node_amount):
        for j in range(i + 1, node_amount):
            distance = random.randint(1, 100)
            graph.add_edge(i, j, weight=distance)
    
    matrix = nx.to_numpy_array(graph)
    print(matrix)

    return matrix

def nearest_neighbor(adj_matrix):
    start = 1

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
