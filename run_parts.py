# here is where all the parts will be run
from part1 import make_graph, process_graph_family, make_part1_graphs, read_stats
import os
import matplotlib.pyplot as plt

def main():
    size_5_graphs = [] 
    size_10_graphs = []
    size_15_graphs = []
    size_20_graphs = []
    size_25_graphs = []
    size_30_graphs = []

    for i in range(30):
        print('happening')
        size_5_graphs.append(make_graph(5))
        size_10_graphs.append(make_graph(10))
        size_15_graphs.append(make_graph(15))
        size_20_graphs.append(make_graph(20))
        size_25_graphs.append(make_graph(25))
        size_30_graphs.append(make_graph(30))


    size_5_stats = process_graph_family(size_5_graphs, 'Size_5')
    size_10_stats = process_graph_family(size_10_graphs, 'Size_10')
    size_15_stats = process_graph_family(size_15_graphs, 'Size_15')
    size_20_stats = process_graph_family(size_20_graphs, 'Size_20')
    size_25_stats = process_graph_family(size_25_graphs, 'Size_25')
    size_30_stats = process_graph_family(size_30_graphs, 'Size_30')

    make_part1_graphs()