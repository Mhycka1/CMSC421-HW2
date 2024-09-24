# here is where all the parts will be run
from part1 import make_graph, process_graph_family, make_part1_graphs, read_stats
from part2 import run_A_star, run_nn, run_nn2o, run_rnn, compute_differences, make_part2_graphs
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

def main():
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


    # size_5_stats = process_graph_family(size_5_graphs, 'Size_5')
    # size_10_stats = process_graph_family(size_10_graphs, 'Size_10')
    # size_15_stats = process_graph_family(size_15_graphs, 'Size_15')
    # size_20_stats = process_graph_family(size_20_graphs, 'Size_20')
    # size_25_stats = process_graph_family(size_25_graphs, 'Size_25')
    # size_30_stats = process_graph_family(size_30_graphs, 'Size_30')

    # make_part1_graphs()

    #Part 2 of HW
    print('Starting Part 2')
    size_5_graphs = [] 
    size_6_graphs = []
    size_7_graphs = []
    size_8_graphs = []
    size_9_graphs = []
    size_10_graphs = []

    for i in range(30):
        size_5_graphs.append(make_graph(5))
        size_6_graphs.append(make_graph(6))
        size_7_graphs.append(make_graph(7))
        size_8_graphs.append(make_graph(8))
        size_9_graphs.append(make_graph(9))
        size_10_graphs.append(make_graph(10))

    a_5 = run_A_star(size_5_graphs)
    a_6 = run_A_star(size_6_graphs)
    a_7 = run_A_star(size_7_graphs)
    a_8 = run_A_star(size_8_graphs)
    a_9 = run_A_star(size_9_graphs)
    a_10 = run_A_star(size_10_graphs)

    nn_5 = run_nn(size_5_graphs)
    nn_6 = run_nn(size_6_graphs)
    nn_7 = run_nn(size_7_graphs)
    nn_8 = run_nn(size_8_graphs)
    nn_9 = run_nn(size_9_graphs)
    nn_10 = run_nn(size_10_graphs)

    nn2o_5 = run_nn2o(size_5_graphs)
    nn2o_6 = run_nn2o(size_6_graphs)
    nn2o_7 = run_nn2o(size_7_graphs)
    nn2o_8 = run_nn2o(size_8_graphs)
    nn2o_9 = run_nn2o(size_9_graphs)
    nn2o_10 = run_nn2o(size_10_graphs)

    rnn_5 = run_rnn(size_5_graphs)
    rnn_6 = run_rnn(size_6_graphs)
    rnn_7 = run_rnn(size_7_graphs)
    rnn_8 = run_rnn(size_8_graphs)
    rnn_9 = run_rnn(size_9_graphs)
    rnn_10 = run_rnn(size_10_graphs)

    differences = {}

    # Calculate differences for each size
    for size, a_results in zip(['5', '6', '7', '8', '9', '10'], [a_5, a_6, a_7, a_8, a_9, a_10]):
        differences[size] = {
            'nn': compute_differences(a_results, eval(f'nn_{size}')),
            'nn2o': compute_differences(a_results, eval(f'nn2o_{size}')),
            'rnn': compute_differences(a_results, eval(f'rnn_{size}')),
        }

    # Store the differences in a structured format
    stored_differences = {}

    for size, diff in differences.items():
        stored_differences[size] = {}
        for algorithm, (cost_diff, expanded_diff) in diff.items():
            # Convert numpy float64 to regular float and store in the dictionary
            stored_differences[size][algorithm] = {
                'cost_diff': [float(c) for c in cost_diff],
                'expanded_diff': [float(e) for e in expanded_diff]
            }

    statistics = {}

    # Calculate statistics for each size and algorithm
    for size, algorithms in stored_differences.items():
        statistics[size] = {}
        for algorithm, diffs in algorithms.items():
            cost_diff = diffs['cost_diff']
            expanded_diff = diffs['expanded_diff']
            
            # Compute average, min, and max for cost differences
            statistics[size][algorithm] = {
                'cost_stats': {
                    'avg': float(np.mean(cost_diff)),
                    'min': float(np.min(cost_diff)),
                    'max': float(np.max(cost_diff)),
                },
                'expanded_stats': {
                    'avg': float(np.mean(expanded_diff)),
                    'min': float(np.min(expanded_diff)),
                    'max': float(np.max(expanded_diff)),
                },
            }

    make_part2_graphs(statistics)

    #part 3
    a_5 = run_A_star(size_5_graphs)
    a_6 = run_A_star(size_6_graphs)
    a_7 = run_A_star(size_7_graphs)
    a_8 = run_A_star(size_8_graphs)
    a_9 = run_A_star(size_9_graphs)
    a_10 = run_A_star(size_10_graphs)
        
    

    

if __name__ == "__main__":
    main()