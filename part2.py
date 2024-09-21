import heapq
import numpy as np
import networkx as nx 

# help adapted from chatgpt prompt
def mst_heuristic(adj_matrix, unvisited):
    """Calculate the Minimum Spanning Tree (MST) cost of the unvisited cities."""
    # Create a subgraph of the unvisited cities
    subgraph = adj_matrix[np.ix_(unvisited, unvisited)]
    graph = nx.Graph()
    
    # Add edges and weights to the graph
    for i in range(len(unvisited)):
        for j in range(i + 1, len(unvisited)):
            if subgraph[i, j] > 0:
                graph.add_edge(unvisited[i], unvisited[j], weight=subgraph[i, j])
    
    # Calculate the MST 
    mst = nx.minimum_spanning_tree(graph)
    
    # Return the sum of the MST weights
    return mst.size(weight='weight')

# A* with MST adapted from chatgpt code
def A_MST(adj_matrix):
    N = adj_matrix.shape[0]
    start_city = 0
    
    # Priority queue with elements as (cost, current_city, visited_cities, path)
    pq = []
    initial_state = (0, start_city, [start_city], [start_city])
    heapq.heappush(pq, (0, initial_state))
    
    while pq:
        # Pop the state with the smallest f(n) = g(n) + h(n)
        current_cost, (g_n, current_city, visited, path) = heapq.heappop(pq)
        
        # If all cities are visited and we are back at the start, goal state is achieved
        if len(visited) == N and path[-1] == start_city:
            return path, current_cost
        
        # Expand successors (visit next city)
        for next_city in range(N):
            if next_city not in visited:
                # Calculate g(n) (current path cost)
                new_g_n = g_n + adj_matrix[current_city][next_city]
                
                # h(n): Calculate the MST heuristic for the remaining unvisited cities
                unvisited = [city for city in range(N) if city not in visited and city != next_city]
                mst_cost = mst_heuristic(adj_matrix, unvisited) if unvisited else 0
                
                # Calculate f(n) = g(n) + h(n)
                f_n = new_g_n + mst_cost
                
                # Create new state with the next city visited
                new_state = (new_g_n, next_city, visited + [next_city], path + [next_city])
                
                # Push the new state into the priority queue
                heapq.heappush(pq, (f_n, new_state))
    
    return None, float('inf')