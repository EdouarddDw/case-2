import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from imp_data import load_maastricht_data
import random
from matplotlib.colors import to_rgba
from collections import defaultdict
import json

# Load the data
data = load_maastricht_data()

# Extract nodes, edges and service points
nodes = data["nodes"]
edges = data["edges"]
service_points = data["service_points"]

# Create a networkx graph
G = nx.Graph()

# Add nodes to the graph
for _, node in nodes.iterrows():
    G.add_node(node['node_id'], pos=(node['x_rd'], node['y_rd']))

# Add edges to the graph
for _, edge in edges.iterrows():
    G.add_edge(edge['from_node'], edge['to_node'], weight=edge['length_m'])

# Find nearest network node for each service point
sp_nodes = {}
for _, sp in service_points.iterrows():
    # Calculate distance to all nodes
    distances = np.sqrt((nodes['x_rd'] - sp['x_rd'])**2 + (nodes['y_rd'] - sp['y_rd'])**2)
    nearest_node_idx = distances.idxmin()
    nearest_node = nodes.iloc[nearest_node_idx]['node_id']
    sp_nodes[sp['sp_id']] = nearest_node

# Run multiple Dijkstra algorithms simultaneously
node_distances = defaultdict(lambda: float('inf'))
node_source = {}

# Initialize with service point nodes
for sp_id, node_id in sp_nodes.items():
    node_distances[node_id] = 0
    node_source[node_id] = sp_id

# Priority queue
frontier = [(0, node_id, sp_id) for sp_id, node_id in sp_nodes.items()]
frontier.sort()  # Sort by distance

# Process nodes until all are assigned
while frontier:
    dist, node, sp_id = frontier.pop(0)
    
    # Skip if we've found a shorter path
    if dist > node_distances[node]:
        continue
    
    # Process neighbors
    for neighbor in G.neighbors(node):
        edge_weight = G[node][neighbor]['weight']
        new_dist = dist + edge_weight
        
        # If we found a shorter path
        if new_dist < node_distances[neighbor]:
            node_distances[neighbor] = new_dist
            node_source[neighbor] = sp_id
            frontier.append((new_dist, neighbor, sp_id))
            frontier.sort()  # Keep the queue sorted

# Create dictionary of service points and their related nodes
service_point_nodes = defaultdict(list)

# Assign each node to its closest service point
for node, sp_id in node_source.items():
    service_point_nodes[sp_id].append(node)

# Print results
print("Service Points and their associated nodes:")
for sp_id, nodes_list in service_point_nodes.items():
    print(f"\nService Point {sp_id} has {len(nodes_list)} associated nodes:")
    print(f"First few nodes: {nodes_list[:5]} {'...' if len(nodes_list) > 5 else ''}")
    
    # Get the distances for these nodes
    distances = [node_distances[node] for node in nodes_list]
    max_distance = max(distances)
    avg_distance = sum(distances) / len(distances)
    
    print(f"Maximum network distance: {max_distance:.2f} meters")
    print(f"Average network distance: {avg_distance:.2f} meters")

# Return the dictionary for further use
service_point_assignment = {
    "service_point_nodes": dict(service_point_nodes),
    "node_distances": dict(node_distances),
    "node_service_point": dict(node_source)
}

# Export to file if needed
with open('service_point_coverage.json', 'w') as f:
    # Convert values to lists where needed to make JSON serializable
    # Also convert numpy.int32 to native Python int
    json_data = {
        "service_point_nodes": {sp: [int(node) for node in nodes] for sp, nodes in service_point_nodes.items()},
        "node_service_point": {str(int(node)): sp for node, sp in node_source.items()}
    }
    json.dump(json_data, f)


print("\nResults saved to service_point_coverage.json")

# Also dump mapping of each service point to its nearest network node
with open('sp_to_network_node.json', 'w') as f:
    # Keys as strings for JSON compatibility, values as plain ints
    json.dump({str(int(sp)): int(node) for sp, node in sp_nodes.items()}, f)

print("Results saved to sp_to_network_node.json")


