
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from imp_data import load_maastricht_data
import random
from matplotlib.colors import to_rgba
from collections import defaultdict

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
    nearest_node_pos = distances.to_numpy().argmin()  # integer position of the nearest node
    nearest_node = nodes.iloc[nearest_node_pos]['node_id']
    sp_nodes[sp['sp_id']] = nearest_node

# Generate distinct colors for each service point
num_sps = len(service_points)
colors = plt.cm.tab20(np.linspace(0, 1, max(num_sps, 20)))
sp_colors = {sp_id: colors[i % len(colors)] for i, sp_id in enumerate(sp_nodes.keys())}

# Run multiple Dijkstra algorithms simultaneously
edge_colors = {}
node_distances = defaultdict(lambda: float('inf'))
node_source = {}

# Initialize with service point nodes
for sp_id, node_id in sp_nodes.items():
    node_distances[node_id] = 0
    node_source[node_id] = sp_id

# Priority queue
frontier = [(0, node_id, sp_id) for sp_id, node_id in sp_nodes.items()]
frontier.sort()  # Sort by distance

# Process nodes until all are colored
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
            
            # Color the edge
            edge_key = tuple(sorted([node, neighbor]))
            edge_colors[edge_key] = sp_id

# Create a plot with a clean, map-like appearance
plt.figure(figsize=(15, 12), facecolor='#f5f5f5')
ax = plt.gca()
ax.set_facecolor('#f5f5f5')

# Plot edges with colors based on nearest service point
for _, edge in edges.iterrows():
    # Get the coordinates of the nodes that this edge connects
    from_node = nodes[nodes['node_id'] == edge['from_node']]
    to_node = nodes[nodes['node_id'] == edge['to_node']]
    
    # Get edge color
    edge_key = tuple(sorted([edge['from_node'], edge['to_node']]))
    sp_id = edge_colors.get(edge_key, None)
    
    if sp_id in sp_colors:
        color = sp_colors[sp_id]
        alpha = 0.8
    else:
        color = '#505050'
        alpha = 0.3
    
    # Draw a line between the two nodes
    plt.plot([from_node['x_rd'].values[0], to_node['x_rd'].values[0]], 
             [from_node['y_rd'].values[0], to_node['y_rd'].values[0]],
             color=color, linewidth=0.9, solid_capstyle='round', alpha=alpha)

# Plot service points with larger markers
for _, sp in service_points.iterrows():
    color = sp_colors[sp['sp_id']]
    plt.scatter(sp['x_rd'], sp['y_rd'], s=80, color=color, 
                edgecolor='black', linewidth=0.5, zorder=10, 
                label=f"SP {sp['sp_id']}")

# Remove axis ticks for cleaner map appearance
plt.xticks([])
plt.yticks([])

# Add a simple border
plt.box(False)

# Add a title with styling
plt.title('Maastricht Street Network with Service Point Areas', fontsize=16, fontweight='bold', pad=20)

# Add a scale bar (approximate)
x_range = nodes['x_rd'].max() - nodes['x_rd'].min()
scale_bar_length = x_range * 0.1  # 10% of the x-range
x_start = nodes['x_rd'].min() + x_range * 0.05
y_start = nodes['y_rd'].min() + (nodes['y_rd'].max() - nodes['y_rd'].min()) * 0.05

plt.plot([x_start, x_start + scale_bar_length], [y_start, y_start], 'k-', linewidth=2)
plt.text(x_start + scale_bar_length/2, y_start - (nodes['y_rd'].max() - nodes['y_rd'].min()) * 0.01, 
         f'{int(scale_bar_length/1000)} km', ha='center', fontsize=9)

# Add a simple North arrow
arrow_x = x_start + x_range * 0.6
arrow_y = y_start
arrow_length = x_range * 0.03
plt.arrow(arrow_x, arrow_y, 0, arrow_length, head_width=arrow_length*0.4, 
          head_length=arrow_length*0.4, fc='black', ec='black')
plt.text(arrow_x, arrow_y + arrow_length*1.5, 'N', ha='center', fontsize=10, fontweight='bold')

# Create a custom legend with unique service points
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), 
           loc='lower right', frameon=True, framealpha=0.9)

# Show the plot with tight layout
plt.tight_layout()
plt.show()

