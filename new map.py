#new map
import matplotlib.pyplot as plt
import geopandas as gpd
from imp_data import load_maastricht_data
import numpy as np

# Load the data
data = load_maastricht_data()

# Extract nodes and edges
nodes = data["nodes"]
edges = data["edges"]

# Create a plot with a clean, map-like appearance
plt.figure(figsize=(12, 10), facecolor='#f5f5f5')
ax = plt.gca()
ax.set_facecolor('#f5f5f5')

# Plot only the edges (roads) with better styling
for _, edge in edges.iterrows():
    # Get the coordinates of the nodes that this edge connects
    from_node = nodes[nodes['node_id'] == edge['from_node']]
    to_node = nodes[nodes['node_id'] == edge['to_node']]
    
    # Draw a line between the two nodes
    plt.plot([from_node['x_rd'].values[0], to_node['x_rd'].values[0]], 
             [from_node['y_rd'].values[0], to_node['y_rd'].values[0]],
             color='#505050', linewidth=0.7, solid_capstyle='round')

# Remove axis ticks for cleaner map appearance
plt.xticks([])
plt.yticks([])

# Add a simple border
plt.box(False)

# Add a title with styling
plt.title('Maastricht Street Network', fontsize=16, fontweight='bold', pad=20)

# Add a subtle grid for geographic context
plt.grid(False)

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

# Show the plot with tight layout
plt.tight_layout()
plt.show()

