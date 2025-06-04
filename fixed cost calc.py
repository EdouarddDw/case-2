import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from imp_data import load_maastricht_data
import random
from matplotlib.colors import to_rgba
from collections import defaultdict
import json
import pandas as pd

# Load the data
data = load_maastricht_data()

# Extract nodes, edges, service points and CBS data
nodes = data["nodes"]
edges = data["edges"]
service_points = data["service_points"]
cbs = data["cbs"]  # This should contain housing data including home values

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
    # Handle all-NaN distances
    if distances.isna().all():
        print(f"Warning: All distances are NaN for service point {sp['sp_id']}. Skipping.")
        continue
    nearest_node_idx = distances.idxmin()
    # If idxmin returns NaN, skip
    if pd.isna(nearest_node_idx):
        print(f"Warning: Could not determine nearest node index for service point {sp['sp_id']}. Skipping.")
        continue
    nearest_node = nodes.loc[nearest_node_idx]['node_id']
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
    # Explicitly convert NumPy int32 to Python int
    json_data = {
        "service_point_nodes": {str(sp): [int(node) for node in nodes] for sp, nodes in service_point_nodes.items()},
        "node_service_point": {str(int(node)): str(sp) for node, sp in node_source.items()}
    }
    json.dump(json_data, f)

print("\nResults saved to service_point_coverage.json")

# Load the service point coverage data that we generated previously
with open('service_point_coverage.json', 'r') as f:
    coverage_data = json.load(f)

# Base cost for a service point per year
BASE_COST = 50000  # €50,000 per year

# First, let's check what columns are actually available in the CBS data
print("\nAvailable columns in CBS data:")
for col in cbs.columns:
    print(f"- {col}")

# Look for any columns that might contain home value information
home_value_columns = [col for col in cbs.columns if 'home' in col.lower() or 'house' in col.lower() or 'value' in col.lower() or 'price' in col.lower()]
print("\nPossible home value columns:", home_value_columns)

# Choose the appropriate column or use a placeholder if nothing is found
if home_value_columns:
    home_value_column = home_value_columns[0]  # Use the first match
    print(f"Using column '{home_value_column}' for home values")
    avg_home_value_overall = cbs[home_value_column].mean()
    print(f"Average home value across all squares: {avg_home_value_overall:.2f}")
else:
    print("No home value columns found. Using placeholder values.")
    home_value_column = None
    avg_home_value_overall = 1.0  # Neutral multiplier as fallback

# Calculate cost multiplier for each service point based on home values
sp_yearly_costs = {}

for _, sp_info in service_points.iterrows():
    sp_id = sp_info['sp_id']
    # Get the CBS square for this service point
    # First find the nearest node to this service point
    distances = np.sqrt((nodes['x_rd'] - sp_info['x_rd'])**2 + (nodes['y_rd'] - sp_info['y_rd'])**2)
    # Handle all-NaN distances
    if distances.isna().all():
        print(f"Warning: All distances are NaN for service point {sp_id}. Using default home value.")
        cbs_square = None
    else:
        nearest_node_idx = distances.idxmin()
        if pd.isna(nearest_node_idx):
            print(f"Warning: Could not find nearest node for service point {sp_id}. Using default home value.")
            cbs_square = None
        else:
            nearest_node = nodes.loc[nearest_node_idx]
            cbs_square = nearest_node['cbs_square']
    
    # Find the home value for this square
    square_data = cbs[cbs['cbs_square'] == cbs_square] if cbs_square is not None else pd.DataFrame()
    # source that proofs the multiplier idea:
    # https://walterliving.com/nl/en/city/maastricht
    if not square_data.empty and home_value_column is not None:
        if home_value_column in square_data.columns:
            avg_home_value = square_data[home_value_column].values[0]
            # Cost multiplier based on home value relative to overall average
            cost_multiplier = avg_home_value / avg_home_value_overall
        else:
            # Default to neutral multiplier if column doesn't exist
            avg_home_value = avg_home_value_overall
            cost_multiplier = 1.0
    else:
        # Default to neutral multiplier if data is missing
        avg_home_value = avg_home_value_overall
        cost_multiplier = 1.0
        print(f"Warning: No home value data for service point {sp_id} (CBS square {cbs_square})")
    
    # Calculate yearly cost with the multiplier
    yearly_cost = BASE_COST * cost_multiplier
    
    # Store the result
    sp_yearly_costs[sp_id] = {
        'cbs_square': cbs_square,
        'avg_home_value': avg_home_value,
        'cost_multiplier': cost_multiplier,
        'yearly_cost_€': yearly_cost,
        'nodes_covered': len(coverage_data['service_point_nodes'].get(str(sp_id), [])),  # Convert sp_id to string
    }

# Calculate total yearly cost
total_yearly_cost = sum(sp['yearly_cost_€'] for sp in sp_yearly_costs.values())

# Print results
print("\nYearly Cost Analysis for Service Points:")
print("=" * 80)
print(f"{'Service Point':<15}{'CBS Square':<15}{'Home Value (k€)':<18}{'Multiplier':<12}{'Yearly Cost (€)':<15}{'Nodes Covered':<15}")
print("-" * 80)

for sp_id, sp_cost in sp_yearly_costs.items():
    # Handle possible None for CBS square
    cbs_sq = sp_cost['cbs_square'] if sp_cost['cbs_square'] is not None else 'N/A'
    # Handle possible missing avg_home_value
    avg_home = sp_cost.get('avg_home_value', None)
    if avg_home is not None:
        avg_home_str = f"{avg_home:<18.2f}"
    else:
        avg_home_str = f"{'N/A':<18}"
    print(f"{sp_id:<15}{cbs_sq:<15}{avg_home_str}{sp_cost['cost_multiplier']:<12.2f}{sp_cost['yearly_cost_€']:<15.2f}{sp_cost['nodes_covered']:<15}")

print("=" * 80)
print(f"Total yearly cost for all service points: €{total_yearly_cost:,.2f}")

# Calculate cost efficiency metrics
print("\nCost Efficiency Analysis:")
print("=" * 80)
print(f"{'Service Point':<15}{'Yearly Cost (€)':<15}{'Nodes Covered':<15}{'Cost per Node (€)':<20}")
print("-" * 80)

for sp_id, sp_cost in sp_yearly_costs.items():
    nodes_covered = sp_cost['nodes_covered']
    cost_per_node = sp_cost['yearly_cost_€'] / nodes_covered if nodes_covered > 0 else float('inf')
    print(f"{sp_id:<15}{sp_cost['yearly_cost_€']:<15.2f}{nodes_covered:<15}{cost_per_node:<20.2f}")

# Export the cost data to a CSV file
cost_df = pd.DataFrame.from_dict(sp_yearly_costs, orient='index')
cost_df.index.name = 'service_point_id'
cost_df.to_csv('service_point_yearly_costs.csv')
print("\nResults saved to service_point_yearly_costs.csv")

# Create a visualization of costs vs. total pickups
plt.figure(figsize=(10, 6))
# Extract total pickups for each service point
total_pickups = [
    service_points.loc[service_points['sp_id'] == sp_id, 'total_pickups'].values[0]
    for sp_id in sp_yearly_costs.keys()
]
# Yearly costs
yearly_costs = [data['yearly_cost_€'] for data in sp_yearly_costs.values()]
labels = list(sp_yearly_costs.keys())

plt.scatter(total_pickups, yearly_costs, s=100, alpha=0.7)

# Add labels to each point
for i, label in enumerate(labels):
    plt.annotate(label, (total_pickups[i], yearly_costs[i]), textcoords="offset points",
                 xytext=(0, 10), ha='center')

plt.xlabel('Total Pickups')
plt.ylabel('Yearly Cost (€)')
plt.title('Service Point Yearly Cost vs. Total Pickups')
plt.grid(True, alpha=0.3)
# Set x-axis maximum to 50,000
plt.xlim(0, 55000)
plt.tight_layout()
plt.savefig('service_point_cost_vs_pickups.png', dpi=300)
plt.show()