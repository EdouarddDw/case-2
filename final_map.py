import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from imp_data import load_maastricht_data
from collections import defaultdict
import os

# --- Configuration ---
OPTIMIZED_SP_CSV = 'data/best_service_points_solution.csv'
APL_CANDIDATES_CSV = 'top_20_apl_candidates.csv'
APL_PLOT_COLOR = 'purple' # Fixed color for APL markers

# --- 1. Load Data ---
print("Loading data...")
try:
    data = load_maastricht_data()
    nodes_df = data["nodes"].copy()
    edges_df = data["edges"].copy()
except Exception as e:
    print(f"Error loading base data from imp_data: {e}")
    exit()

try:
    optimized_sp_df = pd.read_csv(OPTIMIZED_SP_CSV)
    print(f"Loaded optimized SPs from {OPTIMIZED_SP_CSV}")
except FileNotFoundError:
    print(f"ERROR: Optimized SP file not found: {OPTIMIZED_SP_CSV}")
    exit()
except Exception as e:
    print(f"Error loading {OPTIMIZED_SP_CSV}: {e}")
    exit()

try:
    apl_candidates_df = pd.read_csv(APL_CANDIDATES_CSV)
    print(f"Loaded APL candidates from {APL_CANDIDATES_CSV}")
except FileNotFoundError:
    print(f"ERROR: APL candidates file not found: {APL_CANDIDATES_CSV}")
    exit()
except Exception as e:
    print(f"Error loading {APL_CANDIDATES_CSV}: {e}")
    exit()

# --- 2. Prepare Data for Plotting and Dijkstra ---
print("Preparing SPs and APLs...")
points_to_plot = [] # List to hold all SPs and APLs for plotting
sp_dijkstra_sources = [] # List to hold only SPs that will be Dijkstra sources

# Process Open Service Points
open_sps = optimized_sp_df[optimized_sp_df['opened'] == True].copy()
if 'sp_id_int' not in open_sps.columns:
    print(f"ERROR: 'sp_id_int' column missing in {OPTIMIZED_SP_CSV}")
    exit()

open_sps['node_id'] = open_sps['sp_id_int'].astype(int)
open_sps_coords = pd.merge(open_sps, nodes_df[['node_id', 'x_rd', 'y_rd']], on='node_id', how='left')

for _, sp_row in open_sps_coords.iterrows():
    coord_col_x = 'x_rd_y' if 'x_rd_y' in sp_row.index else 'x_rd'
    coord_col_y = 'y_rd_y' if 'y_rd_y' in sp_row.index else 'y_rd'

    if coord_col_x in sp_row.index and coord_col_y in sp_row.index and \
       pd.notna(sp_row[coord_col_x]) and pd.notna(sp_row[coord_col_y]):
        sp_info = {
            'id': f"SP_{sp_row['node_id']}",
            'node_id': int(sp_row['node_id']),
            'x': sp_row[coord_col_x],
            'y': sp_row[coord_col_y],
            'type': 'SP'
        }
        points_to_plot.append(sp_info)
        sp_dijkstra_sources.append(sp_info) # SPs are sources for Dijkstra
    else:
        sp_identifier_for_warning = sp_row.get('sp_id', sp_row.get('sp_id_int', 'Unknown SP'))
        node_id_for_warning = sp_row.get('node_id', 'N/A')
        if not (coord_col_x in sp_row.index and coord_col_y in sp_row.index):
            print(f"WARNING: Expected coordinate columns ('{coord_col_x}', '{coord_col_y}') missing for SP {sp_identifier_for_warning} (node_id: {node_id_for_warning}). Skipping.")
        else: 
            print(f"Warning: Coordinates are NaN for open SP {sp_identifier_for_warning} (node_id: {node_id_for_warning}). Skipping.")

# Process APLs (select top 5 by demand for plotting only)
DEMAND_COLUMN_APL = 'num_parcels_year'
top_5_apl_for_plotting_df = pd.DataFrame()

if DEMAND_COLUMN_APL in apl_candidates_df.columns:
    top_5_apl_for_plotting_df = apl_candidates_df.sort_values(by=DEMAND_COLUMN_APL, ascending=False).head(5)
    print(f"Selected top 5 APLs by demand ({DEMAND_COLUMN_APL}) for plotting. Demand values: {top_5_apl_for_plotting_df[DEMAND_COLUMN_APL].tolist()}")
else:
    print(f"Warning: Demand column '{DEMAND_COLUMN_APL}' not found in APL candidates. Not plotting APLs by demand.")
    # top_5_apl_for_plotting_df remains empty or you can set a different fallback

if not top_5_apl_for_plotting_df.empty:
    apl_coords_for_plotting = pd.merge(top_5_apl_for_plotting_df, nodes_df[['node_id', 'x_rd', 'y_rd']], on='node_id', how='left')
    for _, apl_row in apl_coords_for_plotting.iterrows():
        coord_col_x = 'x_rd_y' if 'x_rd_y' in apl_row.index else 'x_rd'
        coord_col_y = 'y_rd_y' if 'y_rd_y' in apl_row.index else 'y_rd'

        if coord_col_x in apl_row.index and coord_col_y in apl_row.index and \
           pd.notna(apl_row[coord_col_x]) and pd.notna(apl_row[coord_col_y]):
            points_to_plot.append({ # Add APLs for plotting
                'id': f"APL_{apl_row['node_id']}",
                'node_id': int(apl_row['node_id']),
                'x': apl_row[coord_col_x],
                'y': apl_row[coord_col_y],
                'type': 'APL'
            })
            # APLs are NOT added to sp_dijkstra_sources
        else:
            apl_identifier_for_warning = apl_row.get('id', apl_row.get('node_id', 'Unknown APL'))
            node_id_for_warning = apl_row.get('node_id', 'N/A')
            if not (coord_col_x in apl_row.index and coord_col_y in apl_row.index):
                print(f"WARNING: Expected coordinate columns ('{coord_col_x}', '{coord_col_y}') missing for APL {apl_identifier_for_warning} (node_id: {node_id_for_warning}). Skipping.")
            else: 
                print(f"Warning: Coordinates are NaN for APL {apl_identifier_for_warning} (node_id: {node_id_for_warning}). Skipping.")

if not sp_dijkstra_sources:
    print("No open Service Points found to define coverage zones. Road network will not be colored by SP zones.")
if not points_to_plot:
    print("No SPs or APLs found with valid coordinates for plotting. Exiting.")
    exit()

print(f"Total points for plotting (SPs & APLs): {len(points_to_plot)}")
print(f"Total SPs for Dijkstra zone coloring: {len(sp_dijkstra_sources)}")

# --- 3. Create NetworkX Graph ---
print("Creating graph...")
G = nx.Graph()
for _, node_row in nodes_df.iterrows():
    G.add_node(int(node_row['node_id']), pos=(node_row['x_rd'], node_row['y_rd']))
for _, edge_row in edges_df.iterrows():
    G.add_edge(int(edge_row['from_node']), int(edge_row['to_node']), weight=float(edge_row['length_m']))

# --- 4. Assign Colors to SP Coverage Zones ---
sp_zone_colors = {} # Colors for SP zones
if sp_dijkstra_sources:
    num_sp_sources = len(sp_dijkstra_sources)
    # Using tab20, but ensuring enough distinct colors if many SPs
    palette = plt.cm.get_cmap('tab20', max(num_sp_sources, 20)) 
    sp_zone_colors = {source['id']: palette(i % palette.N) for i, source in enumerate(sp_dijkstra_sources)}

# --- 5. Multi-Source Dijkstra for SP Zone Coverage ---
edge_sp_zone_map = {}  # To store which SP_id covers an edge
node_assigned_sp_id = {} # To store which SP_id covers a node

if sp_dijkstra_sources: # Only run Dijkstra if there are SP sources
    print("Running Dijkstra for SP zone coverage...")
    node_distances_from_sp = defaultdict(lambda: float('inf'))
    
    frontier = []
    for sp_source_item in sp_dijkstra_sources: # Initialize ONLY with SPs
        node_id = sp_source_item['node_id']
        sp_source_id_key = sp_source_item['id']
        if G.has_node(node_id):
            node_distances_from_sp[node_id] = 0
            node_assigned_sp_id[node_id] = sp_source_id_key
            frontier.append((0, node_id, sp_source_id_key)) 
        else:
            print(f"Warning: SP source node {node_id} (for {sp_source_id_key}) not in graph G. Skipping.")
    frontier.sort() 

    processed_nodes_count = 0
    while frontier:
        dist, current_node_id, current_sp_source_id = frontier.pop(0)
        processed_nodes_count +=1
        if processed_nodes_count % 2000 == 0: # Print less frequently
            print(f"  Dijkstra processed {processed_nodes_count} nodes for SP zones...")

        if dist > node_distances_from_sp[current_node_id] and node_assigned_sp_id[current_node_id] != current_sp_source_id :
            continue
        
        for neighbor_node_id in G.neighbors(current_node_id):
            edge_data = G.get_edge_data(current_node_id, neighbor_node_id)
            if edge_data is None or 'weight' not in edge_data: continue
            
            edge_weight = edge_data['weight']
            new_dist_to_neighbor = dist + edge_weight

            if new_dist_to_neighbor < node_distances_from_sp[neighbor_node_id]:
                node_distances_from_sp[neighbor_node_id] = new_dist_to_neighbor
                node_assigned_sp_id[neighbor_node_id] = current_sp_source_id
                frontier.append((new_dist_to_neighbor, neighbor_node_id, current_sp_source_id))
                frontier.sort() 

    for u, v in G.edges():
        edge_key = tuple(sorted((u,v)))
        source_u = node_assigned_sp_id.get(u)
        source_v = node_assigned_sp_id.get(v)
        if source_u:
            edge_sp_zone_map[edge_key] = source_u
        elif source_v:
            edge_sp_zone_map[edge_key] = source_v
    print("Dijkstra for SP zones finished.")
else:
    print("Skipping Dijkstra for SP zones as no SP sources are available.")


# --- 6. Plotting ---
print("Creating plot...")
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(18, 15))
ax.set_facecolor('#EAEAEA') 
ax.set_xticks([])
ax.set_yticks([])

# Plot Edges (colored by SP zones)
print("  Plotting road network edges...")
plotted_edge_count = 0
for _, edge_row in edges_df.iterrows():
    u, v = int(edge_row['from_node']), int(edge_row['to_node'])
    edge_key_sorted = tuple(sorted((u, v)))
    
    assigned_sp_id = edge_sp_zone_map.get(edge_key_sorted) # Will be an SP ID or None
    
    color = sp_zone_colors.get(assigned_sp_id, '#CCCCCC') # Color by SP zone, default grey
    alpha = 0.7 if assigned_sp_id else 0.2
    linewidth = 1.0 if assigned_sp_id else 0.5
    zorder = 2 if assigned_sp_id else 1

    node_u_data = nodes_df[nodes_df['node_id'] == u]
    node_v_data = nodes_df[nodes_df['node_id'] == v]

    if not node_u_data.empty and not node_v_data.empty:
        ax.plot([node_u_data['x_rd'].iloc[0], node_v_data['x_rd'].iloc[0]],
                [node_u_data['y_rd'].iloc[0], node_v_data['y_rd'].iloc[0]],
                color=color, linewidth=linewidth, alpha=alpha, zorder=zorder, solid_capstyle='round')
        plotted_edge_count +=1
print(f"    ...{plotted_edge_count} edges plotted.")

# Plot SPs and APLs
print("  Plotting Open SPs and APLs...")
marker_map = {'SP': 'P', 'APL': '*'} # SP: Plus, APL: Star
size_map = {'SP': 180, 'APL': 280} # APLs slightly larger
plot_labels_added = set()

for item_to_plot in points_to_plot:
    item_id = item_to_plot['id']
    item_type = item_to_plot['type']
    
    current_item_color = ''
    legend_label_key = ''

    if item_type == 'SP':
        current_item_color = sp_zone_colors.get(item_id, 'black') # SPs use their zone color
        legend_label_key = "Service Point (Zone Color)"
    elif item_type == 'APL':
        current_item_color = APL_PLOT_COLOR # APLs use a fixed distinct color
        legend_label_key = "Top 5 APL (High Demand)"
    
    ax.scatter(item_to_plot['x'], item_to_plot['y'],
               s=size_map[item_type],
               color=current_item_color,
               marker=marker_map[item_type],
               edgecolors='black', linewidth=0.8,
               zorder=10,
               label=legend_label_key if legend_label_key not in plot_labels_added else "")
    if legend_label_key not in plot_labels_added:
         plot_labels_added.add(legend_label_key)


# --- Map Embellishments ---
ax.set_title('Maastricht Network: SP Coverage Zones & Top APLs', fontsize=18, fontweight='bold', pad=20)
ax.set_aspect('equal', adjustable='box')

# Consolidate legend from plot artists
handles, labels = ax.get_legend_handles_labels()

# Create custom legend elements for marker types if not covered by scatter labels
custom_legend_elements = []
if "Service Point (Zone Color)" not in labels: # If no SPs plotted but we want to show marker
    custom_legend_elements.append(plt.Line2D([0], [0], marker='P', color='w', label='Service Point', markersize=10, markerfacecolor='gray', markeredgecolor='black'))
if "Top 5 APL (High Demand)" not in labels: # If no APLs plotted but we want to show marker
    custom_legend_elements.append(plt.Line2D([0], [0], marker='*', color='w', label='APL', markersize=12, markerfacecolor=APL_PLOT_COLOR, markeredgecolor='black'))

for el in custom_legend_elements:
    if el.get_label() not in labels:
        handles.append(el)
        labels.append(el.get_label())

# Remove duplicate labels from legend
by_label = dict(zip(labels, handles))
if by_label:
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=9, frameon=True, facecolor='white', framealpha=0.85)


# Approximate Scale Bar
x_min, x_max = nodes_df['x_rd'].min(), nodes_df['x_rd'].max()
y_min, y_max = nodes_df['y_rd'].min(), nodes_df['y_rd'].max()
map_width_m = x_max - x_min
scale_bar_length_m = round(map_width_m * 0.1 / 1000) * 1000 
if scale_bar_length_m == 0: scale_bar_length_m = 1000 

sb_x = x_min + (x_max - x_min) * 0.05
sb_y = y_min + (y_max - y_min) * 0.03
ax.plot([sb_x, sb_x + scale_bar_length_m], [sb_y, sb_y], color='black', linewidth=2.5, zorder=20)
ax.text(sb_x + scale_bar_length_m / 2, sb_y - (y_max-y_min)*0.015, f"{int(scale_bar_length_m/1000)} km",
        ha='center', va='top', fontsize=9, color='black', zorder=20)

# North Arrow (using the simpler text annotation from your visible code)
na_x = x_min + (x_max - x_min) * 0.95
na_y = y_min + (y_max - y_min) * 0.95 # Positioned at top right
ax.annotate('N', xy=(na_x, na_y), fontsize=12, ha='center', va='center', color='black', zorder=21,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white', alpha=0.8))

# Save the figure
output_file = "maastricht_network_coverage.png" # Using filename from your visible code
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as {output_file}")

plt.show()
print("Script finished.")