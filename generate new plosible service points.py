import pandas as pd
import numpy as np
import os
from scipy.spatial import distance_matrix # For efficient distance calculation
import matplotlib.pyplot as plt
from imp_data import load_maastricht_data # To load nodes, edges, and original SPs

# --- Configuration ---
output_directory = "data"
output_filename = "all_potential_service_points.csv"
full_output_path = os.path.join(output_directory, output_filename)

POTENTIAL_SPS_FILE = "data/all_potential_service_points.csv"
ORIGINAL_SP_COLOR = 'blue'
NEW_POTENTIAL_SP_COLOR = 'red'
NETWORK_EDGE_COLOR = '#AAAAAA' # Light grey for network edges

# Load existing data (assuming your imp_data module is accessible)
# If not, you might need to load 'service_points.csv' directly
try:
    import imp_data
    maastricht_data = imp_data.load_maastricht_data()
    current_sps_df = maastricht_data['service_points'].copy()
    print(f"Loaded {len(current_sps_df)} current service points.")
except Exception as e:
    print(f"Could not load data using imp_data: {e}")
    print("Attempting to load 'data/service_points.csv' directly as a fallback.")
    try:
        current_sps_df = pd.read_csv("data/service_points.csv")
        if not all(col in current_sps_df.columns for col in ['sp_id', 'x_rd', 'y_rd']):
            raise ValueError("Fallback service_points.csv is missing required columns.")
        print(f"Loaded {len(current_sps_df)} current service points from fallback CSV.")
    except Exception as e_fallback:
        print(f"Error loading fallback 'data/service_points.csv': {e_fallback}")
        print("Please ensure current service points data is available.")
        current_sps_df = pd.DataFrame(columns=['sp_id', 'name', 'x_rd', 'y_rd', 'capacity']) # Empty df

# Parameters for generating new points
num_grid_points_x = 20  # Grid density along X-axis
num_grid_points_y = 20  # Grid density along Y-axis
gap_threshold_distance = 2000  # Meters. Add new potential SP if > this distance from any current SP.
default_capacity_new_potential = 100 # Default capacity for newly generated potential SPs
default_capacity_current_sps = 100 # Default if 'capacity' column is missing in current SPs

# Approximate bounding box for Maastricht (RD New coordinates)
# Adjust these to your actual area if current_sps_df is empty or to define a broader search
if not current_sps_df.empty:
    min_x_rd_bounds = current_sps_df['x_rd'].min() - 2000 # Extend bounds a bit
    max_x_rd_bounds = current_sps_df['x_rd'].max() + 2000
    min_y_rd_bounds = current_sps_df['y_rd'].min() - 2000
    max_y_rd_bounds = current_sps_df['y_rd'].max() + 2000
else: # Fallback if no current SPs loaded
    min_x_rd_bounds, max_x_rd_bounds = 170000, 190000
    min_y_rd_bounds, max_y_rd_bounds = 300000, 320000
    print("Warning: No current SPs loaded, using default bounding box for generating grid.")


# --- 1. Prepare Current Service Points ---
if not current_sps_df.empty:
    # Ensure 'sp_id' is string
    current_sps_df['sp_id'] = current_sps_df['sp_id'].astype(str)
    # Add/ensure capacity column for current SPs
    if 'capacity' not in current_sps_df.columns:
        current_sps_df['capacity'] = default_capacity_current_sps
    else:
        current_sps_df['capacity'] = pd.to_numeric(current_sps_df['capacity'], errors='coerce').fillna(default_capacity_current_sps).astype(int)

    # Select relevant columns
    current_sps_for_potential_list = current_sps_df[['sp_id', 'x_rd', 'y_rd', 'capacity']].copy()
    if 'name' in current_sps_df.columns:
         current_sps_for_potential_list['name'] = current_sps_df['name']
    else:
         current_sps_for_potential_list['name'] = current_sps_for_potential_list['sp_id'] # Use sp_id as name if missing
else:
    current_sps_for_potential_list = pd.DataFrame(columns=['sp_id', 'name', 'x_rd', 'y_rd', 'capacity'])

# --- 2. Generate New Potential Points in Sparse Areas ---
new_potential_points = []

# Create grid cell centers
x_grid = np.linspace(min_x_rd_bounds, max_x_rd_bounds, num_grid_points_x)
y_grid = np.linspace(min_y_rd_bounds, max_y_rd_bounds, num_grid_points_y)
grid_coords = np.array([(x, y) for x in x_grid for y in y_grid])

if not current_sps_df.empty and len(grid_coords) > 0:
    current_sp_coords = current_sps_df[['x_rd', 'y_rd']].values
    
    # Calculate distance from each grid point to all current SPs
    # dist_matrix[i, j] is distance from grid_coords[i] to current_sp_coords[j]
    dist_matrix = distance_matrix(grid_coords, current_sp_coords)
    
    # For each grid point, find the minimum distance to any current SP
    min_dist_to_current_sp = dist_matrix.min(axis=1)
    
    # Identify grid points that are "far" from any current SP
    far_grid_indices = np.where(min_dist_to_current_sp > gap_threshold_distance)[0]
    
    sp_id_counter = len(current_sps_for_potential_list) # Start counter after current SPs

    for idx in far_grid_indices:
        gx, gy = grid_coords[idx]
        new_sp_id = f"new_potential_sp_{sp_id_counter:04d}"
        new_potential_points.append({
            'sp_id': new_sp_id,
            'name': f"Sparse Area Point {sp_id_counter:04d}",
            'x_rd': gx,
            'y_rd': gy,
            'capacity': default_capacity_new_potential
        })
        sp_id_counter += 1
    print(f"Generated {len(new_potential_points)} new potential SPs in sparse areas.")
elif len(grid_coords) > 0: # No current SPs, so all grid points are potential candidates (up to a limit)
    print("No current SPs to compare against. Adding grid points as new potential SPs.")
    sp_id_counter = 0
    # Limit the number if all grid points are added
    max_new_from_grid = num_grid_points_x * num_grid_points_y 
    for i in range(min(len(grid_coords), max_new_from_grid)):
        gx, gy = grid_coords[i]
        new_sp_id = f"new_potential_sp_{sp_id_counter:04d}"
        new_potential_points.append({
            'sp_id': new_sp_id,
            'name': f"Grid Point {sp_id_counter:04d}",
            'x_rd': gx,
            'y_rd': gy,
            'capacity': default_capacity_new_potential
        })
        sp_id_counter += 1
    print(f"Generated {len(new_potential_points)} new potential SPs from grid (no current SPs for reference).")


new_potential_sps_df = pd.DataFrame(new_potential_points)

# --- 3. Combine and Save ---
if not new_potential_sps_df.empty:
    all_potential_sps_df = pd.concat([current_sps_for_potential_list, new_potential_sps_df], ignore_index=True)
else:
    all_potential_sps_df = current_sps_for_potential_list.copy()

# Ensure unique sp_ids (should be if logic is correct, but as a safeguard)
all_potential_sps_df.drop_duplicates(subset=['sp_id'], keep='first', inplace=True)

# Ensure standard column order
final_columns = ['sp_id', 'name', 'x_rd', 'y_rd', 'capacity']
# Add any missing columns with default values if necessary before reordering
for col in final_columns:
    if col not in all_potential_sps_df.columns:
        if col == 'name': all_potential_sps_df[col] = all_potential_sps_df['sp_id']
        elif col == 'capacity': all_potential_sps_df[col] = default_capacity_new_potential
        else: all_potential_sps_df[col] = np.nan # Or some other default

all_potential_sps_df = all_potential_sps_df[final_columns]


# Create the directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

all_potential_sps_df.to_csv(full_output_path, index=False)

print(f"\nSuccessfully created/updated '{full_output_path}' with {len(all_potential_sps_df)} total potential service points.")
print(f"  ({len(current_sps_for_potential_list)} from current, {len(new_potential_sps_df)} newly generated)")

# Display the first few rows
print("\nFirst 5 rows of the generated file:")
print(all_potential_sps_df.head())
print("\nLast 5 rows of the generated file:")
print(all_potential_sps_df.tail())

# --- 4. Plotting ---
# --- 1. Load Data ---
print("Loading Maastricht base data (nodes, edges, original SPs)...")
try:
    maastricht_data = load_maastricht_data()
    nodes_df = maastricht_data["nodes"]
    edges_df = maastricht_data["edges"]
    original_sps_df = maastricht_data["service_points"].copy() # For identifying original SPs
    original_sp_ids = set(original_sps_df['sp_id'].astype(str))
    print(f"Loaded {len(nodes_df)} nodes, {len(edges_df)} edges, {len(original_sps_df)} original SPs.")
except Exception as e:
    print(f"Error loading Maastricht base data: {e}")
    # Create empty dataframes if loading fails, so the script can still try to plot potential SPs
    nodes_df = pd.DataFrame(columns=['node_id', 'x_rd', 'y_rd'])
    edges_df = pd.DataFrame(columns=['from_node', 'to_node'])
    original_sp_ids = set()

print(f"Loading potential service points from: {POTENTIAL_SPS_FILE}")
if not os.path.exists(POTENTIAL_SPS_FILE):
    print(f"Error: File '{POTENTIAL_SPS_FILE}' not found. Please generate it first.")
    exit()

try:
    all_potential_sps_df = pd.read_csv(POTENTIAL_SPS_FILE)
    all_potential_sps_df['sp_id'] = all_potential_sps_df['sp_id'].astype(str)
    print(f"Loaded {len(all_potential_sps_df)} total potential service points.")
except Exception as e:
    print(f"Error loading '{POTENTIAL_SPS_FILE}': {e}")
    exit()

# --- 2. Differentiate SP Types ---
# Method 1: Check against original_sp_ids loaded from imp_data
all_potential_sps_df['is_original'] = all_potential_sps_df['sp_id'].isin(original_sp_ids)

# Method 2: Fallback or alternative - check naming convention if original_sp_ids is empty
# This assumes new potential SPs might have a specific prefix if the above fails.
if not original_sp_ids: # If original_sps_df couldn't be loaded
    print("Warning: Could not identify original SPs from base data. Using naming convention for 'is_original'.")
    # Example: if original SPs don't start with "new_potential_sp_" or "potential_sp_"
    all_potential_sps_df['is_original'] = ~all_potential_sps_df['sp_id'].str.contains("new_potential_sp_|potential_sp_", case=False, na=False)


current_sps_to_plot = all_potential_sps_df[all_potential_sps_df['is_original']].copy()
newly_generated_sps_to_plot = all_potential_sps_df[~all_potential_sps_df['is_original']].copy()

print(f"Identified {len(current_sps_to_plot)} as current/original SPs for plotting.")
print(f"Identified {len(newly_generated_sps_to_plot)} as newly generated potential SPs for plotting.")

# --- 3. Create Plot ---
plt.figure(figsize=(18, 15), facecolor='#f5f5f5') # Adjusted size for potentially more points
ax = plt.gca()
ax.set_facecolor('#f5f5f5')

# Plot network edges (if nodes_df and edges_df loaded)
if not nodes_df.empty and not edges_df.empty:
    print("Plotting road network...")
    # Create a quick lookup for node positions
    node_pos = {node_id: (x, y) for node_id, x, y in zip(nodes_df['node_id'], nodes_df['x_rd'], nodes_df['y_rd'])}
    for _, edge in edges_df.iterrows():
        from_node_id = edge['from_node']
        to_node_id = edge['to_node']
        if from_node_id in node_pos and to_node_id in node_pos:
            x_coords = [node_pos[from_node_id][0], node_pos[to_node_id][0]]
            y_coords = [node_pos[from_node_id][1], node_pos[to_node_id][1]]
            plt.plot(x_coords, y_coords, color=NETWORK_EDGE_COLOR, linewidth=0.7, alpha=0.5, zorder=1)
else:
    print("Skipping road network plotting as node/edge data was not loaded.")


# Plot newly generated potential service points
if not newly_generated_sps_to_plot.empty:
    print(f"Plotting {len(newly_generated_sps_to_plot)} newly generated potential SPs...")
    plt.scatter(newly_generated_sps_to_plot['x_rd'], newly_generated_sps_to_plot['y_rd'],
                s=50,  # Slightly smaller or different shape
                color=NEW_POTENTIAL_SP_COLOR,
                edgecolor='black', linewidth=0.5, zorder=3,
                label='Newly Generated Potential SPs', marker='^') # Triangle marker

# Plot current/original service points (on top)
if not current_sps_to_plot.empty:
    print(f"Plotting {len(current_sps_to_plot)} current/original SPs...")
    plt.scatter(current_sps_to_plot['x_rd'], current_sps_to_plot['y_rd'],
                s=80, # Standard size
                color=ORIGINAL_SP_COLOR,
                edgecolor='black', linewidth=0.5, zorder=5,
                label='Current/Original SPs', marker='o') # Circle marker


# --- 4. Styling and Labels ---
plt.xticks([])
plt.yticks([])
plt.box(False)
plt.title('Map of Current and Potential Service Point Locations', fontsize=18, fontweight='bold', pad=20)

# Determine plot bounds for scale bar and North arrow
if not all_potential_sps_df.empty:
    min_x, max_x = all_potential_sps_df['x_rd'].min(), all_potential_sps_df['x_rd'].max()
    min_y, max_y = all_potential_sps_df['y_rd'].min(), all_potential_sps_df['y_rd'].max()
elif not nodes_df.empty : # Fallback to nodes if potential SPs are empty but network exists
    min_x, max_x = nodes_df['x_rd'].min(), nodes_df['x_rd'].max()
    min_y, max_y = nodes_df['y_rd'].min(), nodes_df['y_rd'].max()
else: # Default if no data at all
    min_x, max_x = 170000, 190000
    min_y, max_y = 300000, 320000

x_range_map = max_x - min_x
y_range_map = max_y - min_y

# Add a scale bar (approximate)
if x_range_map > 0 and y_range_map > 0:
    scale_bar_length_map = x_range_map * 0.1  # 10% of the x-range
    x_start_map = min_x + x_range_map * 0.05
    y_start_map = min_y + y_range_map * 0.05

    plt.plot([x_start_map, x_start_map + scale_bar_length_map], [y_start_map, y_start_map], 'k-', linewidth=2, zorder=10)
    plt.text(x_start_map + scale_bar_length_map / 2, y_start_map - y_range_map * 0.015,
             f'{int(scale_bar_length_map / 1000)} km', ha='center', fontsize=10, zorder=10)

    # Add a simple North arrow
    arrow_x_map = min_x + x_range_map * 0.90 # Position to the right
    arrow_y_map = min_y + y_range_map * 0.10
    arrow_length_map = y_range_map * 0.05
    if arrow_length_map > 0:
        plt.arrow(arrow_x_map, arrow_y_map, 0, arrow_length_map,
                  head_width=arrow_length_map * 0.4, head_length=arrow_length_map * 0.4,
                  fc='black', ec='black', zorder=10)
        plt.text(arrow_x_map, arrow_y_map + arrow_length_map * 1.5, 'N',
                 ha='center', fontsize=11, fontweight='bold', zorder=10)

# Create a custom legend
handles, labels = plt.gca().get_legend_handles_labels()
if handles: # Only show legend if there's something to label
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),
               loc='lower right', frameon=True, framealpha=0.95, fontsize=10)
else:
    print("No data points plotted, legend will be empty.")

plt.axis('equal') # Ensure aspect ratio is correct for map
plt.tight_layout()
print("Showing plot...")
plt.show()
