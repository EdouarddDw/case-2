import pandas as pd
import numpy as np
import random
import json
import os
import shutil # For managing temporary directories if not fully refactored
import imp_data # Assuming this is your data loading module
from collections import defaultdict
import networkx as nx # Make sure networkx is imported
from sklearn.preprocessing import MinMaxScaler # Added for socioeconomic scores

data = imp_data.load_maastricht_data() # Load initial data

# --- Configuration ---
ALL_POTENTIAL_SP_LOCATIONS_FILE = "data/all_potential_service_points.csv" # Needs sp_id, x_rd, y_rd etc.


# Heuristic Parameters
MAX_ITERATIONS = 1000  # Total iterations for the heuristic
NEIGHBORHOOD_SIZE = 10 # How many neighbors to check in each iteration

# --- New Feature Parameters ---

SP_DAILY_CAPACITY = 500  # Uniform daily capacity per service point
HOME_DELIVERY_COST_PER_KM = 1.50 # Cost per km for home delivery (matches existing COST_PER_KM_EURO_DELIVERY)
GLOBAL_BOUNCE_RATE_THRESHOLD = 0.01 # Max 1% global bounce rate
LOCAL_BOUNCE_RATE_THRESHOLD = 0.02  # Max 2% local bounce rate per SP
PENALTY_GLOBAL_BOUNCE_BREACH = 1000000  # Large penalty if global bounce rate KPI is missed
PENALTY_LOCAL_BOUNCE_BREACH_PER_SP = 100000 # Large penalty if local bounce rate KPI is missed by an SP

# --- Capacity bounds & defaults ---
DEFAULT_SP_CAPACITY = SP_DAILY_CAPACITY          # keep old name for backward compat
MIN_SP_CAPACITY = 20
MAX_SP_CAPACITY = 1000
CAPACITY_STEP = 1   # granularity when mutating capacities



def calculate_service_point_coverage(current_sps_df, nodes_df, edges_df):
    # Create a networkx graph
    G = nx.Graph()

    # Add nodes to the graph
    for _, node in nodes_df.iterrows():
        G.add_node(int(node['node_id']), pos=(node['x_rd'], node['y_rd'])) # Ensure node_id is int

    # Add edges to the graph
    for _, edge in edges_df.iterrows():
        G.add_edge(int(edge['from_node']), int(edge['to_node']), weight=float(edge['length_m'])) # Ensure int and float

    # Find nearest network node for each service point
    sp_nodes = {}
    if 'sp_id' in current_sps_df.columns and 'x_rd' in current_sps_df.columns and 'y_rd' in current_sps_df.columns:
        for _, sp in current_sps_df.iterrows():
            # Calculate distance to all nodes
            distances = np.sqrt((nodes_df['x_rd'] - sp['x_rd'])**2 + (nodes_df['y_rd'] - sp['y_rd'])**2)
            if not distances.empty:
                nearest_node_idx = distances.idxmin()
                nearest_node = nodes_df.iloc[nearest_node_idx]['node_id']
                sp_nodes[str(sp['sp_id'])] = int(nearest_node) # Ensure sp_id is str, node_id is int
            else:
                print(f"Warning: Could not find nearest node for SP {sp['sp_id']} due to empty nodes_df or distances.")
    else:
        print("Warning: current_sps_df is missing required columns (sp_id, x_rd, y_rd) for SP node mapping.")


    # Run multiple Dijkstra algorithms simultaneously
    node_distances = defaultdict(lambda: float('inf'))
    node_source = {}

    # Initialize with service point nodes
    for sp_id, node_id in sp_nodes.items(): # sp_id is already str here
        node_distances[node_id] = 0 # node_id is int
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
        if G.has_node(node):
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
        service_point_nodes[sp_id].append(node) # sp_id is str, node is int

    return {
        "service_point_nodes": {k: [int(n) for n in v] for k, v in service_point_nodes.items()}, # Ensure nodes are int
        "node_distances": {int(k): v for k, v in node_distances.items()}, # Ensure node_id is int
        "node_service_point": {int(k): str(v) for k, v in node_source.items()}, # Ensure node_id is int, sp_id is str
        "sp_to_network_node": {str(k): int(v) for k, v in sp_nodes.items()}, # Ensure sp_id is str, node_id is int
        "graph": G  # Return the created graph
    }


def calculate_fixed_costs(current_sps_df, cbs_df, nodes_df, coverage_data):
    """
    Refactored from fixed_cost_calc.py
    Input: DataFrame of active SPs, CBS data, nodes data, coverage_data (from previous step)
    Output: DataFrame with 'sp_id' and 'fixed_yearly_cost'
    """
    print(f"    Calculating fixed costs for {len(current_sps_df)} SPs...")
    if current_sps_df.empty or 'sp_id' not in current_sps_df.columns:
        return pd.DataFrame(columns=['sp_id', 'fixed_yearly_cost'])
    if cbs_df.empty or nodes_df.empty:
        print("    Warning: CBS data or Nodes data is empty. Cannot calculate location-based fixed costs accurately. Returning base costs.")
        fixed_costs = []
        BASE_COST = 50000  # Default base cost
        for _, sp_row in current_sps_df.iterrows():
            fixed_costs.append({'sp_id': sp_row['sp_id'], 'fixed_yearly_cost': BASE_COST})
        return pd.DataFrame(fixed_costs)

    BASE_COST = 50000  # €50,000 per year

    # Identify home value column in CBS data
    home_value_column = None
    possible_home_value_cols = [col for col in cbs_df.columns if 'home' in col.lower() or 'house' in col.lower() or 'value' in col.lower() or 'price' in col.lower() or 'gm_woz' in col.lower()]
    if possible_home_value_cols:
        home_value_column = possible_home_value_cols[0] # Take the first likely candidate
        print(f"    Using CBS column '{home_value_column}' for home values.")
        # Ensure the column is numeric and handle non-numeric gracefully
        if not pd.api.types.is_numeric_dtype(cbs_df[home_value_column]):
            print(f"    Warning: CBS column '{home_value_column}' is not numeric. Attempting to convert.")
            try:
                cbs_df[home_value_column] = pd.to_numeric(cbs_df[home_value_column], errors='coerce')
            except Exception as e:
                print(f"    Error converting '{home_value_column}' to numeric: {e}. Using neutral multiplier.")
                home_value_column = None # Fallback
        
        if home_value_column and cbs_df[home_value_column].isnull().all(): # If all values are NaN after conversion
            print(f"    Warning: CBS column '{home_value_column}' contains all NaN values. Using neutral multiplier.")
            home_value_column = None


    avg_home_value_overall = 1.0 # Default to neutral multiplier
    if home_value_column:
        # Calculate overall average, handling potential NaNs
        valid_home_values = cbs_df[home_value_column].dropna()
        if not valid_home_values.empty:
            mean_val = valid_home_values.mean()
            if pd.notna(mean_val) and mean_val > 0: # Ensure mean is valid and positive
                avg_home_value_overall = mean_val
            else: # Mean is NaN, zero, or negative
                print(f"    Warning: Overall average home value from '{home_value_column}' is not positive or valid ({mean_val}). Using neutral multiplier.")
                avg_home_value_overall = 1.0 
                home_value_column = None # Disable multiplier if overall average is problematic
        else:
            print(f"    Warning: No valid home values found in CBS column '{home_value_column}' to calculate overall average. Using neutral multiplier.")
            home_value_column = None # Effectively disable multiplier if no valid data
    else:
        print("    No suitable home value column found in CBS data. Using neutral multiplier for fixed costs.")

    fixed_costs_list = []
    for _, sp_row in current_sps_df.iterrows():
        sp_id = sp_row['sp_id']
        sp_x = sp_row['x_rd']
        sp_y = sp_row['y_rd']

        cost_multiplier = 1.0 # Default multiplier

        if nodes_df.empty or 'x_rd' not in nodes_df.columns or 'y_rd' not in nodes_df.columns or 'cbs_square' not in nodes_df.columns:
            print(f"    Warning: Nodes data is incomplete for SP {sp_id}. Using neutral fixed cost multiplier.")
            # cost_multiplier remains 1.0
        elif home_value_column: # Only proceed if a valid home_value_column was identified and avg_home_value_overall is sensible
            distances = np.sqrt((nodes_df['x_rd'] - sp_x)**2 + (nodes_df['y_rd'] - sp_y)**2)
            if distances.empty:
                 print(f"    Warning: Could not calculate distances to nodes for SP {sp_id}. Using neutral fixed cost multiplier.")
                 # cost_multiplier remains 1.0
            else:
                nearest_node_idx = distances.idxmin()
                nearest_node = nodes_df.iloc[nearest_node_idx]
                cbs_square_for_sp = nearest_node['cbs_square']

                square_data = cbs_df[cbs_df['cbs_square'] == cbs_square_for_sp]
                if not square_data.empty and home_value_column in square_data.columns:
                    local_home_value_series = square_data[home_value_column]
                    if not local_home_value_series.empty and not pd.isna(local_home_value_series.iloc[0]):
                        local_avg_home_value = local_home_value_series.iloc[0]
                        
                        if avg_home_value_overall > 0: # This check is crucial
                            calculated_multiplier = local_avg_home_value / avg_home_value_overall
                            # Cap the multiplier to prevent extreme values
                            cost_multiplier = max(0.5, min(calculated_multiplier, 3.0)) # Cap between 0.5x and 3.0x
                            if cost_multiplier <= 0: # Should be handled by max(0.5,...) but as a safeguard
                                cost_multiplier = 1.0 
                        # else: avg_home_value_overall is not positive, multiplier remains 1.0 as set by default
                    # else: local home value is NaN for the square, multiplier remains 1.0
                # else: cbs square not found or column missing for this square, multiplier remains 1.0
        # else: nodes data incomplete or no valid home_value_column, multiplier remains 1.0
        
        yearly_cost = BASE_COST * cost_multiplier
        fixed_costs_list.append({'sp_id': sp_id, 'fixed_yearly_cost': yearly_cost})
        
    return pd.DataFrame(fixed_costs_list)

def simulate_parcel_allocations(current_sps_df, deliveries_df, service_point_nodes_map,
                                sp_to_network_node_map, nodes_df, cbs_df, G_graph,
                                capacity_map):
    """
    Simulates parcel allocations based on delivery_costs.py logic,
    including SP capacity constraints and parcel bouncing, and a per‑SP capacity_map dict.
    Input: Active SPs, deliveries data, coverage maps, graph data, CBS data, capacity_map dict.
    Output: DataFrame of parcel allocations, global bounce rate, dict of local bounce rates, total bounced count.
    """
    print(f"    Simulating parcel allocations for {len(current_sps_df)} SPs with per-SP capacities...")

    if deliveries_df.empty or current_sps_df.empty or 'sp_id' not in current_sps_df.columns or G_graph is None:
        print("    Skipping parcel simulation: Missing deliveries, SPs, sp_id column, or graph.")
        return pd.DataFrame(columns=['day', 'sp_id', 'parcel_id', 'node_id', 'distance_m', 'draw_prob']), 0.0, {}, 0

    # Ensure 'parcels' column exists in deliveries_df, renaming from 'deliveries' if necessary
    if 'parcels' not in deliveries_df.columns and 'deliveries' in deliveries_df.columns:
        print("    Renaming 'deliveries' column to 'parcels' in deliveries_df for simulation.")
        deliveries_df = deliveries_df.rename(columns={'deliveries': 'parcels'})
    elif 'parcels' not in deliveries_df.columns:
        print("    CRITICAL ERROR: 'parcels' column not found in deliveries_df. Cannot simulate parcel allocations.")
        return pd.DataFrame(columns=['day', 'sp_id', 'parcel_id', 'node_id', 'distance_m', 'draw_prob']), 0.0, {}, 0
    
    # Ensure 'parcels' column is numeric and fill NaNs with 0
    deliveries_df['parcels'] = pd.to_numeric(deliveries_df['parcels'], errors='coerce').fillna(0).astype(int)


    # --- Socioeconomic Score Calculation ---
    # This section aims to get a socioeconomic score for each node_id.
    # It uses the 'cbs_square' associated with each node to look up socioeconomic indicators
    # from the cbs_df.

    nodes_ext_df = nodes_df[['node_id', 'cbs_square']].copy()
    nodes_ext_df['node_id'] = nodes_ext_df['node_id'].astype(int)
    nodes_ext_df['cbs_square'] = nodes_ext_df['cbs_square'].astype(str)
    
    cbs_df_copy = cbs_df.copy() # Work on a copy of cbs_df
    cbs_df_copy['cbs_square'] = cbs_df_copy['cbs_square'].astype(str)

    # --- Flexible Socioeconomic Column Detection in cbs_df_copy ---
    # These will be the names of the columns in cbs_df_copy to be merged.
    # If actual columns are not found, fallback columns with 0s will be created in cbs_df_copy.

    # Try to find income column
    socio_col_income_to_use = None
    # Keywords for income, including Dutch terms that might appear after snake_casing
    income_keywords = ['income', 'inkomen'] 
    possible_income_cols = [col for col in cbs_df_copy.columns if any(kw in col.lower() for kw in income_keywords)]
    
    if possible_income_cols:
        socio_col_income_to_use = possible_income_cols[0]
        print(f"    Socioeconomic score: Using CBS column '{socio_col_income_to_use}' for income.")
        if not pd.api.types.is_numeric_dtype(cbs_df_copy[socio_col_income_to_use]):
            print(f"    Warning: Income column '{socio_col_income_to_use}' is not numeric. Attempting conversion.")
            cbs_df_copy[socio_col_income_to_use] = pd.to_numeric(cbs_df_copy[socio_col_income_to_use], errors='coerce')
        cbs_df_copy[socio_col_income_to_use] = cbs_df_copy[socio_col_income_to_use].fillna(0) # Fill NaNs with 0
    else:
        print(f"    Warning: No suitable income column found in CBS data. Income component for socio-economic score will be 0.")
        socio_col_income_to_use = 'socio_income_fallback'
        cbs_df_copy[socio_col_income_to_use] = 0

    # Try to find home value column
    socio_col_home_value_to_use = None
    # Keywords for home value, including Dutch terms like 'woz' or 'waarde'
    home_value_keywords = ['home', 'house', 'value', 'price', 'woz', 'waarde']
    # Prefer columns that also indicate an average or median
    specific_home_value_cols = [
        col for col in cbs_df_copy.columns 
        if any(kw in col.lower() for kw in home_value_keywords) and \
           ('avg' in col.lower() or 'gem' in col.lower() or 'median' in col.lower() or 'gm_' in col.lower())
    ]
    if specific_home_value_cols:
        possible_home_value_cols = specific_home_value_cols
    else: # Broader search if specific not found
        possible_home_value_cols = [col for col in cbs_df_copy.columns if any(kw in col.lower() for kw in home_value_keywords)]

    if possible_home_value_cols:
        socio_col_home_value_to_use = possible_home_value_cols[0]
        print(f"    Socioeconomic score: Using CBS column '{socio_col_home_value_to_use}' for home value.")
        if not pd.api.types.is_numeric_dtype(cbs_df_copy[socio_col_home_value_to_use]):
            print(f"    Warning: Home value column '{socio_col_home_value_to_use}' is not numeric. Attempting conversion.")
            cbs_df_copy[socio_col_home_value_to_use] = pd.to_numeric(cbs_df_copy[socio_col_home_value_to_use], errors='coerce')
        cbs_df_copy[socio_col_home_value_to_use] = cbs_df_copy[socio_col_home_value_to_use].fillna(0) # Fill NaNs with 0
    else:
        print(f"    Warning: No suitable home value column found in CBS data. Home value component for socio-economic score will be 0.")
        socio_col_home_value_to_use = 'socio_home_value_fallback'
        cbs_df_copy[socio_col_home_value_to_use] = 0
    
    # Columns from cbs_df_copy to merge into nodes_ext_df
    cbs_cols_for_merge = ['cbs_square']
    if socio_col_income_to_use not in cbs_cols_for_merge: cbs_cols_for_merge.append(socio_col_income_to_use)
    if socio_col_home_value_to_use not in cbs_cols_for_merge: cbs_cols_for_merge.append(socio_col_home_value_to_use)
    
    # Perform the merge: This is where each node gets its socio-economic data based on its cbs_square
    nodes_ext_df = pd.merge(nodes_ext_df, cbs_df_copy[cbs_cols_for_merge], on='cbs_square', how='left')

    # After merging, fill any NaNs that might have occurred if a cbs_square in nodes_df
    # was not found in cbs_df_copy, or if original data was NaN (though pre-filled in cbs_df_copy).
    # These columns now exist in nodes_ext_df.
    nodes_ext_df[socio_col_income_to_use] = nodes_ext_df[socio_col_income_to_use].fillna(0)
    nodes_ext_df[socio_col_home_value_to_use] = nodes_ext_df[socio_col_home_value_to_use].fillna(0)
            
    # Calculate raw_socio_score using the columns now present in nodes_ext_df
    # .get() is not strictly necessary here as we've ensured columns exist and are filled.
    nodes_ext_df['raw_socio_score'] = nodes_ext_df[socio_col_income_to_use] + \
                                      nodes_ext_df[socio_col_home_value_to_use]

    scaler = MinMaxScaler()
    if not nodes_ext_df['raw_socio_score'].empty and nodes_ext_df['raw_socio_score'].nunique() > 1:
        nodes_ext_df['normalized_socio_score'] = scaler.fit_transform(nodes_ext_df[['raw_socio_score']])
    else:
        nodes_ext_df['normalized_socio_score'] = 0.5

    node_to_norm_socio_map = nodes_ext_df.set_index('node_id')['normalized_socio_score'].to_dict()
    # --- End Socioeconomic Score Calculation ---

    all_successful_allocations_list = []
    distance_cache = {}
    
    # Bouncing related tracking
    bounced_parcels_for_next_day = defaultdict(list) # sp_id -> list of parcel counts
    total_parcels_initially_scheduled = deliveries_df['parcels'].sum() # 'parcels' column now guaranteed to exist and be numeric
    total_parcels_bounced_count = 0
    parcels_scheduled_per_sp = defaultdict(int) # Total originally scheduled for SP before any bouncing
    parcels_bounced_per_sp_count = defaultdict(int) # Total bounced by SP

    # Store original deliveries to calculate scheduled per SP correctly
    original_deliveries_grouped = deliveries_df.groupby(['day', 'sp_id'])['parcels'].sum().reset_index()
    for _, row in original_deliveries_grouped.iterrows():
        parcels_scheduled_per_sp[str(row['sp_id'])] += row['parcels'] # 'parcels' is already int here

    unique_days = sorted(deliveries_df['day'].unique())
    max_sim_day = unique_days[-1] if unique_days else -1

    for day_val in unique_days:
        # deliveries_for_today from original schedule
        daily_deliveries_df = deliveries_df[deliveries_df['day'] == day_val].copy()
        
        # Aggregate parcels by SP for the current day from schedule + bounced
        parcels_to_process_today = defaultdict(int)
        
        # Add originally scheduled parcels for today
        for _, delivery_row in daily_deliveries_df.iterrows():
            sp_id = str(delivery_row['sp_id'])
            # parcels_to_process_today[sp_id] += int(delivery_row.get('parcels', delivery_row.get('deliveries', 0))) # Old line
            # 'parcels' column is now guaranteed to exist and be int, NaNs handled
            parcels_to_process_today[sp_id] += delivery_row['parcels'] 
            
        # Add parcels bounced from the previous day
        if day_val in bounced_parcels_for_next_day:
            for bounced_item in bounced_parcels_for_next_day[day_val]: # Iterate through list of dicts
                parcels_to_process_today[bounced_item['sp_id']] += bounced_item['parcels']
            del bounced_parcels_for_next_day[day_val] # Processed

        for sp_id, num_parcels_for_sp in parcels_to_process_today.items():
            if num_parcels_for_sp == 0:
                continue

            # Ensure sp_id is in the current active service points
            if sp_id not in sp_to_network_node_map:
                # print(f"    Warning: SP {sp_id} (from deliveries/bounced) not in current active SP set. Bouncing its {num_parcels_for_sp} parcels.")
                if day_val < max_sim_day:
                    #  bounced_parcels_for_next_day[day_val + 1].setdefault(sp_id, []).append(num_parcels_for_sp) # Old line
                    bounced_parcels_for_next_day[day_val + 1].append({'sp_id': sp_id, 'parcels': num_parcels_for_sp}) # Corrected: append dict
                total_parcels_bounced_count += num_parcels_for_sp
                parcels_bounced_per_sp_count[sp_id] += num_parcels_for_sp
                continue

            capacity_for_sp = capacity_map.get(sp_id, DEFAULT_SP_CAPACITY)
            
            parcels_to_allocate_this_sp_day = num_parcels_for_sp
            if num_parcels_for_sp > capacity_for_sp:
                num_to_bounce = num_parcels_for_sp - capacity_for_sp
                parcels_to_allocate_this_sp_day = capacity_for_sp
                
                total_parcels_bounced_count += num_to_bounce
                parcels_bounced_per_sp_count[sp_id] += num_to_bounce
                
                if day_val < max_sim_day: # If not the last day, bounce to next day
                    # bounced_parcels_for_next_day[day_val + 1].setdefault(sp_id, []).append(num_to_bounce) # Old line
                    bounced_parcels_for_next_day[day_val + 1].append({'sp_id': sp_id, 'parcels': num_to_bounce}) # Corrected: append dict
                # Parcels bounced on the last day are terminally bounced (already counted)

            if parcels_to_allocate_this_sp_day == 0:
                continue

            # --- Probabilistic allocation for parcels_to_allocate_this_sp_day ---
            sp_network_node = sp_to_network_node_map.get(sp_id)
            if sp_network_node is None: continue # Should be caught above, but as safeguard
            sp_network_node = int(sp_network_node)
            cluster_node_ids = service_point_nodes_map.get(sp_id, [])
            if not cluster_node_ids: continue

            candidate_info_list = []
            for node_id in cluster_node_ids: 
                distance_m = float('inf')
                cache_key = tuple(sorted((sp_network_node, node_id))) 

                if cache_key in distance_cache:
                    distance_m = distance_cache[cache_key]
                else:
                    try:
                        if G_graph.has_node(sp_network_node) and G_graph.has_node(node_id):
                            distance_m = 0.0 if sp_network_node == node_id else nx.shortest_path_length(G_graph, source=sp_network_node, target=node_id, weight='weight')
                        distance_cache[cache_key] = distance_m
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        distance_cache[cache_key] = float('inf')
                
                norm_socio = node_to_norm_socio_map.get(node_id, 0.0)
                dist_component = 0.0
                if distance_m != float('inf') and distance_m >= 0:
                     dist_component = 1.0 / (1.0 + (distance_m / 1000.0)) 
                logit_weight = (0.7 * dist_component) + (0.3 * norm_socio) 
                
                if logit_weight > 1e-9:
                    candidate_info_list.append({'node_id': node_id, 'distance_m': distance_m, 'logit_weight': logit_weight})
            
            if not candidate_info_list: # Fallback if no candidates
                if cluster_node_ids:
                    chosen_nodes_for_sp = np.random.choice(cluster_node_ids, size=parcels_to_allocate_this_sp_day, replace=True)
                    prob_fallback = 1.0 / len(cluster_node_ids)
                    for parcel_idx, assigned_node_id in enumerate(chosen_nodes_for_sp):
                        dist_fallback = distance_cache.get(tuple(sorted((sp_network_node, assigned_node_id))), float('inf'))
                        all_successful_allocations_list.append({
                            'day': day_val, 'sp_id': sp_id, 
                            'parcel_id': f"{sp_id}_{day_val}_{parcel_idx+1}", 'node_id': assigned_node_id,
                            'draw_prob': prob_fallback, 'distance_m': dist_fallback
                        })
                continue

            total_logit_weight = sum(c['logit_weight'] for c in candidate_info_list)
            if total_logit_weight <= 1e-9: # Fallback if all weights are zero
                fallback_nodes = [c['node_id'] for c in candidate_info_list]
                if fallback_nodes:
                    chosen_nodes_for_sp = np.random.choice(fallback_nodes, size=parcels_to_allocate_this_sp_day, replace=True)
                    prob_fallback = 1.0 / len(fallback_nodes)
                    for parcel_idx, assigned_node_id in enumerate(chosen_nodes_for_sp):
                        dist_fallback = distance_cache.get(tuple(sorted((sp_network_node, assigned_node_id))), float('inf'))
                        all_successful_allocations_list.append({
                            'day': day_val, 'sp_id': sp_id, 
                            'parcel_id': f"{sp_id}_{day_val}_{parcel_idx+1}", 'node_id': assigned_node_id,
                            'draw_prob': prob_fallback, 'distance_m': dist_fallback
                        })
                continue
                
            probabilities = [c['logit_weight'] / total_logit_weight for c in candidate_info_list]
            candidate_nodes = [c['node_id'] for c in candidate_info_list]
            node_details_map = {c['node_id']: {'prob': prob, 'dist': c['distance_m']} for c, prob in zip(candidate_info_list, probabilities)}
            
            # Ensure probabilities sum to 1 for np.random.choice
            if not np.isclose(sum(probabilities), 1.0):
                probabilities = np.array(probabilities) / np.sum(probabilities)


            chosen_node_indices = np.random.choice(len(candidate_nodes), size=parcels_to_allocate_this_sp_day, replace=True, p=probabilities)
            
            for parcel_idx, chosen_idx in enumerate(chosen_node_indices):
                assigned_node_id = candidate_nodes[chosen_idx]
                details = node_details_map[assigned_node_id]
                all_successful_allocations_list.append({
                    'day': day_val, 'sp_id': sp_id,
                    'parcel_id': f"{sp_id}_{day_val}_{parcel_idx+1}", 
                    'node_id': assigned_node_id, 'draw_prob': details['prob'], 'distance_m': details['dist']
                })
    
    # Handle any parcels bounced beyond the last simulation day (terminally bounced)
    # These are already counted in total_parcels_bounced_count if bounced_parcels_for_next_day[day_val + 1] was used on last day.

    global_bounce_rate = (total_parcels_bounced_count / total_parcels_initially_scheduled) if total_parcels_initially_scheduled > 0 else 0.0
    
    local_bounce_rates_map = {}
    for sp_id_key in current_sps_df['sp_id'].astype(str).unique(): # Iterate over active SPs
        scheduled_for_sp = parcels_scheduled_per_sp.get(sp_id_key, 0)
        bounced_by_sp = parcels_bounced_per_sp_count.get(sp_id_key, 0)
        local_bounce_rates_map[sp_id_key] = (bounced_by_sp / scheduled_for_sp) if scheduled_for_sp > 0 else 0.0
        
    return pd.DataFrame(all_successful_allocations_list), global_bounce_rate, local_bounce_rates_map, total_parcels_bounced_count


def calculate_total_operational_costs(fixed_costs_df, parcel_allocations_df,
                                      global_bounce_rate, local_bounce_rates_map, total_parcels_bounced_globally,
                                      active_sp_ids):
    """
    Calculates total operational costs including fixed, home delivery, storage, and penalties.
    Input: fixed_costs_df, parcel_allocations_df (successfully delivered), bounce stats, active_sp_ids list
    Output: Dictionary of cost components and grand_total_cost
    """
    print(f"    Calculating total operational costs...")
    costs = {
        'fixed': 0,
        'home_delivery': 0,
        'storage': 0,
        'penalty_bounce': 0,
        'grand_total': 0
    }

    if not fixed_costs_df.empty:
        costs['fixed'] = fixed_costs_df['fixed_yearly_cost'].sum()

    # Home Delivery costs (assuming all in parcel_allocations_df are home-delivered, round trip)
    home_delivery_total_cost = 0
    if not parcel_allocations_df.empty and 'distance_m' in parcel_allocations_df.columns:
        # Ensure 'distance_m' is numeric and handle NaNs by converting to 0 for cost calculation
        parcel_allocations_df['distance_m_numeric'] = pd.to_numeric(parcel_allocations_df['distance_m'], errors='coerce').fillna(0)
        
        parcel_allocations_df['distance_km'] = parcel_allocations_df['distance_m_numeric'] / 1000.0
        # Cost for round trip
        parcel_allocations_df['delivery_cost_indiv'] = parcel_allocations_df['distance_km'] * HOME_DELIVERY_COST_PER_KM * 2 
        
        if 'day' in parcel_allocations_df.columns and parcel_allocations_df['day'].nunique() > 0:
            num_sim_days = parcel_allocations_df['day'].nunique()
            total_sim_period_delivery_cost = parcel_allocations_df['delivery_cost_indiv'].sum()
            home_delivery_total_cost = (total_sim_period_delivery_cost / num_sim_days) * 365
        elif not parcel_allocations_df.empty: # Fallback if 'day' is missing or only one day
             # Assume simulation period is 30 days if not specified by multiple unique days
            total_sim_period_delivery_cost = parcel_allocations_df['delivery_cost_indiv'].sum()
            home_delivery_total_cost = total_sim_period_delivery_cost * (365 / 30.0) 
    costs['home_delivery'] = home_delivery_total_cost

    # --- 4.5 Calculate Storage Costs ---
    print("\nStep 4.5: Calculating Yearly Storage Costs per Service Point")
    STORAGE_COST_PER_PACKAGE_DAY = 0.10  # € per parcel‑day

    if parcel_allocations_df.empty or 'sp_id' not in parcel_allocations_df.columns or 'day' not in parcel_allocations_df.columns:
        print("  Parcel allocations data is empty or missing 'sp_id'/'day' columns. Storage costs will be zero.")
        storage_costs_per_sp_df = pd.DataFrame(columns=['storage_yearly_cost'])
        storage_costs_per_sp_df.index.name = 'service_point_id'
    else:
        # Each row in parcel_allocations_df represents one parcel.
        # Group by day and sp_id, then count parcels for each sp_id on each day.
        daily_parcels_per_sp = parcel_allocations_df.groupby(['day', 'sp_id']).size().reset_index(name='num_daily_parcels')

        if daily_parcels_per_sp.empty:
            print("  No daily parcel counts could be derived. Storage costs will be zero.")
            storage_costs_per_sp_df = pd.DataFrame(columns=['storage_yearly_cost'])
            storage_costs_per_sp_df.index.name = 'service_point_id'
        else:
            # Find the maximum number of packages for each SP on any simulation day
            max_parcels_per_sp = daily_parcels_per_sp.groupby('sp_id')['num_daily_parcels'].max()

            # Calculate yearly storage cost
            storage_yearly_cost_per_sp = max_parcels_per_sp * STORAGE_COST_PER_PACKAGE_DAY * 365

            storage_costs_per_sp_df = storage_yearly_cost_per_sp.reset_index()
            storage_costs_per_sp_df.rename(columns={'num_daily_parcels': 'storage_yearly_cost',
                                                    'sp_id': 'service_point_id'}, inplace=True)
            storage_costs_per_sp_df.set_index('service_point_id', inplace=True)
            # Ensure index is string
            storage_costs_per_sp_df.index = storage_costs_per_sp_df.index.astype(str)
            print(f"  Calculated yearly storage costs for {len(storage_costs_per_sp_df)} service points.")

    # Aggregate across all service points for the objective function
    storage_total_cost = storage_costs_per_sp_df['storage_yearly_cost'].sum() if not storage_costs_per_sp_df.empty else 0
    costs['storage'] = storage_total_cost
    
    # Penalty costs for bouncing - NOW A HARD CONSTRAINT
    if global_bounce_rate > GLOBAL_BOUNCE_RATE_THRESHOLD:
        print(f"    HARD CONSTRAINT VIOLATED: Global bounce rate {global_bounce_rate:.2%} > {GLOBAL_BOUNCE_RATE_THRESHOLD:.2%}. Cost is infinite.")
        return float('inf')

    for sp_id_key in active_sp_ids: # Check local rates for all active SPs
        rate = local_bounce_rates_map.get(str(sp_id_key), 0.0) # Ensure sp_id_key is string for lookup
        if rate > LOCAL_BOUNCE_RATE_THRESHOLD:
            print(f"    HARD CONSTRAINT VIOLATED: Local bounce rate for SP {sp_id_key} {rate:.2%} > {LOCAL_BOUNCE_RATE_THRESHOLD:.2%}. Cost is infinite.")
            return float('inf')
    
    # If bounce rate constraints are met, the penalty cost is zero.
    costs['penalty_bounce'] = 0

    costs['grand_total'] = costs['fixed'] + costs['home_delivery'] + costs['storage'] + costs['penalty_bounce']
    
    print(f"    Costs: Fixed={costs['fixed']:.0f}, HomeDelivery={costs['home_delivery']:.0f}, Storage={costs['storage']:.0f}, Penalty={costs['penalty_bounce']:.0f} -> Total={costs['grand_total']:.0f}")
    return costs['grand_total'] # For heuristic comparison, return total. Can return dict if needed by caller.


def evaluate_solution(current_sps_df, all_data):
    """
    Calculates the total cost for a given set of service points.
    `all_data` is a dictionary holding nodes_df, edges_df, deliveries_df, cbs_df
    """
    if current_sps_df.empty:
        print("    Evaluating an empty SP configuration. Cost is effectively infinite (or very high).")
        return float('inf') # Or a very large number

    nodes_df = all_data['nodes']
    edges_df = all_data['edges']
    deliveries_df = all_data['deliveries'] # Original deliveries
    cbs_df = all_data['cbs']

    coverage_results = calculate_service_point_coverage(current_sps_df, nodes_df, edges_df)
    
    service_point_nodes_map = coverage_results["service_point_nodes"]
    sp_to_network_node_map = coverage_results["sp_to_network_node"]
    G_for_simulation = coverage_results["graph"]

    active_sp_ids_in_coverage = set(service_point_nodes_map.keys())
    
    # Filter original deliveries to only include those whose initially assigned SP is currently active.
    # Bouncing logic within simulate_parcel_allocations will handle if an SP (even if active) cannot be found in sp_to_network_node_map.
    deliveries_df_for_active_sps = deliveries_df[deliveries_df['sp_id'].astype(str).isin(active_sp_ids_in_coverage)].copy()
    
    if deliveries_df_for_active_sps.empty and not deliveries_df.empty:
        print(f"    Warning: No deliveries match active SPs for initial scheduling. Original deliveries: {len(deliveries_df)}, Filtered for active SPs: 0")
        # If no deliveries can even be scheduled for the active SPs, parcel_allocations will be empty.
        # Fixed costs will still apply. Bounce rate will be 0 or undefined.

    fixed_costs_df = calculate_fixed_costs(current_sps_df, cbs_df, nodes_df, service_point_nodes_map)

    # Build capacity map (str -> int)
    if 'capacity' not in current_sps_df.columns:
        current_sps_df['capacity'] = DEFAULT_SP_CAPACITY
    capacity_map = dict(zip(current_sps_df['sp_id'].astype(str), current_sps_df['capacity'].astype(int)))

    parcel_allocations_df, global_bounce_rate, local_bounce_rates_map, total_bounced = \
        simulate_parcel_allocations(
            current_sps_df, deliveries_df_for_active_sps, 
            service_point_nodes_map, sp_to_network_node_map,
            nodes_df, cbs_df, G_for_simulation,
            capacity_map
        )
    
    # Get list of sp_ids from the current solution DataFrame for penalty checking
    active_sp_ids_list = current_sps_df['sp_id'].astype(str).tolist() if 'sp_id' in current_sps_df.columns else []

    total_cost = calculate_total_operational_costs(
        fixed_costs_df, parcel_allocations_df,
        global_bounce_rate, local_bounce_rates_map, total_bounced,
        active_sp_ids_list # Pass the list of active SP IDs
    )
    
    # To see the breakdown, you might want evaluate_solution to return the dict from calculate_total_operational_costs
    # For now, it returns the grand total for the heuristic's comparison.
    return total_cost

# --- Main Heuristic Functions ---

def evaluate_solution(current_sps_df, all_data):
    """
    Calculates the total cost for a given set of service points.
    `all_data` is a dictionary holding nodes_df, edges_df, deliveries_df, cbs_df
    """
    if current_sps_df.empty:
        print("    Evaluating an empty SP configuration. Cost is effectively infinite (or very high).")
        return float('inf')

    nodes_df = all_data['nodes']
    edges_df = all_data['edges']
    deliveries_df = all_data['deliveries']
    cbs_df = all_data['cbs']

    # 1. Calculate coverage and get the graph
    # Ensures node_ids are consistently int and sp_ids str in the returned maps
    coverage_results = calculate_service_point_coverage(current_sps_df, nodes_df, edges_df)
    
    service_point_nodes_map = coverage_results["service_point_nodes"]
    sp_to_network_node_map = coverage_results["sp_to_network_node"]
    G_for_simulation = coverage_results["graph"] # Use the graph returned by coverage calculation

    # ------------------------------------------------------------------
    # 1b. Reallocate demand from inactive SPs to the nearest ACTIVE SP
    # ------------------------------------------------------------------
    all_potential_sps_df = all_data['all_potential_sps']  # Needed for coordinates of deleted SPs

    # Quick lookup dictionaries for coordinates
    coord_map_all = all_potential_sps_df.set_index('sp_id')[['x_rd', 'y_rd']].to_dict('index')
    active_coord_map = current_sps_df.set_index('sp_id')[['x_rd', 'y_rd']].to_dict('index')

    def _nearest_active_sp(orig_sp_id_str: str) -> str:
        """Return the ID of the nearest active SP (Euclidean distance)."""
        # If already active, keep as is
        if orig_sp_id_str in active_coord_map:
            return orig_sp_id_str

        # Safety: if somehow no active SPs, return original ID unchanged
        if not active_coord_map:
            return orig_sp_id_str

        # Coordinates of the original (now inactive) SP
        if orig_sp_id_str not in coord_map_all:
            # Fallback: assign randomly to one of the active SPs
            return random.choice(list(active_coord_map.keys()))

        ox, oy = coord_map_all[orig_sp_id_str]['x_rd'], coord_map_all[orig_sp_id_str]['y_rd']
        best_id, best_dist = None, float('inf')
        for act_id, coord in active_coord_map.items():
            dx = coord['x_rd'] - ox
            dy = coord['y_rd'] - oy
            dist = (dx * dx + dy * dy) ** 0.5  # Euclidean distance
            if dist < best_dist:
                best_dist = dist
                best_id = act_id
        return str(best_id)

    # Apply reallocation
    deliveries_df_reallocated = deliveries_df.copy()
    deliveries_df_reallocated['sp_id'] = deliveries_df_reallocated['sp_id'].astype(str).apply(_nearest_active_sp)

    # After reallocation, every sp_id should be active, so no further filtering needed
    deliveries_df_filtered = deliveries_df_reallocated
    # ------------------------------------------------------------------


    # 2. Calculate fixed costs
    # The coverage_data argument for calculate_fixed_costs was originally the service_point_nodes_map.
    fixed_costs_df = calculate_fixed_costs(current_sps_df, cbs_df, nodes_df, service_point_nodes_map)

    # Build capacity map (str -> int)
    if 'capacity' not in current_sps_df.columns:
        current_sps_df['capacity'] = DEFAULT_SP_CAPACITY
    capacity_map = dict(zip(current_sps_df['sp_id'].astype(str), current_sps_df['capacity'].astype(int)))

    # 3. Simulate parcel allocations ...
    parcel_allocations_df, global_bounce_rate, local_bounce_rates_map, total_bounced = \
        simulate_parcel_allocations(
            current_sps_df, deliveries_df_filtered, # Use filtered deliveries
            service_point_nodes_map, sp_to_network_node_map,
            nodes_df, cbs_df, G_for_simulation,
            capacity_map
        )

    # 4. Calculate total operational costs
    active_sp_ids_list = current_sps_df['sp_id'].astype(str).tolist() if 'sp_id' in current_sps_df.columns else []
    total_cost = calculate_total_operational_costs(
        fixed_costs_df, parcel_allocations_df,
        global_bounce_rate, local_bounce_rates_map, total_bounced,
        active_sp_ids_list
    )
    
    return total_cost

def generate_neighbor_solution(current_sps_df, all_potential_sps_df):
    """
    Generates a neighbor solution by adding, removing, or adjusting capacity of SP(s).
    """
    neighbor_sps_df = current_sps_df.copy()
    # Give more weight to adjust_capacity if there are SPs to adjust
    actions = ["add", "remove"]
    if not neighbor_sps_df.empty and 'capacity' in neighbor_sps_df.columns:
        actions.extend(["adjust_capacity", "adjust_capacity"]) # Increase probability of capacity adjustment

    action = random.choice(actions)

    if action == "add":
        active_sp_ids = set(neighbor_sps_df['sp_id'].unique())
        available_to_add = all_potential_sps_df[~all_potential_sps_df['sp_id'].isin(active_sp_ids)]
        if not available_to_add.empty:
            sp_to_add = available_to_add.sample(1)
            # Ensure the new SP gets its capacity from all_potential_sps_df (which should be MIN_SP_CAPACITY if new)
            if 'capacity' not in sp_to_add.columns and 'capacity' in all_potential_sps_df.columns:
                 sp_to_add['capacity'] = all_potential_sps_df.loc[sp_to_add.index, 'capacity'].values # Ensure it carries capacity
            elif 'capacity' not in sp_to_add.columns:
                 sp_to_add['capacity'] = MIN_SP_CAPACITY


            neighbor_sps_df = pd.concat([neighbor_sps_df, sp_to_add], ignore_index=True).drop_duplicates(subset=['sp_id']).reset_index(drop=True)
            print(f"    Generated neighbor by ADDING SP: {sp_to_add['sp_id'].iloc[0]} with capacity {sp_to_add['capacity'].iloc[0]}")
        else: # No SPs to add, try removing instead if possible
            action = "remove" 
            print(f"    No SPs to add, trying to remove instead.")

    if action == "remove": # Handles original "remove" or fallback from "add"
        if len(neighbor_sps_df) > 1: # Ensure we don't remove the last SP, or handle min SPs constraint
            sp_to_remove_idx = random.choice(neighbor_sps_df.index)
            removed_sp_id = neighbor_sps_df.loc[sp_to_remove_idx, 'sp_id']
            neighbor_sps_df = neighbor_sps_df.drop(sp_to_remove_idx).reset_index(drop=True)
            print(f"    Generated neighbor by REMOVING SP: {removed_sp_id}")
        elif len(neighbor_sps_df) == 1 and "adjust_capacity" in actions: # If only one SP, try to adjust capacity instead
            action = "adjust_capacity"
            print(f"    Attempted to remove, but only one SP left. Trying to adjust capacity instead.")
        elif len(neighbor_sps_df) == 1:
            print(f"    Attempted to remove, but only one SP left. No change.")

    if action == "adjust_capacity":
        if not neighbor_sps_df.empty and 'capacity' in neighbor_sps_df.columns:
            num_to_adjust = random.randint(1, min(3, len(neighbor_sps_df)))
            idxs_to_change = random.sample(list(neighbor_sps_df.index), num_to_adjust)
            adjusted_sps_info = []
            for idx in idxs_to_change:
                cur_cap = neighbor_sps_df.loc[idx, 'capacity']
                sp_id_adjusted = neighbor_sps_df.loc[idx, 'sp_id']
                
                # Always try to shrink capacity
                if cur_cap > MIN_SP_CAPACITY:
                    new_cap = max(MIN_SP_CAPACITY, cur_cap - CAPACITY_STEP)
                else:
                    new_cap = MIN_SP_CAPACITY # Already at min, no change from shrinking

                if new_cap != cur_cap:
                    neighbor_sps_df.loc[idx, 'capacity'] = new_cap
                    adjusted_sps_info.append(f"SP {sp_id_adjusted}: {cur_cap} -> {new_cap}")
                else:
                    adjusted_sps_info.append(f"SP {sp_id_adjusted}: {cur_cap} (no change)")

            if adjusted_sps_info:
                print(f"    ADJUSTED capacity (tried to shrink) for {len(idxs_to_change)} SP(s): {'; '.join(adjusted_sps_info)}.")
            else:
                print(f"    Attempted to adjust capacity, but no SPs were eligible for change or no change occurred.")
        elif not neighbor_sps_df.empty:
             print(f"    Attempted to adjust capacity, but 'capacity' column missing.")


    return neighbor_sps_df



def run_iterative_improvement_heuristic():
    print("Starting Iterative Improvement Heuristic...")
    all_potential_sps_df = data["all_potential_sps"].copy()
    deliveries_df = data["deliveries"].copy()

    # --- Initialize SP capacities ---
    all_potential_sps_df['sp_id'] = all_potential_sps_df['sp_id'].astype(str)

    # Initialize 'capacity' column to NaN. This allows us to distinguish SPs
    # that get capacity from history vs. those that need a default.
    all_potential_sps_df['capacity'] = np.nan
    print(f"    Initializing 'capacity' for {len(all_potential_sps_df)} potential SPs as undefined (NaN).")

    # Step 1: Set capacity for SPs with historical data to their max observed daily parcels.
    if not deliveries_df.empty and 'sp_id' in deliveries_df.columns and 'parcels' in deliveries_df.columns:
        deliveries_df['sp_id'] = deliveries_df['sp_id'].astype(str)
        
        max_observed_parcels = deliveries_df.groupby('sp_id')['parcels'].max().reset_index()
        max_observed_parcels.rename(columns={'parcels': 'historical_max_capacity'}, inplace=True)
        
        all_potential_sps_df = pd.merge(all_potential_sps_df, max_observed_parcels, on='sp_id', how='left')
        
        all_potential_sps_df['capacity'] = np.where(
            all_potential_sps_df['historical_max_capacity'].notna(),
            all_potential_sps_df['historical_max_capacity'],
            all_potential_sps_df['capacity'] 
        )
        
        count_updated_by_history = all_potential_sps_df['historical_max_capacity'].notna().sum()
        print(f"    Set capacities for {count_updated_by_history} SPs based on their historical max daily parcels.")
        
        all_potential_sps_df.drop(columns=['historical_max_capacity'], inplace=True, errors='ignore')
    else:
        print(f"    No historical delivery data provided or usable for setting initial capacities based on max demand.")

    # Step 2: For SPs whose capacity is still NaN (i.e., new SPs with no historical data),
    # set their capacity to MAX_SP_CAPACITY.
    all_potential_sps_df['capacity'].fillna(MAX_SP_CAPACITY, inplace=True)
    print(f"    SPs with still undefined capacity (e.g., new SPs with no history) set to MAX_SP_CAPACITY ({MAX_SP_CAPACITY}).")
    
    # Step 3: Ensure 'capacity' is numeric, clip all capacities to defined bounds, and ensure integer type.
    all_potential_sps_df['capacity'] = pd.to_numeric(all_potential_sps_df['capacity'], errors='coerce').fillna(MAX_SP_CAPACITY) 
    
    all_potential_sps_df['capacity'] = all_potential_sps_df['capacity'].clip(
        lower=MIN_SP_CAPACITY, upper=MAX_SP_CAPACITY
    ).astype(int)
    print(f"    All {len(all_potential_sps_df)} potential SP capacities are now initialized and clipped between {MIN_SP_CAPACITY} and {MAX_SP_CAPACITY}.")
    # --- End of capacity initialization ---

    if all_potential_sps_df.empty:
        print("No potential service points defined. Exiting.")
        return

    # Initial solution: start with a random subset
    num_initial_sps = max(1, min(5, len(all_potential_sps_df) // 2)) 
    # Ensure random_state for reproducibility if desired, or remove for true randomness each run
    current_sps_df = all_potential_sps_df.sample(num_initial_sps, random_state=np.random.RandomState()) if len(all_potential_sps_df) >= num_initial_sps else all_potential_sps_df.copy()

    # The 'capacity' column should now be correctly set from all_potential_sps_df
    if 'capacity' not in current_sps_df.columns: # Should not happen if logic above is correct
        print("    Warning: 'capacity' column missing in initial_sps_df after sampling. Assigning default.")
        current_sps_df['capacity'] = DEFAULT_SP_CAPACITY
    else:
        current_sps_df['capacity'] = current_sps_df['capacity'].astype(int)

    
    print(f"Initial SPs ({len(current_sps_df)}):")
    if 'sp_id' in current_sps_df.columns and 'capacity' in current_sps_df.columns:
        for _, row in current_sps_df.iterrows():
            print(f"  - SP ID: {row['sp_id']}, Initial Capacity: {row['capacity']}")
    else:
        print("  Initial SPs DataFrame is missing 'sp_id' or 'capacity' columns.")


    current_cost = evaluate_solution(current_sps_df, data) # Pass the global 'data'
    print(f"Initial cost: {current_cost:.2f}\n")

    best_sps_df = current_sps_df.copy()
    best_cost = current_cost

    for i in range(MAX_ITERATIONS):
        print(f"--- Iteration {i+1}/{MAX_ITERATIONS} ---")
        
        best_neighbor_in_iteration_df = None
        best_neighbor_in_iteration_cost = current_cost 

        improved_in_iteration = False
        for _ in range(NEIGHBORHOOD_SIZE): 
            neighbor_sps_df = generate_neighbor_solution(current_sps_df, all_potential_sps_df)
            
            if neighbor_sps_df.equals(current_sps_df) or neighbor_sps_df.empty:
                continue

            neighbor_cost = evaluate_solution(neighbor_sps_df, data) # Pass the global 'data'
            print(f"    Neighbor SPs ({len(neighbor_sps_df)}), Cost: {neighbor_cost:.2f}")

            if neighbor_cost < best_neighbor_in_iteration_cost:
                best_neighbor_in_iteration_df = neighbor_sps_df.copy()
                best_neighbor_in_iteration_cost = neighbor_cost
                improved_in_iteration = True
        
        if improved_in_iteration and best_neighbor_in_iteration_cost < current_cost:
            current_sps_df = best_neighbor_in_iteration_df.copy()
            current_cost = best_neighbor_in_iteration_cost
            print(f"  Moved to new solution. Cost: {current_cost:.2f}, SPs: {current_sps_df['sp_id'].tolist() if 'sp_id' in current_sps_df.columns else 'None'}")

            if current_cost < best_cost:
                best_sps_df = current_sps_df.copy()
                best_cost = current_cost
                print(f"  *** New global best solution found! Cost: {best_cost:.2f} ***")
        else:
            print(f"  No improvement found in iteration {i+1}. Stopping.")
            break 
            
    print("\n--- Heuristic Finished ---")
    print(f"Best solution found with cost: {best_cost:.2f}")
    print("Best service point configuration:")
    if 'sp_id' in best_sps_df.columns and not best_sps_df.empty:
        print(best_sps_df[['sp_id', 'x_rd', 'y_rd']])
        best_sps_df.to_csv("best_heuristic_service_points.csv", index=False)
        print("Saved best configuration to 'best_heuristic_service_points.csv'")
    else:
        print("Best SP DataFrame is empty or missing 'sp_id'.")


if __name__ == "__main__":
    run_iterative_improvement_heuristic()