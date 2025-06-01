#to who ever is going to read this code I am terably sorry
#did my best :/

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Configuration Constants ---
SP_TO_NETWORK_NODE_FILE = 'sp_to_network_node.json'
SERVICE_POINT_COVERAGE_FILE = 'service_point_coverage.json'
RANDOM_SEED = 42
DEMAND_SIM_CONFIG = {
    'W_DIST_PICKUP': 0.7,
    'W_SOCIO_PICKUP': 0.3,
    'W_DIST_ATHOME': 0.3,
    'W_SOCIO_ATHOME': 0.7,
    'DEFAULT_SOCIO_SCORE_IF_MISSING': 0.5
}

# --- Plotting Colors (Add these) ---
COLOR_PICKUP_NODE = 'red'
COLOR_ATHOME_NODE = 'lightblue'
COLOR_MIXED_NODE = 'purple'
COLOR_NO_DEMAND_NODE = '#d3d3d3' # Light gray for nodes with no demand in simulation
COLOR_SERVICE_POINT = 'darkorange'
COLOR_EDGES = '#cccccc' # Light gray for network edges

#tempature presets
INITIAL_TEMPERATURE = 1000  # Initial temperature for simulated annealing
MAX_ITERATIONS_PER_TEMP = 50  # Maximum iterations per temperature level
MIN_TEMPERATURE = 1  # Minimum temperature to stop the annealing process
MAX_TOTAL_ITERATIONS = 1000  # Maximum total iterations for the entire simulated annealing process
COOLING_RATE = 0.99  # Cooling rate for temperature reduction
# Generate demand simulation data for Maastricht
print("Step 1: Loading data and initial setup...")
np.random.seed(RANDOM_SEED)

# --- Load Base Data ---
try:
    data = load_maastricht_data()
    nodes_df = data["nodes"].copy()
    edges_df = data["edges"].copy()
    cbs_df = data["cbs"].copy()
    deliveries_df = data["deliveries"].copy()
    pickups_df = data.get("pickups", pd.DataFrame()).copy()
    service_points_df = data["service_points"].copy()

    # Basic preprocessing for deliveries_df and pickups_df (from simulated_anealing.py)
    deliveries_df['parcels'] = deliveries_df['parcels'].fillna(0).astype(int)
    if not pickups_df.empty:
        if 'parcels' in pickups_df.columns:
            pickups_df['parcels'] = pickups_df['parcels'].fillna(0).astype(int)
        elif 'pickups' in pickups_df.columns:
            pickups_df.rename(columns={'pickups': 'parcels'}, inplace=True)
            pickups_df['parcels'] = pickups_df['parcels'].fillna(0).astype(int)
        else:
            pickups_df['parcels'] = 0 # Ensure parcels column exists
    else: # Ensure pickups_df has expected columns even if empty
        pickups_df = pd.DataFrame(columns=['day', 'sp_id', 'parcels'])


except ImportError:
    print("ERROR: Could not import 'load_maastricht_data' from 'imp_data'. Ensure imp_data.py is accessible.")
    exit()
except Exception as e:
    print(f"ERROR: Failed to load data using imp_data.py: {e}")
    exit()

# --- Load Pre-calculated Maps ---
try:
    with open(SERVICE_POINT_COVERAGE_FILE, 'r') as f:
        coverage_data = json.load(f)
    service_point_nodes_map = coverage_data.get('service_point_nodes', {})
    service_point_nodes_map = {str(k): [int(n) for n in v] for k, v in service_point_nodes_map.items()}
except Exception as e:
    print(f"ERROR loading {SERVICE_POINT_COVERAGE_FILE}: {e}")
    exit()

try:
    with open(SP_TO_NETWORK_NODE_FILE, 'r') as f:
        sp_to_network_node_map = json.load(f)
    sp_to_network_node_map = {str(k): int(v) for k, v in sp_to_network_node_map.items()}
except Exception as e:
    print(f"ERROR loading {SP_TO_NETWORK_NODE_FILE}: {e}")
    exit()

# --- Create Graph ---
G = nx.Graph()
for _, node_row in nodes_df.iterrows():
    G.add_node(int(node_row['node_id']), pos=(float(node_row['x_rd']), float(node_row['y_rd'])))
for _, edge_row in edges_df.iterrows():
    G.add_edge(int(edge_row['from_node']), int(edge_row['to_node']), weight=float(edge_row['length_m']))
print(f"Graph G created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# --- Socioeconomic Score Calculation (adapted from simulated_anealing.py) ---
print("Step 2: Preprocessing node data for socioeconomic scores...")
nodes_df['cbs_square'] = nodes_df['cbs_square'].astype(str)
cbs_df['cbs_square'] = cbs_df['cbs_square'].astype(str)
SOCIO_COL_INCOME = 'median_income_k€' # Adjust if your column names differ
SOCIO_COL_HOME_VALUE = 'avg_home_value_k€' # Adjust if your column names differ
cbs_relevant_cols = ['cbs_square', SOCIO_COL_INCOME, SOCIO_COL_HOME_VALUE]
for col_name in [SOCIO_COL_INCOME, SOCIO_COL_HOME_VALUE]:
    if col_name not in cbs_df.columns:
        print(f"Warning: CBS data missing '{col_name}'. Filling with 0.")
        cbs_df[col_name] = 0
nodes_ext_df = pd.merge(nodes_df[['node_id', 'cbs_square']], cbs_df[cbs_relevant_cols], on='cbs_square', how='left')
for col in [SOCIO_COL_INCOME, SOCIO_COL_HOME_VALUE]:
    median_val = nodes_ext_df[col].median()
    if pd.isna(median_val): median_val = 0
    nodes_ext_df[col].fillna(median_val, inplace=True)
nodes_ext_df['raw_socio_score'] = nodes_ext_df[SOCIO_COL_INCOME] + nodes_ext_df[SOCIO_COL_HOME_VALUE]
scaler = MinMaxScaler()
if not nodes_ext_df['raw_socio_score'].empty and nodes_ext_df['raw_socio_score'].nunique() > 1:
    nodes_ext_df['normalized_socio_score'] = scaler.fit_transform(nodes_ext_df[['raw_socio_score']])
else:
    nodes_ext_df['normalized_socio_score'] = 0.5
node_to_norm_socio_map = nodes_ext_df.set_index('node_id')['normalized_socio_score'].to_dict()
print("Socioeconomic factors calculated.")

# --- Demand Simulation Functions (copied & adapted from simulated_anealing.py) ---
distance_cache = {}

def get_dynamic_pickup_probability(distance_meters):
    """
    Calculates the probability of a customer picking up a parcel based on distance.
    - <= 200m: 80%
    - > 2000m: 0%
    - Linear interpolation between 200m and 2000m.
    """
    if distance_meters <= 200:
        return 0.8
    elif distance_meters > 2000:
        return 0.0
    else:
        # Linear interpolation: P(d) = P_near - ( (d - D_near) / (D_far - D_near) * (P_near - P_far) )
        # P_near = 0.8 at D_near = 200
        # P_far = 0.0 at D_far = 2000
        # P(d) = 0.8 - ( (distance_meters - 200) / (2000 - 200) * (0.8 - 0.0) )
        return 0.8 - (((distance_meters - 200.0) / 1800.0) * 0.8)

def _select_node_for_delivery_parcel(base_candidate_info_list, config, parcel_status_for_weighting='at home'):
    """
    Selects a single node for a parcel, using specified attractiveness logic.
    Returns the chosen node_info dictionary or None.
    """
    if not base_candidate_info_list:
        return None

    weighted_candidates = []
    for cand_info in base_candidate_info_list:
        dist_m = cand_info['distance_m']
        norm_socio = cand_info['norm_socio_score']
        dist_component = 0.0
        if dist_m != float('inf') and dist_m >= 0:
            dist_component = 1.0 / (1.0 + dist_m / 1000.0)
        
        logit_w = 0.0
        if parcel_status_for_weighting == 'pick up': # Should primarily be used for 'at home' here
            socio_factor_pickup = 1.0 - norm_socio
            logit_w = (config['W_DIST_PICKUP'] * dist_component) + (config['W_SOCIO_PICKUP'] * socio_factor_pickup)
        else: # 'at home'
            socio_factor_athome = norm_socio
            logit_w = (config['W_DIST_ATHOME'] * dist_component) + (config['W_SOCIO_ATHOME'] * socio_factor_athome)
        
        if logit_w > 1e-9:
            weighted_candidates.append({'node_id': cand_info['node_id'], 'distance_m': dist_m, 'logit_weight': logit_w, 'norm_socio_score': norm_socio})

    if not weighted_candidates: # Fallback if no positive weights
        return random.choice(base_candidate_info_list) if base_candidate_info_list else None

    total_logit_weight = sum(c['logit_weight'] for c in weighted_candidates)
    if total_logit_weight <= 1e-9: # Further fallback
        return random.choice(weighted_candidates) if weighted_candidates else None
        
    probabilities = [c['logit_weight'] / total_logit_weight for c in weighted_candidates]
    chosen_candidate_idx = np.random.choice(len(weighted_candidates), p=probabilities)
    return weighted_candidates[chosen_candidate_idx]


def _assign_parcels_of_type(
    num_parcels_to_assign, parcel_status, base_candidate_info_list,
    sp_id_str, day_val, parcel_idx_start, config, sp_network_node, dist_cache_ref
):
    assigned_parcels_for_type = []
    if num_parcels_to_assign == 0 or not base_candidate_info_list:
        return assigned_parcels_for_type, parcel_idx_start
    type_specific_candidates = []
    for cand_info in base_candidate_info_list:
        dist_m = cand_info['distance_m']
        norm_socio = cand_info['norm_socio_score']
        dist_component = 0.0
        if dist_m != float('inf') and dist_m >= 0: # Ensure dist_m is non-negative
             dist_component = 1.0 / (1.0 + dist_m / 1000.0) # Normalize distance a bit (e.g. per km)
        logit_w = 0.0
        if parcel_status == 'pick up':
            socio_factor_pickup = 1.0 - norm_socio
            logit_w = (config['W_DIST_PICKUP'] * dist_component) + (config['W_SOCIO_PICKUP'] * socio_factor_pickup)
        else: # at home
            socio_factor_athome = norm_socio
            logit_w = (config['W_DIST_ATHOME'] * dist_component) + (config['W_SOCIO_ATHOME'] * socio_factor_athome)
        if logit_w > 1e-9:
            type_specific_candidates.append({'node_id': cand_info['node_id'], 'distance_m': dist_m, 'logit_weight': logit_w})

    if not type_specific_candidates: # Fallback
        if base_candidate_info_list:
            chosen_nodes = np.random.choice([c['node_id'] for c in base_candidate_info_list], size=num_parcels_to_assign, replace=True)
            prob_fallback = 1.0 / len(base_candidate_info_list) if base_candidate_info_list else 0
            for i, assigned_node_id in enumerate(chosen_nodes):
                fallback_dist = dist_cache_ref.get(tuple(sorted((sp_network_node, assigned_node_id))), float('inf'))
                assigned_parcels_for_type.append({'day': day_val, 'sp_id': sp_id_str, 'parcel_id': f"{sp_id_str}_{parcel_idx_start + i + 1}", 'node_id': assigned_node_id, 'status': parcel_status, 'draw_prob': prob_fallback, 'distance_m': fallback_dist})
            parcel_idx_start += num_parcels_to_assign
        return assigned_parcels_for_type, parcel_idx_start

    total_logit_weight = sum(c['logit_weight'] for c in type_specific_candidates)
    if total_logit_weight <= 1e-9: # Further fallback
        chosen_nodes = np.random.choice([c['node_id'] for c in type_specific_candidates], size=num_parcels_to_assign, replace=True)
        prob_fallback = 1.0 / len(type_specific_candidates) if type_specific_candidates else 0
        for i, assigned_node_id in enumerate(chosen_nodes):
            node_detail = next((c for c in type_specific_candidates if c['node_id'] == assigned_node_id), {'distance_m': float('inf')})
            assigned_parcels_for_type.append({'day': day_val, 'sp_id': sp_id_str, 'parcel_id': f"{sp_id_str}_{parcel_idx_start + i + 1}", 'node_id': assigned_node_id, 'status': parcel_status, 'draw_prob': prob_fallback, 'distance_m': node_detail['distance_m']})
        parcel_idx_start += num_parcels_to_assign
        return assigned_parcels_for_type, parcel_idx_start

    probabilities = [c['logit_weight'] / total_logit_weight for c in type_specific_candidates]
    candidate_node_ids_for_type = [c['node_id'] for c in type_specific_candidates]
    node_details_map = {c['node_id']: {'prob': c['logit_weight'] / total_logit_weight, 'dist': c['distance_m']} for c in type_specific_candidates}
    
    chosen_node_ids = np.random.choice(candidate_node_ids_for_type, size=num_parcels_to_assign, replace=True, p=probabilities)

    for i, assigned_node_id in enumerate(chosen_node_ids):
        assigned_parcels_for_type.append({'day': day_val, 'sp_id': sp_id_str, 'parcel_id': f"{sp_id_str}_{parcel_idx_start + i + 1}", 'node_id': assigned_node_id, 'status': parcel_status, 'draw_prob': node_details_map[assigned_node_id]['prob'], 'distance_m': node_details_map[assigned_node_id]['dist']})
    parcel_idx_start += num_parcels_to_assign
    return assigned_parcels_for_type, parcel_idx_start

def simulate_parcel_dispatch_for_day(
    day_to_simulate_value, 
    deliveries_data_df, # Original deliveries
    pickups_data_df,    # Original pickups
    service_points_nodes_map,
    sp_to_net_node_map, graph_obj, node_socio_map, dist_cache, seed, iteration_num, total_iterations
):
    day_specific_seed = seed + int(day_to_simulate_value) if isinstance(day_to_simulate_value, (int, float)) else seed
    np.random.seed(day_specific_seed)
    all_allocations_list = []
    
    # Progress printing (optional)
    # if iteration_num % 20 == 0 or iteration_num == 1 or iteration_num == total_iterations:
    # print(f"  Simulating parcel dispatch for day: {day_to_simulate_value} ({iteration_num}/{total_iterations})")

    daily_deliveries_raw = deliveries_data_df[deliveries_data_df['day'] == day_to_simulate_value].copy()
    daily_pickups_raw = pd.DataFrame()
    if not pickups_data_df.empty and 'day' in pickups_data_df.columns:
        daily_pickups_raw = pickups_data_df[pickups_data_df['day'] == day_to_simulate_value].copy()

    if not daily_deliveries_raw.empty: daily_deliveries_raw['sp_id'] = daily_deliveries_raw['sp_id'].astype(str)
    if not daily_pickups_raw.empty: daily_pickups_raw['sp_id'] = daily_pickups_raw['sp_id'].astype(str)

    agg_deliveries = daily_deliveries_raw.groupby('sp_id')['parcels'].sum().reset_index().rename(columns={'parcels': 'parcels_from_delivery'}) if not daily_deliveries_raw.empty else pd.DataFrame(columns=['sp_id', 'parcels_from_delivery'])
    agg_pickups = daily_pickups_raw.groupby('sp_id')['parcels'].sum().reset_index().rename(columns={'parcels': 'parcels_from_pickup'}) if not daily_pickups_raw.empty and 'parcels' in daily_pickups_raw.columns else pd.DataFrame(columns=['sp_id', 'parcels_from_pickup'])
    
    if not agg_deliveries.empty and not agg_pickups.empty: daily_combined_demand = pd.merge(agg_deliveries, agg_pickups, on='sp_id', how='outer')
    elif not agg_deliveries.empty: daily_combined_demand = agg_deliveries
    elif not agg_pickups.empty: daily_combined_demand = agg_pickups
    else: return pd.DataFrame()
    for col in ['parcels_from_delivery', 'parcels_from_pickup']:
        if col not in daily_combined_demand.columns: daily_combined_demand[col] = 0
        else: daily_combined_demand[col] = daily_combined_demand[col].fillna(0).astype(int)

    if daily_combined_demand.empty: return pd.DataFrame()
    
    parcel_sp_idx_counter = defaultdict(int) # To create unique parcel_ids per SP

    for _, row in daily_combined_demand.iterrows():
        sp_id = str(row['sp_id'])
        parcels_direct_delivery = int(row['parcels_from_delivery'])
        parcels_direct_pickup = int(row['parcels_from_pickup'])
        
        if (parcels_direct_delivery + parcels_direct_pickup) == 0: continue
        sp_network_node = sp_to_net_node_map.get(sp_id)
        if sp_network_node is None: continue
        cluster_node_ids = service_points_nodes_map.get(sp_id, [])
        if not cluster_node_ids: continue
        
        base_candidate_info = []
        for node_id_int in cluster_node_ids: 
            node_id = int(node_id_int) 
            distance_m = float('inf')
            cache_key = tuple(sorted((sp_network_node, node_id)))
            if cache_key in dist_cache_ref: distance_m = dist_cache_ref[cache_key]
            else:
                try:
                    if graph_obj.has_node(sp_network_node) and graph_obj.has_node(node_id):
                        distance_m = nx.shortest_path_length(graph_obj, source=sp_network_node, target=node_id, weight='weight') if sp_network_node != node_id else 0.0
                    dist_cache_ref[cache_key] = distance_m
                except (nx.NetworkXNoPath, nx.NodeNotFound): dist_cache_ref[cache_key] = float('inf')
            norm_socio = node_socio_map.get(node_id, DEMAND_SIM_CONFIG['DEFAULT_SOCIO_SCORE_IF_MISSING'])
            base_candidate_info.append({'node_id': node_id, 'distance_m': distance_m, 'norm_socio_score': norm_socio})
        if not base_candidate_info: continue

        # 1. Assign explicit pickups (from pickups_df)
        if parcels_direct_pickup > 0:
            assigned_pickups, _ = _assign_parcels_of_type( # parcel_idx_start is managed by parcel_sp_idx_counter now
                num_parcels_to_assign=parcels_direct_pickup, parcel_status='pick up',
                base_candidate_info_list=base_candidate_info, sp_id_str=sp_id,
                day_val=day_to_simulate_value, parcel_idx_start=parcel_sp_idx_counter[sp_id], # Use counter
                config=DEMAND_SIM_CONFIG, sp_network_node=sp_network_node, dist_cache_ref=dist_cache_ref
            )
            all_allocations_list.extend(assigned_pickups)
            parcel_sp_idx_counter[sp_id] += len(assigned_pickups)


        # 2. Assign parcels from deliveries_df, determining status dynamically
        for _ in range(parcels_direct_delivery):
            # Select a node for this parcel using 'at home' logic as a proxy for initial assignment
            chosen_node_details = _select_node_for_delivery_parcel(base_candidate_info, DEMAND_SIM_CONFIG, parcel_status_for_weighting='at home')

            if chosen_node_details is None: # Should be rare if base_candidate_info exists
                # print(f"Warning: Could not select a node for a delivery parcel for SP {sp_id}. Skipping parcel.")
                continue

            assigned_node_id = chosen_node_details['node_id']
            distance_to_sp = chosen_node_details['distance_m'] # This is distance from SP to customer node

            # Determine final status based on dynamic probability
            pickup_prob = get_dynamic_pickup_probability(distance_to_sp)
            final_status = 'pick up' if np.random.rand() < pickup_prob else 'at home'
            
            parcel_sp_idx_counter[sp_id] += 1
            parcel_id = f"{sp_id}_{parcel_sp_idx_counter[sp_id]}"

            all_allocations_list.append({
                'day': day_to_simulate_value,
                'sp_id': sp_id,
                'parcel_id': parcel_id,
                'node_id': assigned_node_id,
                'status': final_status,
                'draw_prob': chosen_node_details.get('logit_weight',0) / sum(c.get('logit_weight',0) for c in base_candidate_info if c.get('logit_weight',0) > 0) if sum(c.get('logit_weight',0) for c in base_candidate_info if c.get('logit_weight',0) > 0) > 0 else 0, # Approx prob
                'distance_m': distance_to_sp,
                'origin_source': 'deliveries_data'
            })
            
    return pd.DataFrame(all_allocations_list)

# --- Run Demand Simulation ---
print("Step 3: Running demand simulation...")
all_days_allocations_dfs = []
# Simulate for a limited number of days for faster plotting, e.g., first 5 unique days
# Or all days: unique_days_to_simulate = deliveries_df['day'].unique()
unique_days_in_deliveries = deliveries_df['day'].unique()
unique_days_in_pickups = pickups_df['day'].unique() if not pickups_df.empty else np.array([])
unique_days_to_simulate = np.union1d(unique_days_in_deliveries, unique_days_in_pickups)

# For plotting, maybe simulate only a subset of days or one day to keep it fast
# unique_days_to_simulate = unique_days_to_simulate[:1] # Simulate only the first available day for speed

if not unique_days_to_simulate.any():
    print("No days to simulate based on deliveries or pickups data. Exiting.")
    exit()

print(f"Simulating for {len(unique_days_to_simulate)} unique day(s).")

for i, day_val in enumerate(unique_days_to_simulate):
    print(f"  Simulating day {day_val} ({i+1}/{len(unique_days_to_simulate)})...")
    daily_df = simulate_parcel_dispatch_for_day(
        day_val, deliveries_df, pickups_df, service_point_nodes_map,
        sp_to_network_node_map, G, node_to_norm_socio_map,
        distance_cache, RANDOM_SEED, i+1, len(unique_days_to_simulate)
    )
    if not daily_df.empty:
        all_days_allocations_dfs.append(daily_df)

full_period_allocations_df = pd.DataFrame()
if all_days_allocations_dfs:
    full_period_allocations_df = pd.concat(all_days_allocations_dfs, ignore_index=True)
    print(f"Total parcels allocated over the simulated period: {len(full_period_allocations_df)}")

    # --- Calculate Service Point Capacities based on Max Daily Pickups ---
    print("\nStep 3.5: Calculating Service Point Capacities...")
    if not full_period_allocations_df.empty and 'status' in full_period_allocations_df.columns:
        pickup_parcels_df = full_period_allocations_df[full_period_allocations_df['status'] == 'pick up'].copy()
        if not pickup_parcels_df.empty:
            # Ensure 'sp_id' and 'day' are present for grouping
            if 'sp_id' in pickup_parcels_df.columns and 'day' in pickup_parcels_df.columns:
                daily_pickups_per_sp = pickup_parcels_df.groupby(['sp_id', 'day']).size().reset_index(name='pickup_count')
                sp_capacities = daily_pickups_per_sp.groupby('sp_id')['pickup_count'].max().to_dict()
                
                # Add capacities to the service_points_df for easy access
                service_points_df['capacity'] = service_points_df['sp_id'].map(sp_capacities).fillna(0).astype(int)
                
                print("Service Point Capacities (Max Daily Pickups):")
                for sp_id_str, cap in sp_capacities.items():
                    print(f"  SP {sp_id_str}: {cap} parcels")
                # print("\nService Points DataFrame with capacities:")
                # print(service_points_df[['sp_id', 'sp_name', 'capacity']].head())
            else:
                print("Warning: 'sp_id' or 'day' column missing in pickup_parcels_df. Cannot calculate SP capacities accurately.")
                service_points_df['capacity'] = 0 # Default capacity
        else:
            print("No 'pick up' parcels found in simulation. Setting all SP capacities to 0.")
            service_points_df['capacity'] = 0 # Default capacity
    else:
        print("Warning: full_period_allocations_df is empty or 'status' column missing. Cannot calculate SP capacities.")
        service_points_df['capacity'] = 0 # Default capacity
    # --- End of Capacity Calculation ---

else:
    print("No parcel allocations generated from simulation. Cannot plot demand types or calculate capacities.")
    service_points_df['capacity'] = 0 # Default capacity if no simulation data

# --- Determine Node Demand Types for Plotting ---
node_demand_info_map = {} # Changed from node_demand_type_map to store more info
if not full_period_allocations_df.empty:
    print("Step 4: Determining node demand types...")
    node_status_counts = full_period_allocations_df.groupby('node_id')['status'].value_counts().unstack(fill_value=0)
    
    for node_id, row in node_status_counts.iterrows():
        num_pickup = row.get('pick up', 0)
        num_athome = row.get('at home', 0)
        total_parcels_for_node = num_pickup + num_athome

        if total_parcels_for_node == 0: # Should not happen if node_id is in node_status_counts
            continue

        if num_pickup > 0 and num_athome > 0:
            pickup_proportion = num_pickup / total_parcels_for_node
            node_demand_info_map[node_id] = {'type': 'mixed', 'pickup_proportion': pickup_proportion}
        elif num_pickup > 0:
            node_demand_info_map[node_id] = {'type': 'pick up', 'pickup_proportion': 1.0}
        elif num_athome > 0:
            node_demand_info_map[node_id] = {'type': 'at home', 'pickup_proportion': 0.0}
else:
    print("Skipping node demand type determination as no parcels were allocated.")


print("Script finished.")

# goign to need a couple of functions to proceed with simulated anealing

# reevaluate demand node status (at home, pick up) based on the open service points

# --- Simulated Annealing Functions ---

def find_closest_active_sp(customer_node_id, active_sp_network_nodes, G_graph, dist_cache_local):
    """
    Finds the closest active service point for a given customer node.
    Uses pre-calculated distances if available, otherwise calculates using the graph.
    Returns the closest service point node and the distance.
    """
    min_dist = float('inf')
    closest_sp_node = None

    for sp_node in active_sp_network_nodes:
        dist = dist_cache_local.get((customer_node_id, sp_node), float('inf'))
        if dist < min_dist:
            min_dist = dist
            closest_sp_node = sp_node

    # If no direct cache hit, calculate using the graph (as a fallback)
    if closest_sp_node is None or min_dist == float('inf'):
        try:
            if G_graph.has_node(customer_node_id) and G_graph.has_node(closest_sp_node):
                min_dist = nx.shortest_path_length(G_graph, source=customer_node_id, target=closest_sp_node, weight='weight')
            else:
                min_dist = float('inf')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            min_dist = float('inf')

    return closest_sp_node, min_dist

def reevaluate_parcels_for_active_sps(
    base_allocations_df, 
    active_sp_ids,       
    G_graph,
    sp_to_network_node_map_local, 
    distance_cache_local,
    config_local,
    service_point_capacities_map 
):
    """
    Re-evaluates parcel assignments and statuses based on a new set of active service points,
    considering service point capacities for 'pick up' parcels and tracking bounces.
    Returns the updated allocations DataFrame and a map of observed peak daily loads.
    """
    print(f"Re-evaluating parcel assignments for {len(active_sp_ids)} active SPs, considering capacities...")
    active_sp_ids_set = set(map(str, active_sp_ids)) 

    active_sp_network_nodes = []
    for sp_id_str in active_sp_ids_set:
        node = sp_to_network_node_map_local.get(sp_id_str)
        if node is not None:
            active_sp_network_nodes.append(node)
    
    temp_allocations_df = base_allocations_df.copy()
    # Initialize new columns for tracking
    temp_allocations_df['assigned_sp_id'] = None 
    temp_allocations_df['status_reevaluated'] = temp_allocations_df['status'] 
    temp_allocations_df['is_serviceable_reevaluated'] = True
    temp_allocations_df['distance_to_assigned_sp'] = float('inf')
    temp_allocations_df['intended_pickup_attempt_at_sp'] = None 
    temp_allocations_df['bounced_due_to_capacity'] = False

    # Track daily demand and bounces for accurate bounce rate calculation
    sp_daily_pickup_demand = defaultdict(lambda: defaultdict(int))  # sp_id -> day -> attempted_pickups
    sp_daily_pickup_bounced = defaultdict(lambda: defaultdict(int))  # sp_id -> day -> bounced_pickups
    sp_daily_pickup_served = defaultdict(lambda: defaultdict(int))   # sp_id -> day -> served_pickups

    for index, parcel_row in temp_allocations_df.iterrows():
        customer_node_id = parcel_row['node_id']
        origin_source = parcel_row.get('origin_source', 'deliveries_data') 
        parcel_day = parcel_row['day']
        
        closest_active_sp_node, dist_to_closest_active_sp = find_closest_active_sp(
            customer_node_id, active_sp_network_nodes, G_graph, distance_cache_local
        )
        
        potential_assigned_sp_id = None
        if closest_active_sp_node is not None:
            for spid, net_node in sp_to_network_node_map_local.items():
                if net_node == closest_active_sp_node and spid in active_sp_ids_set:
                    potential_assigned_sp_id = spid
                    break
        
        temp_allocations_df.loc[index, 'distance_to_assigned_sp'] = dist_to_closest_active_sp
        
        current_final_status = parcel_row['status'] 
        current_is_serviceable = True
        current_assigned_sp_for_parcel = None 
        bounced_this_parcel = False

        is_intended_pickup = False
        if origin_source == 'pickups_data':
            is_intended_pickup = True
        elif origin_source == 'deliveries_data':
            if potential_assigned_sp_id: 
                pickup_prob = get_dynamic_pickup_probability(dist_to_closest_active_sp)
                if np.random.rand() < pickup_prob:
                    is_intended_pickup = True

        if is_intended_pickup and potential_assigned_sp_id:
            temp_allocations_df.loc[index, 'intended_pickup_attempt_at_sp'] = potential_assigned_sp_id
            daily_capacity_for_sp = (service_point_capacities_map.get(potential_assigned_sp_id, 0) / DAYS_IN_YEAR) if service_point_capacities_map.get(potential_assigned_sp_id, 0) > 0 else 0
            sp_daily_pickup_demand[potential_assigned_sp_id][parcel_day] += 1
            current_daily_served = sp_daily_pickup_served[potential_assigned_sp_id][parcel_day]
            
            if current_daily_served < daily_capacity_for_sp:
                current_final_status = 'pick up'
                current_assigned_sp_for_parcel = potential_assigned_sp_id
                sp_daily_pickup_served[potential_assigned_sp_id][parcel_day] += 1
            else:
                bounced_this_parcel = True
                sp_daily_pickup_bounced[potential_assigned_sp_id][parcel_day] += 1
                if origin_source == 'pickups_data': 
                    current_is_serviceable = False
                    current_final_status = 'pick up'
                elif origin_source == 'deliveries_data':
                    current_final_status = 'at home'
                    # If it bounces from pickup and was a delivery, it's still 'at home'
                    # and should be assigned to the potential_assigned_sp_id for cost purposes
                    if potential_assigned_sp_id:
                         current_assigned_sp_for_parcel = potential_assigned_sp_id
        else: 
            current_final_status = 'at home'
            # Assign SP for 'at home' deliveries based on closest active SP
            if potential_assigned_sp_id:
                current_assigned_sp_for_parcel = potential_assigned_sp_id


        if origin_source not in ['pickups_data', 'deliveries_data']:
            current_is_serviceable = False 

        temp_allocations_df.loc[index, 'status_reevaluated'] = current_final_status
        temp_allocations_df.loc[index, 'is_serviceable_reevaluated'] = current_is_serviceable
        temp_allocations_df.loc[index, 'assigned_sp_id'] = current_assigned_sp_for_parcel # Ensure this is set
        temp_allocations_df.loc[index, 'bounced_due_to_capacity'] = bounced_this_parcel
    
    # Calculate bounce rates and observed loads
    sp_bounce_rate_stats = {}
    observed_max_daily_pickup_demand_in_solution = defaultdict(int)
    
    for sp_id in active_sp_ids_set:
        total_attempts = sum(sp_daily_pickup_demand[sp_id].values())
        total_bounced = sum(sp_daily_pickup_bounced[sp_id].values())
        total_served = sum(sp_daily_pickup_served[sp_id].values())
        
        # Calculate bounce rate
        bounce_rate = total_bounced / total_attempts if total_attempts > 0 else 0.0
        
        # Calculate peak daily demand
        peak_daily_demand = max(sp_daily_pickup_demand[sp_id].values()) if sp_daily_pickup_demand[sp_id] else 0
        observed_max_daily_pickup_demand_in_solution[sp_id] = peak_daily_demand
        
        # Store statistics
        sp_bounce_rate_stats[sp_id] = {
            'bounce_rate': bounce_rate,
            'total_attempts': total_attempts,
            'total_bounced': total_bounced,
            'total_served': total_served,
            'peak_daily_demand': peak_daily_demand,
            'daily_capacity': service_point_capacities_map.get(sp_id, 0) / DAYS_IN_YEAR
        }
        
        # Debug output for high bounce rates
        if bounce_rate > 0.05:  # 5% threshold for debug
            print(f"  SP {sp_id}: Bounce rate: {bounce_rate:.2%}, Attempts: {total_attempts}, "
                  f"Bounced: {total_bounced}, Peak daily demand: {peak_daily_demand}")
    
    # Ensure all active SPs are represented
    for sp_id_str_active in active_sp_ids_set:
        if sp_id_str_active not in observed_max_daily_pickup_demand_in_solution:
            observed_max_daily_pickup_demand_in_solution[sp_id_str_active] = 0
            sp_bounce_rate_stats[sp_id_str_active] = {
                'bounce_rate': 0.0, 'total_attempts': 0, 'total_bounced': 0, 
                'total_served': 0, 'peak_daily_demand': 0,
                'daily_capacity': service_point_capacities_map.get(sp_id_str_active, 0) / DAYS_IN_YEAR
            }
        
    # Calculate at-home delivery counts per SP for the current solution
    sp_at_home_delivery_counts = defaultdict(int)
    at_home_deliveries_for_costing = temp_allocations_df[
        (temp_allocations_df['status_reevaluated'] == 'at home') & \
        (temp_allocations_df['is_serviceable_reevaluated'] == True) & \
        (temp_allocations_df['assigned_sp_id'].notna())
    ]
    if not at_home_deliveries_for_costing.empty:
        counts_series = at_home_deliveries_for_costing.groupby('assigned_sp_id').size()
        for sp_id_count, count_val in counts_series.items():
            sp_at_home_delivery_counts[str(sp_id_count)] = count_val
        
    return temp_allocations_df, observed_max_daily_pickup_demand_in_solution, sp_bounce_rate_stats, sp_at_home_delivery_counts

def calculate_bounce_rate_kpis_and_penalty(reevaluated_allocations_df, active_sp_ids_in_solution, sp_bounce_rate_stats=None):
    """
    Calculates bounce rates and a penalty if KPIs are violated.
    Now uses the pre-calculated bounce rate statistics for more accurate calculation.
    """
    kpi_penalty = 0.0
    
    if sp_bounce_rate_stats:
        # Use pre-calculated statistics for more accurate bounce rate calculation
        total_attempts_city = sum(stats['total_attempts'] for stats in sp_bounce_rate_stats.values())
        total_bounced_city = sum(stats['total_bounced'] for stats in sp_bounce_rate_stats.values())
        
        city_wide_bounce_rate = total_bounced_city / total_attempts_city if total_attempts_city > 0 else 0.0
        
        if city_wide_bounce_rate > MAX_CITY_WIDE_BOUNCE_RATE:
            kpi_penalty += KPI_VIOLATION_PENALTY_BASE + (city_wide_bounce_rate - MAX_CITY_WIDE_BOUNCE_RATE) * 10 * KPI_VIOLATION_PENALTY_BASE
            print(f"Warning: City-wide bounce rate KPI violated: {city_wide_bounce_rate:.2%} (Target: {MAX_CITY_WIDE_BOUNCE_RATE:.2%})")

        # Check individual SP bounce rates
        for sp_id_str in active_sp_ids_in_solution:
            sp_stats = sp_bounce_rate_stats.get(sp_id_str, {})
            sp_bounce_rate = sp_stats.get('bounce_rate', 0.0)
            
            if sp_bounce_rate > MAX_SP_BOUNCE_RATE:
                kpi_penalty += KPI_VIOLATION_PENALTY_BASE / 10 + (sp_bounce_rate - MAX_SP_BOUNCE_RATE) * KPI_VIOLATION_PENALTY_BASE
                print(f"Warning: SP {sp_id_str} bounce rate KPI violated: {sp_bounce_rate:.2%} "
                      f"(Target: {MAX_SP_BOUNCE_RATE:.2%}), Attempts: {sp_stats.get('total_attempts', 0)}")
    else:
        # Fallback to original calculation method
        pickup_attempts_df = reevaluated_allocations_df[
            reevaluated_allocations_df['intended_pickup_attempt_at_sp'].notna()
        ]
        
        total_pickup_attempts = len(pickup_attempts_df)
        total_bounced_pickups = pickup_attempts_df[pickup_attempts_df['bounced_due_to_capacity'] == True].shape[0]
        
        # Calculate city-wide bounce rate
        city_wide_bounce_rate = total_bounced_pickups / total_pickup_attempts if total_pickup_attempts > 0 else 0.0
        
        if city_wide_bounce_rate > MAX_CITY_WIDE_BOUNCE_RATE:
            kpi_penalty += KPI_VIOLATION_PENALTY_BASE + (city_wide_bounce_rate - MAX_CITY_WIDE_BOUNCE_RATE) * 10 * KPI_VIOLATION_PENALTY_BASE
            print(f"Warning: City-wide bounce rate KPI violated (fallback): {city_wide_bounce_rate:.2%} (Target: {MAX_CITY_WIDE_BOUNCE_RATE:.2%})")
        
        # Individual SP bounce rates
        for sp_id_str in active_sp_ids_in_solution:
            sp_attempts = pickup_attempts_df[pickup_attempts_df['assigned_sp_id'] == sp_id_str]
            total_attempts_sp = len(sp_attempts)
            total_bounced_sp = sp_attempts[sp_attempts['bounced_due_to_capacity'] == True].shape[0]
            
            sp_bounce_rate = total_bounced_sp / total_attempts_sp if total_attempts_sp > 0 else 0.0
            
            if sp_bounce_rate > MAX_SP_BOUNCE_RATE:
                kpi_penalty += KPI_VIOLATION_PENALTY_BASE / 10 + (sp_bounce_rate - MAX_SP_BOUNCE_RATE) * KPI_VIOLATION_PENALTY_BASE
                print(f"Warning: SP {sp_id_str} bounce rate KPI violated (fallback): {sp_bounce_rate:.2%} "
                      f"(Target: {MAX_SP_BOUNCE_RATE:.2%}), Attempts: {total_attempts_sp}")
    
    return kpi_penalty

def calculate_total_cost(allocations_df, active_sp_ids_in_solution, proposed_capacities_for_solution, service_points_df_all_sps, sp_bounce_rate_stats, sp_at_home_delivery_counts, sp_avg_delivery_distance_map_local):
    total_yearly_cost = 0.0
    if allocations_df.empty and not sp_at_home_delivery_counts:
        if not active_sp_ids_in_solution: return 0.0 # No SPs, no parcels, no cost
    variable_cost_sim_period = 0.0
    for sp_id_str_active in active_sp_ids_in_solution:
        sp_id_str = str(sp_id_str_active); num_at_home_deliveries = sp_at_home_delivery_counts.get(sp_id_str, 0); avg_dist_km = sp_avg_delivery_distance_map_local.get(sp_id_str, 0.0)
        if avg_dist_km == float('inf'):
            if num_at_home_deliveries > 0: print(f"Warning: SP {sp_id_str} has inf avg_dist and {num_at_home_deliveries} at-home deliveries. Penalizing."); variable_cost_sim_period += num_at_home_deliveries * PENALTY_UNSERVICEABLE
        elif num_at_home_deliveries > 0: variable_cost_sim_period += num_at_home_deliveries * avg_dist_km * COST_PER_KM_DELIVERY
    if 'is_serviceable_reevaluated' in allocations_df.columns:
        problematic_sp_ids_for_avg_dist = {sp_id for sp_id, dist in sp_avg_delivery_distance_map_local.items() if dist == float('inf')}
        unserviceable_parcels_count = allocations_df[ (allocations_df['is_serviceable_reevaluated'] == False) & ~( (allocations_df['status_reevaluated'] == 'at home') & (allocations_df['assigned_sp_id'].astype(str).isin(problematic_sp_ids_for_avg_dist)) )].shape[0]
        variable_cost_sim_period += unserviceable_parcels_count * PENALTY_UNSERVICEABLE
    yearly_variable_cost = 0.0
    num_simulated_days = allocations_df['day'].nunique() if not allocations_df.empty and 'day' in allocations_df.columns else 0
    if num_simulated_days > 0: yearly_variable_cost = (variable_cost_sim_period / num_simulated_days) * DAYS_IN_YEAR
    elif variable_cost_sim_period > 0: print(f"Warning: num_simulated_days is {num_simulated_days}. Annualizing based on 1 day if variable_cost_sim_period > 0."); yearly_variable_cost = variable_cost_sim_period * DAYS_IN_YEAR
    total_yearly_cost += yearly_variable_cost
    total_yearly_cost += len(active_sp_ids_in_solution) * FIXED_COST_SP_YEARLY
    for sp_id_str_active_storage in active_sp_ids_in_solution:
        yearly_capacity = proposed_capacities_for_solution.get(str(sp_id_str_active_storage))
        if yearly_capacity is not None and pd.notna(yearly_capacity) and yearly_capacity > 0: total_yearly_cost += yearly_capacity * STORAGE_COST_PER_CAPACITY_UNIT_YEARLY
    total_yearly_cost += calculate_bounce_rate_kpis_and_penalty(allocations_df, active_sp_ids_in_solution, sp_bounce_rate_stats)
    return total_yearly_cost

def generate_initial_solution(all_sp_ids_list, initial_sp_capacities_map):
    if not all_sp_ids_list: return set(), {}
    num_total_sps = len(all_sp_ids_list)
    min_sps = max(1, int(num_total_sps * MIN_INITIAL_ACTIVE_SP_PERCENT)); max_sps = max(min_sps, int(num_total_sps * MAX_INITIAL_ACTIVE_SP_PERCENT))
    if max_sps > num_total_sps: max_sps = num_total_sps
    if min_sps > num_total_sps: min_sps = num_total_sps
    num_active_sps = random.randint(min_sps, max_sps) if min_sps <= max_sps else min_sps
    active_sps_list = random.sample(all_sp_ids_list, num_active_sps)
    active_sps_set = set(map(str, active_sps_list))
    proposed_capacities_map = {sp_id_str: initial_sp_capacities_map.get(sp_id_str, 0) for sp_id_str in active_sps_set}
    print(f"Generated Initial Solution: {len(active_sps_set)} active SPs.")
    return active_sps_set, proposed_capacities_map

def generate_neighbor_solution(current_active_sps_set, current_capacities_map, all_sp_ids_list, initial_sp_capacities_map, G_graph):
    neighbor_active_sps_set = set(current_active_sps_set); neighbor_capacities_map = dict(current_capacities_map)
    mutation_type = random.choice(['toggle_sp', 'change_capacity', 'change_capacity']) # Bias towards capacity change
    if mutation_type == 'toggle_sp' and all_sp_ids_list:
        sp_to_toggle = random.choice(all_sp_ids_list); sp_to_toggle_str = str(sp_to_toggle)
        if sp_to_toggle_str in neighbor_active_sps_set:
            if len(neighbor_active_sps_set) > 1: neighbor_active_sps_set.remove(sp_to_toggle_str); neighbor_capacities_map.pop(sp_to_toggle_str, None)
        else:
            neighbor_active_sps_set.add(sp_to_toggle_str); neighbor_capacities_map[sp_to_toggle_str] = initial_sp_capacities_map.get(sp_to_toggle_str, 0)
    elif mutation_type == 'change_capacity' and neighbor_active_sps_set:
        sp_to_change_cap_str = random.choice(list(neighbor_active_sps_set))
        max_cap_for_sp = initial_sp_capacities_map.get(sp_to_change_cap_str, 0); current_cap = neighbor_capacities_map.get(sp_to_change_cap_str, 0)
        if max_cap_for_sp > 0:
            change_factor = random.uniform(0.75, 1.25) # Change by -25% to +25%
            new_capacity = current_cap * change_factor
            # More aggressive changes:
            # if random.random() < 0.3: # Small nudge
            #     change_amount = max(1, int(max_cap_for_sp * random.uniform(0.05, 0.15)))
            #     new_capacity = current_cap + random.choice([-1,1]) * change_amount
            # else: # Larger jump
            #     new_capacity = random.uniform(max_cap_for_sp * 0.1, max_cap_for_sp) # 10% to 100% of max

            new_capacity = max(0, int(new_capacity)); new_capacity = min(new_capacity, int(max_cap_for_sp))
            neighbor_capacities_map[sp_to_change_cap_str] = new_capacity
    for sp_id_active in neighbor_active_sps_set: # Ensure all active have capacity
        if sp_id_active not in neighbor_capacities_map: neighbor_capacities_map[sp_id_active] = initial_sp_capacities_map.get(sp_id_active, 0)
    return neighbor_active_sps_set, neighbor_capacities_map

def run_simulated_annealing(base_allocations_df, all_sp_ids_list, initial_sp_capacities_map_yearly, G_graph, sp_to_network_node_map_local, distance_cache_local, config_local, service_points_df_all_sps_indexed, sp_avg_delivery_distance_map_main):
    temperature = INITIAL_TEMPERATURE
    current_active_sps, current_proposed_capacities = generate_initial_solution(all_sp_ids_list, initial_sp_capacities_map_yearly)
    current_allocations_df, current_observed_loads, current_bounce_stats, current_at_home_counts = reevaluate_parcels_for_active_sps(base_allocations_df, current_active_sps, G_graph, sp_to_network_node_map_local, distance_cache_local, config_local, current_proposed_capacities)
    current_cost = calculate_total_cost(current_allocations_df, current_active_sps, current_proposed_capacities, service_points_df_all_sps_indexed, current_bounce_stats, current_at_home_counts, sp_avg_delivery_distance_map_main)
    best_solution_sps = list(current_active_sps); best_solution_capacities = dict(current_proposed_capacities); best_cost = current_cost
    best_allocations_df = current_allocations_df.copy() if current_allocations_df is not None else pd.DataFrame()
    print(f"Initial Solution: Active SPs: {len(current_active_sps)}, Cost: {current_cost:.2f}")
    total_iterations = 0; history = []
    while temperature > MIN_TEMPERATURE and total_iterations < MAX_TOTAL_ITERATIONS:
        print(f"\nTemperature: {temperature:.2f} (Iter {total_iterations}/{MAX_TOTAL_ITERATIONS})")
        for i_iter_temp in range(MAX_ITERATIONS_PER_TEMP):
            total_iterations += 1
            if total_iterations > MAX_TOTAL_ITERATIONS: break
            neighbor_active_sps, neighbor_proposed_capacities = generate_neighbor_solution(current_active_sps, current_proposed_capacities, all_sp_ids_list, initial_sp_capacities_map_yearly, G_graph)
            neighbor_allocations_df, neighbor_observed_loads, neighbor_bounce_stats, neighbor_at_home_counts = reevaluate_parcels_for_active_sps(base_allocations_df, neighbor_active_sps, G_graph, sp_to_network_node_map_local, distance_cache_local, config_local, neighbor_proposed_capacities)
            neighbor_cost = calculate_total_cost(neighbor_allocations_df, neighbor_active_sps, neighbor_proposed_capacities, service_points_df_all_sps_indexed, neighbor_bounce_stats, neighbor_at_home_counts, sp_avg_delivery_distance_map_main)
            cost_delta = neighbor_cost - current_cost
            if cost_delta < 0:
                current_active_sps = list(neighbor_active_sps); current_proposed_capacities = dict(neighbor_proposed_capacities); current_cost = neighbor_cost
                if current_cost < best_cost:
                    best_solution_sps = list(current_active_sps); best_solution_capacities = dict(current_proposed_capacities); best_cost = current_cost
                    best_allocations_df = neighbor_allocations_df.copy() if neighbor_allocations_df is not None else pd.DataFrame()
                    print(f"  Iter {total_iterations}: New best solution! Cost: {best_cost:.2f}, SPs: {len(best_solution_sps)}")
            else:
                acceptance_probability = np.exp(-cost_delta / temperature)
                if random.random() < acceptance_probability:
                    current_active_sps = list(neighbor_active_sps); current_proposed_capacities = dict(neighbor_proposed_capacities); current_cost = neighbor_cost
                    print(f"  Iter {total_iterations}: Accepted worse solution. Cost: {current_cost:.2f}, Prob: {acceptance_probability:.4f}")
            history.append({'iteration': total_iterations, 'cost': current_cost, 'best_cost': best_cost, 'temp': temperature})
            if i_iter_temp % (MAX_ITERATIONS_PER_TEMP // 5 if MAX_ITERATIONS_PER_TEMP >=5 else 1) == 0 : print(f"    SubIter {i_iter_temp}: Current Cost: {current_cost:.2f}, Best Cost: {best_cost:.2f}")
        temperature *= COOLING_RATE
    print("\nSimulated Annealing Finished."); print(f"Best solution found: Active SPs: {best_solution_sps}"); print(f"Best solution capacities: {best_solution_capacities}"); print(f"Best cost: {best_cost:.2f}")
    return best_solution_sps, best_solution_capacities, best_cost, best_allocations_df, pd.DataFrame(history)

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Step 1: Loading data and initial setup...")
    np.random.seed(RANDOM_SEED)

    # --- Load Base Data ---
    try:
        data = load_maastricht_data()
        nodes_df = data["nodes"].copy()
        edges_df = data["edges"].copy()
        cbs_df = data["cbs"].copy()
        deliveries_df = data["deliveries"].copy()
        pickups_df = data.get("pickups", pd.DataFrame()).copy() # Handle missing 'pickups'
        service_points_df = data["service_points"].copy()

        deliveries_df['parcels'] = deliveries_df['parcels'].fillna(0).astype(int)
        if not pickups_df.empty:
            if 'parcels' in pickups_df.columns: pickups_df['parcels'] = pickups_df['parcels'].fillna(0).astype(int)
            elif 'pickups' in pickups_df.columns: pickups_df.rename(columns={'pickups': 'parcels'}, inplace=True); pickups_df['parcels'] = pickups_df['parcels'].fillna(0).astype(int)
            else: pickups_df['parcels'] = 0
        else: pickups_df = pd.DataFrame(columns=['day', 'sp_id', 'parcels'])
    except ImportError: print("ERROR: Could not import 'load_maastricht_data'. Ensure imp_data.py is accessible."); exit()
    except Exception as e: print(f"ERROR: Failed to load data: {e}"); exit()

    # --- Load Pre-calculated Maps ---
    try:
        with open(SERVICE_POINT_COVERAGE_FILE, 'r') as f: service_point_nodes_map = json.load(f).get('service_point_nodes', {})
        service_point_nodes_map = {str(k): [int(n) for n in v] for k, v in service_point_nodes_map.items()}
        with open(SP_TO_NETWORK_NODE_FILE, 'r') as f: sp_to_network_node_map = json.load(f)
        sp_to_network_node_map = {str(k): int(v) for k, v in sp_to_network_node_map.items()}
    except Exception as e: print(f"ERROR loading map files: {e}"); exit()

    # --- Create Graph ---
    G = nx.Graph()
    for _, node_row in nodes_df.iterrows(): G.add_node(int(node_row['node_id']), pos=(float(node_row['x_rd']), float(node_row['y_rd'])))
    for _, edge_row in edges_df.iterrows(): G.add_edge(int(edge_row['from_node']), int(edge_row['to_node']), weight=float(edge_row['length_m']))
    print(f"Graph G created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- Socioeconomic Score Calculation ---
    print("Step 2: Preprocessing node data for socioeconomic scores...")
    nodes_df['cbs_square'] = nodes_df['cbs_square'].astype(str); cbs_df['cbs_square'] = cbs_df['cbs_square'].astype(str)
    SOCIO_COL_INCOME = 'median_income_k€'; SOCIO_COL_HOME_VALUE = 'avg_home_value_k€'
    for col_name in [SOCIO_COL_INCOME, SOCIO_COL_HOME_VALUE]:
        if col_name not in cbs_df.columns: print(f"Warning: CBS data missing '{col_name}'. Filling with 0."); cbs_df[col_name] = 0
    nodes_ext_df = pd.merge(nodes_df[['node_id', 'cbs_square']], cbs_df[['cbs_square', SOCIO_COL_INCOME, SOCIO_COL_HOME_VALUE]], on='cbs_square', how='left')
    for col in [SOCIO_COL_INCOME, SOCIO_COL_HOME_VALUE]: nodes_ext_df[col].fillna(nodes_ext_df[col].median() if not pd.isna(nodes_ext_df[col].median()) else 0, inplace=True)
    nodes_ext_df['raw_socio_score'] = nodes_ext_df[SOCIO_COL_INCOME] + nodes_ext_df[SOCIO_COL_HOME_VALUE]
    scaler = MinMaxScaler()
    nodes_ext_df['normalized_socio_score'] = scaler.fit_transform(nodes_ext_df[['raw_socio_score']]) if not nodes_ext_df['raw_socio_score'].empty and nodes_ext_df['raw_socio_score'].nunique() > 1 else 0.5
    node_to_norm_socio_map = nodes_ext_df.set_index('node_id')['normalized_socio_score'].to_dict()
    print("Socioeconomic factors calculated.")

    # --- Run Demand Simulation ---
    print("Step 3: Running demand simulation...")
    all_days_allocations_dfs = []
    unique_days_to_simulate = np.union1d(deliveries_df['day'].unique(), pickups_df['day'].unique() if not pickups_df.empty else np.array([]))
    if not unique_days_to_simulate.any(): print("No days to simulate. Exiting."); exit()
    print(f"Simulating for {len(unique_days_to_simulate)} unique day(s).")
    for i, day_val in enumerate(unique_days_to_simulate):
        # print(f"  Simulating day {day_val} ({i+1}/{len(unique_days_to_simulate)})...") # Verbose
        daily_df = simulate_parcel_dispatch_for_day(day_val, deliveries_df, pickups_df, service_point_nodes_map, sp_to_network_node_map, G, node_to_norm_socio_map, distance_cache, RANDOM_SEED, i+1, len(unique_days_to_simulate))
        if not daily_df.empty: all_days_allocations_dfs.append(daily_df)
    
    full_period_allocations_df = pd.DataFrame()
    if all_days_allocations_dfs:
        full_period_allocations_df = pd.concat(all_days_allocations_dfs, ignore_index=True)
        print(f"Total parcels allocated over the simulated period: {len(full_period_allocations_df)}")
        if 'origin_source' not in full_period_allocations_df.columns: # Ensure origin_source exists
            print("Warning: 'origin_source' was not set during demand simulation. This is unexpected.")
            # Attempt to infer or set a default, though this indicates an issue in simulate_parcel_dispatch_for_day
            # For now, we'll let the SA setup handle it if still missing.
    else:
        print("No parcel allocations generated from simulation. SA cannot run effectively.")
        # service_points_df['capacity'] = 0 # Handled below

    # --- Calculate Service Point Capacities (Daily) ---
    print("\nStep 3.5: Calculating Service Point Capacities (Max Daily Pickups)...")
    service_points_df['capacity'] = 0 # Initialize
    if not full_period_allocations_df.empty and 'status' in full_period_allocations_df.columns and 'sp_id' in full_period_allocations_df.columns and 'day' in full_period_allocations_df.columns:
        pickup_parcels_df = full_period_allocations_df[full_period_allocations_df['status'] == 'pick up'].copy()
        if not pickup_parcels_df.empty:
            daily_pickups_per_sp = pickup_parcels_df.groupby(['sp_id', 'day']).size().reset_index(name='pickup_count')
            sp_daily_capacities = daily_pickups_per_sp.groupby('sp_id')['pickup_count'].max().to_dict()
            service_points_df['capacity'] = service_points_df['sp_id'].astype(str).map(sp_daily_capacities).fillna(0).astype(int)
            print("Service Point Capacities (Max Daily Pickups):")
            for sp_id_str, cap in sp_capacities.items():
                print(f"  SP {sp_id_str}: {cap} parcels")
            # print("\nService Points DataFrame with capacities:")
            # print(service_points_df[['sp_id', 'sp_name', 'capacity']].head())
        else: print("No 'pick up' parcels found in simulation. Setting all SP capacities to 0.")
    else: print("Warning: Cannot calculate SP capacities due to missing data in full_period_allocations_df.")
    
    # --- Determine Node Demand Types for Plotting (Optional, can be removed if not used) ---
    # ... (This section can be kept or removed if plotting is not the focus for SA)

    # --- Simulated Annealing Execution ---
    if not full_period_allocations_df.empty:
        print("DEBUG: full_period_allocations_df is NOT empty. Proceeding to SA setup.")
        
        all_sp_ids = service_points_df['sp_id'].astype(str).unique().tolist()
        if not all_sp_ids: print("ERROR: No service point IDs found. Cannot run SA."); exit()

        if 'sp_id' in service_points_df.columns and service_points_df.index.name != 'sp_id': service_points_df_indexed = service_points_df.set_index('sp_id')
        elif service_points_df.index.name == 'sp_id': service_points_df_indexed = service_points_df.copy()
        else: print("ERROR: service_points_df cannot be indexed by 'sp_id'. Cannot proceed with SA."); exit()

        # Convert daily capacities from service_points_df to yearly for SA
        # initial_sp_capacities_map_yearly will be the basis for SA's capacity decisions
        initial_sp_capacities_map_yearly = (service_points_df_indexed['capacity'] * DAYS_IN_YEAR).to_dict()
        # Ensure all sp_ids from all_sp_ids list are in this map, even if capacity is 0
        for sp_id_ensure in all_sp_ids:
            if str(sp_id_ensure) not in initial_sp_capacities_map_yearly:
                 initial_sp_capacities_map_yearly[str(sp_id_ensure)] = 0 # Default to 0 yearly capacity if not in service_points_df_indexed


        if 'origin_source' not in full_period_allocations_df.columns:
            print("Warning: 'origin_source' column missing in full_period_allocations_df. Defaulting to 'deliveries_data'. This should have been set during demand simulation.");
            full_period_allocations_df['origin_source'] = 'deliveries_data' # Fallback

        print("\n--- Pre-calculating average delivery distances per SP ---")
        sp_avg_delivery_distance_map = {}
        # Ensure G, distance_cache, sp_to_network_node_map, service_point_nodes_map are defined and populated
        if 'G' not in globals() or 'distance_cache' not in globals() or 'sp_to_network_node_map' not in globals() or 'service_point_nodes_map' not in globals():
            print("ERROR: Missing necessary data structures for average distance calculation. SA cannot proceed."); exit()
        else:
            for sp_id_str_key, customer_node_ids_for_sp in service_point_nodes_map.items():
                sp_id_str_avg = str(sp_id_str_key)
                if not customer_node_ids_for_sp: sp_avg_delivery_distance_map[sp_id_str_avg] = 0.0; continue
                sp_network_node_avg = sp_to_network_node_map.get(sp_id_str_avg)
                if sp_network_node_avg is None: sp_avg_delivery_distance_map[sp_id_str_avg] = float('inf'); print(f"Warning: SP {sp_id_str_avg} not in sp_to_network_node_map for avg dist."); continue
                total_distance_m_for_sp = 0; count_valid_distances = 0
                for cust_node_id_val in customer_node_ids_for_sp:
                    cust_node_id = int(cust_node_id_val); dist_key = tuple(sorted((sp_network_node_avg, cust_node_id))); distance_m = distance_cache.get(dist_key)
                    if distance_m is None:
                        try:
                            if G.has_node(sp_network_node_avg) and G.has_node(cust_node_id): distance_m = 0.0 if sp_network_node_avg == cust_node_id else nx.shortest_path_length(G, source=sp_network_node_avg, target=cust_node_id, weight='weight')
                            else: distance_m = float('inf')
                        except (nx.NetworkXNoPath, nx.NodeNotFound): distance_m = float('inf')
                        distance_cache[dist_key] = distance_m
                    if distance_m != float('inf'): total_distance_m_for_sp += distance_m; count_valid_distances += 1
                if count_valid_distances > 0: sp_avg_delivery_distance_map[sp_id_str_avg] = (total_distance_m_for_sp / count_valid_distances) / 1000.0
                else: sp_avg_delivery_distance_map[sp_id_str_avg] = float('inf'); print(f"Warning: SP {sp_id_str_avg} had no valid paths for avg dist.")
            print(f"Calculated average delivery distances for {len(sp_avg_delivery_distance_map)} SPs.")


        print("\n--- Starting Simulated Annealing ---")
        best_sps, best_caps, best_c, best_alloc_df, cost_history_df = run_simulated_annealing(
            full_period_allocations_df,
            all_sp_ids,
            initial_sp_capacities_map_yearly, # Pass YEARLY capacities
            G,
            sp_to_network_node_map, # Global map
            distance_cache,         # Global cache
            DEMAND_SIM_CONFIG,
            service_points_df_indexed,
            sp_avg_delivery_distance_map
        )
        
        print("\n--- SA Results ---")
        print(f"Best SPs ({len(best_sps)}): {best_sps}")
        print("Best Yearly Capacities:")
        for sp_res, cap_res in best_caps.items(): print(f"  SP {sp_res}: {cap_res}")
        print(f"Best Cost: {best_c:.2f}")
        
        if not cost_history_df.empty:
            plt.figure(figsize=(12, 6))
            plt.plot(cost_history_df['iteration'], cost_history_df['cost'], label='Current Cost', alpha=0.7)
            plt.plot(cost_history_df['iteration'], cost_history_df['best_cost'], label='Best Cost', linestyle='--', color='red')
            plt.xlabel("Iteration"); plt.ylabel("Total Cost"); plt.title("Simulated Annealing Cost Progression"); plt.legend(); plt.grid(True); plt.tight_layout()
            plt.savefig("sa_cost_progression.png"); print("Saved SA cost progression plot to sa_cost_progression.png"); #plt.show()
        else: print("Cost history is empty, cannot plot progression.")

    else:
        print("Full period allocations DataFrame is empty. Cannot run Simulated Annealing.")

    print("\nScript execution finished.")

def create_daily_sp_df(sim_allocations_df):
    """
    Transforms the full simulation allocations DataFrame into the daily_sp format.

    Args:
        sim_allocations_df (pd.DataFrame): DataFrame containing detailed parcel allocations
                                           with columns 'sp_id', 'day', 'status', 'distance_m'.

    Returns:
        pd.DataFrame: DataFrame with columns ['LocationID', 'day', 'pred_pickup', 'pred_home', 'distance'].
                      'distance' is the average home delivery distance in meters for the SP.
    """
    if sim_allocations_df.empty or not all(col in sim_allocations_df.columns for col in ['sp_id', 'day', 'status', 'distance_m']):
        print("Warning: Input DataFrame is empty or missing required columns. Returning empty daily_sp DataFrame.")
        return pd.DataFrame(columns=['LocationID', 'day', 'pred_pickup', 'pred_home', 'distance'])

    # Calculate pred_pickup and pred_home per sp_id, day
    # Ensure 'status' is string to avoid issues with mixed types if any
    sim_allocations_df['status'] = sim_allocations_df['status'].astype(str)
    daily_counts = sim_allocations_df.groupby(['sp_id', 'day', 'status']).size().unstack(fill_value=0)
    
    if 'pick up' in daily_counts.columns:
        daily_counts.rename(columns={'pick up': 'pred_pickup'}, inplace=True)
    else:
        daily_counts['pred_pickup'] = 0
        
    if 'at home' in daily_counts.columns:
        daily_counts.rename(columns={'at home': 'pred_home'}, inplace=True)
    else:
        daily_counts['pred_home'] = 0
        
    daily_sp_intermediate = daily_counts[['pred_pickup', 'pred_home']].reset_index()

    if daily_sp_intermediate.empty: # No relevant 'pick up' or 'at home' data
        print("Warning: No 'pick up' or 'at home' data found after grouping. Returning empty daily_sp DataFrame.")
        return pd.DataFrame(columns=['LocationID', 'day', 'pred_pickup', 'pred_home', 'distance'])

    # Calculate characteristic average 'distance_m' for 'at home' parcels per sp_id
    at_home_parcels = sim_allocations_df[sim_allocations_df['status'] == 'at home']
    
    if not at_home_parcels.empty:
        sp_avg_home_distance = at_home_parcels.groupby('sp_id')['distance_m'].mean().reset_index()
        sp_avg_home_distance.rename(columns={'distance_m': 'distance'}, inplace=True)
    else:
        # Create an empty df with 'sp_id' and 'distance' columns, matching sp_id type
        sp_id_dtype = sim_allocations_df['sp_id'].dtype if 'sp_id' in sim_allocations_df else object
        sp_avg_home_distance = pd.DataFrame({'sp_id': pd.Series(dtype=sp_id_dtype), 'distance': pd.Series(dtype=float)})

    # Merge daily counts with the average SP home delivery distance
    daily_sp_df = pd.merge(daily_sp_intermediate, sp_avg_home_distance, on='sp_id', how='left')
    
    daily_sp_df['distance'].fillna(0, inplace=True) # For SPs with no 'at home' deliveries

    daily_sp_df.rename(columns={'sp_id': 'LocationID'}, inplace=True)

    final_columns = ['LocationID', 'day', 'pred_pickup', 'pred_home', 'distance']
    daily_sp_df = daily_sp_df[final_columns]

    return daily_sp_df