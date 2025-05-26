# --- Notebook Setup ---
import pandas as pd
import numpy as np
import networkx as nx
import json
from sklearn.preprocessing import MinMaxScaler
import pyarrow  # For to_parquet

# --- 0. Configuration ---
SP_TO_NETWORK_NODE_FILE = 'sp_to_network_node.json'
SERVICE_POINT_COVERAGE_FILE = 'service_point_coverage.json'
OUTPUT_PARQUET_FILE = 'all_days_parcel_allocations.parquet'  # Changed filename
RANDOM_SEED = 42  # For reproducibility

# --- 1. Load Data & Initial Setup ---
print("Step 1: Loading data and initial setup...")
np.random.seed(RANDOM_SEED)

try:
    from imp_data import load_maastricht_data  # Assuming imp_data.py is accessible
    data = load_maastricht_data()
    nodes_df = data["nodes"].copy()
    edges_df = data["edges"].copy()
    cbs_df = data["cbs"].copy()
    deliveries_df = data["deliveries"].copy()
    deliveries_df.rename(columns={'deliveries': 'parcels'}, inplace=True)

    print("\n[DEBUG] Deliveries data loaded:")
    print(f"  • Rows: {len(deliveries_df)}")
    print(f"  • Columns: {list(deliveries_df.columns)}")
    unique_days_in_data = deliveries_df['day'].unique()
    print(f"  • Unique days in data (count): {len(unique_days_in_data)}")
    print(f"  • Unique days (up to 5 shown): {unique_days_in_data[:5]}")
    print(f"  • Service points in deliveries: {deliveries_df['sp_id'].nunique()}")
except ImportError:
    print("ERROR: Could not import 'load_maastricht_data' from 'imp_data'. Ensure imp_data.py is in the correct path.")
    exit()
except Exception as e:
    print(f"ERROR: Failed to load data using imp_data.py: {e}")
    exit()

try:
    with open(SERVICE_POINT_COVERAGE_FILE, 'r') as f:
        coverage_data = json.load(f)
    service_point_nodes_map = coverage_data.get('service_point_nodes')
    if service_point_nodes_map is None:
        raise KeyError("'service_point_nodes' key not found in coverage data.")
    for sp_id_key in service_point_nodes_map:
        service_point_nodes_map[sp_id_key] = [int(n) for n in service_point_nodes_map[sp_id_key]]
    print(f"Loaded service_point_nodes_map for {len(service_point_nodes_map)} service points.")
except FileNotFoundError:
    print(f"ERROR: File not found: {SERVICE_POINT_COVERAGE_FILE}")
    exit()
except json.JSONDecodeError:
    print(f"ERROR: Could not decode JSON from {SERVICE_POINT_COVERAGE_FILE}")
    exit()
except KeyError as e:
    print(f"ERROR: Missing key in {SERVICE_POINT_COVERAGE_FILE}: {e}")
    exit()

try:
    with open(SP_TO_NETWORK_NODE_FILE, 'r') as f:
        sp_to_network_node_map = json.load(f)
    sp_to_network_node_map = {str(k): int(v) for k, v in sp_to_network_node_map.items()}
    print(f"Loaded sp_to_network_node_map for {len(sp_to_network_node_map)} service points.")
except FileNotFoundError:
    print(f"ERROR: File not found: {SP_TO_NETWORK_NODE_FILE}")
    exit()
except json.JSONDecodeError:
    print(f"ERROR: Could not decode JSON from {SP_TO_NETWORK_NODE_FILE}")
    exit()

G = nx.Graph()
for _, node_row in nodes_df.iterrows():
    G.add_node(int(node_row['node_id']))
for _, edge_row in edges_df.iterrows():
    G.add_edge(int(edge_row['from_node']), int(edge_row['to_node']), weight=float(edge_row['length_m']))
print(f"Graph G created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# --- 2. Data Preprocessing for Socioeconomic Scores ---
print("\nStep 2: Preprocessing node data for socioeconomic scores...")
nodes_df['cbs_square'] = nodes_df['cbs_square'].astype(str)
cbs_df['cbs_square'] = cbs_df['cbs_square'].astype(str)

SOCIO_COL_INCOME = 'median_income_k€'
SOCIO_COL_HOME_VALUE = 'avg_home_value_k€'
cbs_relevant_cols = ['cbs_square', SOCIO_COL_INCOME, SOCIO_COL_HOME_VALUE]

missing_cbs_cols = [col for col in cbs_relevant_cols if col not in cbs_df.columns and col != 'cbs_square']
if missing_cbs_cols:
    print(f"WARNING: CBS data is missing expected columns: {missing_cbs_cols}. Filling with 0 or median.")
    for col in missing_cbs_cols:
        cbs_df[col] = np.nan

nodes_ext_df = pd.merge(nodes_df[['node_id', 'cbs_square']], cbs_df[cbs_relevant_cols], on='cbs_square', how='left')

for col in [SOCIO_COL_INCOME, SOCIO_COL_HOME_VALUE]:
    if col in nodes_ext_df:
        median_val = nodes_ext_df[col].median()
        if pd.isna(median_val): median_val = 0
        nodes_ext_df[col].fillna(median_val, inplace=True)
    else:
        nodes_ext_df[col] = 0

nodes_ext_df['raw_socio_score'] = nodes_ext_df[SOCIO_COL_INCOME] + nodes_ext_df[SOCIO_COL_HOME_VALUE]

scaler = MinMaxScaler()
if not nodes_ext_df['raw_socio_score'].empty and nodes_ext_df['raw_socio_score'].nunique() > 1:
    nodes_ext_df['normalized_socio_score'] = scaler.fit_transform(nodes_ext_df[['raw_socio_score']])
else:
    nodes_ext_df['normalized_socio_score'] = 0.5

node_to_norm_socio_map = nodes_ext_df.set_index('node_id')['normalized_socio_score'].to_dict()
print("Socioeconomic factors calculated and mapped for nodes.")

# --- 3. Parcel Allocation Simulation Function ---
print("\nStep 3: Defining parcel allocation simulation function...")
distance_cache = {}

def simulate_parcel_dispatch_for_day(
    day_to_simulate_value, deliveries_data_df, service_points_nodes_map,
    sp_to_net_node_map, graph_obj, node_socio_map, dist_cache, seed, iteration_num, total_iterations
):
    day_specific_seed = seed + int(day_to_simulate_value) if isinstance(day_to_simulate_value, (int, float)) else seed
    np.random.seed(day_specific_seed)
    all_allocations_list = []
    
    if iteration_num % 20 == 0 or iteration_num == 1:
         print(f"  Simulating parcel dispatch for day: {day_to_simulate_value} ({iteration_num}/{total_iterations})")

    daily_deliveries = deliveries_data_df[deliveries_data_df['day'] == day_to_simulate_value].copy()
    if daily_deliveries.empty:
        return pd.DataFrame()

    for _, row in daily_deliveries.iterrows():
        sp_id = str(row['sp_id'])
        num_parcels_for_sp = int(row['parcels'])

        if num_parcels_for_sp == 0:
            continue

        sp_network_node = sp_to_net_node_map.get(sp_id)
        if sp_network_node is None:
            continue

        cluster_node_ids = service_points_nodes_map.get(sp_id, [])
        if not cluster_node_ids:
            continue

        candidate_info_list = []
        for node_id in cluster_node_ids:
            node_id = int(node_id)
            distance_m = float('inf')
            if (sp_network_node, node_id) in dist_cache:
                distance_m = dist_cache[(sp_network_node, node_id)]
            else:
                try:
                    if graph_obj.has_node(sp_network_node) and graph_obj.has_node(node_id):
                        if sp_network_node == node_id:
                            distance_m = 0.0
                        else:
                            distance_m = nx.shortest_path_length(graph_obj, source=sp_network_node, target=node_id, weight='weight')
                    dist_cache[(sp_network_node, node_id)] = distance_m
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    dist_cache[(sp_network_node, node_id)] = float('inf')
            
            norm_socio = node_socio_map.get(node_id, 0.0)
            
            if distance_m == float('inf'):
                dist_component = 0.0
            else:
                dist_component = 1.0 / (1.0 + distance_m)

            logit_weight = (0.7 * dist_component) + (0.3 * norm_socio)
            
            if logit_weight > 1e-9:
                candidate_info_list.append({
                    'node_id': node_id,
                    'distance_m': distance_m,
                    'norm_socio_score': norm_socio,
                    'logit_weight': logit_weight
                })
        
        if not candidate_info_list:
            if cluster_node_ids:
                chosen_nodes_for_sp = np.random.choice(cluster_node_ids, size=num_parcels_for_sp, replace=True)
                prob_fallback = 1.0 / len(cluster_node_ids) if len(cluster_node_ids) > 0 else 0
                for i, assigned_node_id in enumerate(chosen_nodes_for_sp):
                    all_allocations_list.append({
                        'day': day_to_simulate_value, 'sp_id': sp_id, 
                        'parcel_id': f"{sp_id}_{i+1}", 'node_id': assigned_node_id,
                        'draw_prob': prob_fallback,
                        'distance_m': dist_cache.get((sp_network_node, assigned_node_id), float('inf'))
                    })
            continue

        total_logit_weight = sum(c['logit_weight'] for c in candidate_info_list)
        if total_logit_weight <= 1e-9:
            if cluster_node_ids:
                nodes_for_fallback = [c['node_id'] for c in candidate_info_list] if candidate_info_list else cluster_node_ids
                if not nodes_for_fallback: continue

                chosen_nodes_for_fallback = np.random.choice(nodes_for_fallback, size=num_parcels_for_sp, replace=True)
                prob_fallback = 1.0 / len(nodes_for_fallback)
                for i, assigned_node_id in enumerate(chosen_nodes_for_fallback):
                    all_allocations_list.append({
                        'day': day_to_simulate_value, 'sp_id': sp_id, 
                        'parcel_id': f"{sp_id}_{i+1}", 'node_id': assigned_node_id,
                        'draw_prob': prob_fallback,
                        'distance_m': dist_cache.get((sp_network_node, assigned_node_id), float('inf'))
                    })
            continue

        for c in candidate_info_list:
            c['probability'] = c['logit_weight'] / total_logit_weight
        
        candidate_nodes = [c['node_id'] for c in candidate_info_list]
        probabilities = [c['probability'] for c in candidate_info_list]
        node_details_map = {c['node_id']: {'prob': c['probability'], 'dist': c['distance_m']} for c in candidate_info_list}
        num_effective_candidates = len(candidate_nodes)

        if num_parcels_for_sp <= num_effective_candidates:
            chosen_node_ids_for_batch = np.random.choice(
                candidate_nodes, size=num_parcels_for_sp, replace=False, p=probabilities
            )
            for i, assigned_node_id in enumerate(chosen_node_ids_for_batch):
                all_allocations_list.append({
                    'day': day_to_simulate_value, 'sp_id': sp_id,
                    'parcel_id': f"{sp_id}_{i+1}", 'node_id': assigned_node_id,
                    'draw_prob': node_details_map[assigned_node_id]['prob'],
                    'distance_m': node_details_map[assigned_node_id]['dist']
                })
        else:
            first_m_assigned_nodes = np.random.choice(
                candidate_nodes, size=num_effective_candidates, replace=False, p=probabilities
            )
            for i, assigned_node_id in enumerate(first_m_assigned_nodes):
                all_allocations_list.append({
                    'day': day_to_simulate_value, 'sp_id': sp_id,
                    'parcel_id': f"{sp_id}_{i+1}", 'node_id': assigned_node_id,
                    'draw_prob': node_details_map[assigned_node_id]['prob'],
                    'distance_m': node_details_map[assigned_node_id]['dist']
                })
            
            remaining_parcels_count = num_parcels_for_sp - num_effective_candidates
            if remaining_parcels_count > 0:
                chosen_nodes_for_remaining = np.random.choice(
                    candidate_nodes, size=remaining_parcels_count, replace=True, p=probabilities
                )
                for j, assigned_node_id in enumerate(chosen_nodes_for_remaining):
                    all_allocations_list.append({
                        'day': day_to_simulate_value, 'sp_id': sp_id,
                        'parcel_id': f"{sp_id}_{num_effective_candidates + j + 1}",
                        'node_id': assigned_node_id,
                        'draw_prob': node_details_map[assigned_node_id]['prob'],
                        'distance_m': node_details_map[assigned_node_id]['dist']
                    })
    
    return pd.DataFrame(all_allocations_list)

# --- 4. Main Execution: Simulate for all days and Calculate Yearly Cost ---
print("\nStep 4: Running main execution for all available days...")

if 'service_point_nodes_map' not in locals() or service_point_nodes_map is None:
    print("Critical Error: service_point_nodes_map not loaded. Exiting.")
    exit()
if 'sp_to_network_node_map' not in locals() or sp_to_network_node_map is None:
    print("Critical Error: sp_to_network_node_map not loaded. Exiting.")
    exit()

all_days_allocations_dfs = []
unique_days_to_simulate = deliveries_df['day'].unique()
total_days_to_simulate = len(unique_days_to_simulate)
print(f"Found {total_days_to_simulate} unique days in deliveries data to simulate.")

for i, day_val in enumerate(unique_days_to_simulate):
    daily_df = simulate_parcel_dispatch_for_day(
        day_val, deliveries_df, service_point_nodes_map,
        sp_to_network_node_map, G, node_to_norm_socio_map, 
        distance_cache, RANDOM_SEED, iteration_num=i+1, total_iterations=total_days_to_simulate
    )
    if not daily_df.empty:
        all_days_allocations_dfs.append(daily_df)

if not all_days_allocations_dfs:
    print("\nNo allocations were made across any simulated days. Cannot calculate costs.")
    exit()

full_period_allocations_df = pd.concat(all_days_allocations_dfs, ignore_index=True)

if not full_period_allocations_df.empty:
    print(f"\nParcel allocation simulation complete for {total_days_to_simulate} days.")
    print(f"Total parcels allocated over the period: {len(full_period_allocations_df)}")

    try:
        full_period_allocations_df.to_parquet(OUTPUT_PARQUET_FILE, engine='pyarrow', index=False)
        print(f"\nSaved all_days_allocations_df to {OUTPUT_PARQUET_FILE}")
    except Exception as e:
        print(f"Error saving to Parquet: {e}")
else:
    print("\nNo allocations were made across any simulated days. DataFrame is empty.")
    exit()

# --- 5. Yearly Cost Calculation ---
print("\nStep 5: Calculating yearly delivery cost...")
COST_PER_KM_EURO = 1.5

valid_allocations_df = full_period_allocations_df[
    np.isfinite(full_period_allocations_df['distance_m']) & (full_period_allocations_df['distance_m'] >= 0)
].copy()

if valid_allocations_df.empty:
    print("No valid allocations with finite distances found. Cannot calculate delivery costs.")
    estimated_yearly_cost = 0
else:
    valid_allocations_df['distance_km'] = valid_allocations_df['distance_m'] / 1000.0
    valid_allocations_df['parcel_delivery_cost'] = COST_PER_KM_EURO * valid_allocations_df['distance_km']
    
    total_cost_simulated_period = valid_allocations_df['parcel_delivery_cost'].sum()
    
    num_simulated_days = len(unique_days_to_simulate)
    
    if num_simulated_days > 0:
        average_daily_cost = total_cost_simulated_period / num_simulated_days
        estimated_yearly_cost = average_daily_cost * 365
        print(f"\nTotal delivery cost for the simulated period ({num_simulated_days} days): €{total_cost_simulated_period:,.2f}")
        print(f"Average daily delivery cost (from simulated period): €{average_daily_cost:,.2f}")
        print(f"Estimated total delivery cost for a full year (365 days): €{estimated_yearly_cost:,.2f}")
    else:
        print("No days were simulated (num_simulated_days is 0). Cannot calculate yearly cost.")
        estimated_yearly_cost = 0

print("\nSimulation and yearly cost estimation finished.")

