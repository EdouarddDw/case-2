# --- Notebook Setup ---
import pandas as pd
import numpy as np
import networkx as nx
import json
from sklearn.preprocessing import MinMaxScaler
import pyarrow  # For to_parquet
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# --- 0. Configuration ---
SP_TO_NETWORK_NODE_FILE = 'sp_to_network_node.json'
SERVICE_POINT_COVERAGE_FILE = 'service_point_coverage.json'
OUTPUT_PARQUET_FILE = 'all_days_parcel_allocations_optimized.parquet'
RANDOM_SEED = 42  # For reproducibility
NODES_EDGES_SP_SOURCE = 'imp_data'  # 'imp_data' or 'direct_files'
ALLOCATIONS_FILE = 'all_days_parcel_allocations_optimized.parquet'
COST_PER_KM_EURO = 1.5
# Unified constant for optimized service point CSV path
OPTIMIZED_SP_CSV = 'data/best_service_points_solution.csv'  # unified path

# --- 1. Load Data & Initial Setup ---
print("Step 1: Loading data and initial setup...")
np.random.seed(RANDOM_SEED)

if NODES_EDGES_SP_SOURCE == 'imp_data':
    try:
        from imp_data import load_maastricht_data  # Assuming imp_data.py is accessible
        data = load_maastricht_data()
        nodes_df = data["nodes"].copy()
        edges_df = data["edges"].copy()
        cbs_df = data["cbs"].copy()
        deliveries_df = data["deliveries"].copy()
        service_points_df = data["service_points"].copy()
        deliveries_df.rename(columns={'deliveries': 'parcels'}, inplace=True)
        deliveries_df['day'] = deliveries_df['day'].astype(str).str.strip()
        deliveries_df['sp_id'] = deliveries_df['sp_id'].astype(str).str.strip()

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
else:
    print("ERROR: Direct file loading not implemented in this example. Please use 'imp_data'.")
    exit()

try:
    with open(SERVICE_POINT_COVERAGE_FILE, 'r') as f:
        coverage_data = json.load(f)
    raw_sp_nodes_map = coverage_data.get('service_point_nodes')
    if raw_sp_nodes_map is None:
        raise KeyError("'service_point_nodes' key not found in coverage data.")
    temp_sp_nodes_map = {}
    for sp_id_key, node_list in raw_sp_nodes_map.items():
        key_str = str(sp_id_key).strip()
        temp_sp_nodes_map[key_str] = [int(n) for n in node_list]
    service_point_nodes_map = temp_sp_nodes_map
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
    sp_to_network_node_map = {str(k).strip(): int(v) for k, v in sp_to_network_node_map.items()}
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

# --- Load Optimized Configuration ---
print("\n[INFO] Loading optimized service point configuration and reassigning nodes and demand...")
try:
    optimized_df = pd.read_csv(OPTIMIZED_SP_CSV)
    if 'sp_id_int' not in optimized_df.columns:
        raise KeyError("'sp_id_int' column missing in optimized configuration file.")
    optimized_df['sp_id_int'] = optimized_df['sp_id_int'].astype(str)

    # Verify essential columns
    required_cols = {'opened'}
    missing_cols = required_cols - set(optimized_df.columns)
    if missing_cols:
        raise KeyError(f"Missing required column(s) in optimized configuration: {missing_cols}")

    # If manager_service_point column is absent, default each SP to manage itself
    if 'manager_service_point' not in optimized_df.columns:
        print("[WARN] 'manager_service_point' column missing. Defaulting each closed SP to its own ID (no demand transfer).")
        optimized_df['manager_service_point'] = optimized_df['sp_id_int']

    optimized_df['manager_service_point'] = optimized_df['manager_service_point'].astype(str)

    sp_open_status = optimized_df.set_index('sp_id_int')['opened'].to_dict()
    sp_managers = optimized_df.set_index('sp_id_int')['manager_service_point'].to_dict()

    # List of SP IDs that remain open in the optimized configuration
    open_sp_ids = [sp for sp, is_open in sp_open_status.items() if is_open]

except FileNotFoundError:
    print(f"[ERROR] Optimized configuration file not found: {OPTIMIZED_SP_CSV}")
    exit()
except Exception as e:
    print(f"[ERROR] Failed to load or process optimized service point configuration: {e}")
    exit()

# Redirect demand
deliveries_df['sp_id'] = deliveries_df['sp_id'].apply(
    lambda x: sp_managers.get(x, x) if not sp_open_status.get(x, True) else x
)

# --- Sanitize parcels column ---
deliveries_df['parcels'] = pd.to_numeric(deliveries_df['parcels'], errors='coerce').fillna(0).astype(int)

# Redirect coverage
new_service_point_nodes_map = {}
for sp_id, nodes in service_point_nodes_map.items():
    if not sp_open_status.get(sp_id, True):
        manager = sp_managers.get(sp_id)
        if manager:
            new_service_point_nodes_map.setdefault(manager, []).extend(nodes)
    else:
        new_service_point_nodes_map.setdefault(sp_id, []).extend(nodes)
# Remove duplicates
service_point_nodes_map = {k: list(set(v)) for k, v in new_service_point_nodes_map.items()}
print(f"[INFO] Updated coverage map to {len(service_point_nodes_map)} open service points.")

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
        # Guard against NaN or non‑positive parcel counts
        if pd.isna(row['parcels']) or row['parcels'] <= 0:
            continue
        num_parcels_for_sp = int(row['parcels'])

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
    print("\nNo allocations were made across any simulated days. Proceeding with zero allocations and a zero‑cost estimate.")
    full_period_allocations_df = pd.DataFrame(columns=[
        'day', 'sp_id', 'parcel_id', 'node_id', 'draw_prob', 'distance_m'
    ])
else:
    full_period_allocations_df = pd.concat(all_days_allocations_dfs, ignore_index=True)

# ---------------------------------------------------------------------------
# Guard: if there are still zero allocations, skip cost calculations cleanly
# ---------------------------------------------------------------------------
if full_period_allocations_df.empty:
    print("\n[INFO] No parcel allocations to cost. Setting estimated_yearly_cost = 0 and skipping cost analysis.")
    estimated_yearly_cost = 0
    valid_allocations_df = pd.DataFrame()  # ensure variable exists for later steps
    # Attempt a fallback: if a historical allocations Parquet exists, load it so
    # we can still compute costs and APL logic even when the current simulation
    # produced no rows.
    if os.path.exists(OUTPUT_PARQUET_FILE):
        try:
            full_period_allocations_df = pd.read_parquet(OUTPUT_PARQUET_FILE)
            print(f"[INFO] Fallback loaded {len(full_period_allocations_df)} allocation rows from {OUTPUT_PARQUET_FILE}.")
            valid_allocations_df = full_period_allocations_df.copy()
        except Exception as e:
            print(f"[WARN] Could not load fallback allocations file: {e}")
else:
    # Ensure distance_m is numeric even if the column is entirely NaN/empty
    full_period_allocations_df['distance_m'] = pd.to_numeric(
        full_period_allocations_df['distance_m'], errors='coerce'
    )
    if not full_period_allocations_df.empty:
        print(f"\nParcel allocation simulation complete for {total_days_to_simulate} days.")
        print(f"Total parcels allocated over the period: {len(full_period_allocations_df)}")

        try:
            full_period_allocations_df.to_parquet(OUTPUT_PARQUET_FILE, engine='pyarrow', index=False)
            print(f"\nSaved all_days_allocations_df to {OUTPUT_PARQUET_FILE}")
        except Exception as e:
            print(f"Error saving to Parquet: {e}")
    else:
        print("\nNo allocations were made across any simulated days. DataFrame is empty. Continuing with zero costs.")

    # --- 5. Yearly Cost Calculation ---
    print("\nStep 5: Calculating yearly delivery cost...")

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
    # If we reach here without setting estimated_yearly_cost, default to 0
    if 'estimated_yearly_cost' not in locals():
        estimated_yearly_cost = 0
 # ------------------------------------------------------------
# Will hold the node IDs of the 20 best APL candidates
# ------------------------------------------------------------
top_apl_node_ids: list[int] = []

# --- 5b. Evaluate potential Automated Parcel Locker (APL) locations ---
print("\nStep 5b: Evaluating potential Automated Parcel Locker (APL) placements...")

ALP_CAPACITY_PER_DAY = 24              # max parcels the locker can absorb daily
ALP_ANNUAL_COST_EURO = 10000            # fixed annual cost of a locker

# Decide which allocations set to evaluate: prefer the current run; fallback to parquet file
if valid_allocations_df.empty:
    try:
        fallback_alloc_df = pd.read_parquet(OUTPUT_PARQUET_FILE)
        if not fallback_alloc_df.empty:
            print("  Loaded allocations from existing parquet for APL evaluation.")
            # ensure numeric distance and cost
            fallback_alloc_df = fallback_alloc_df[np.isfinite(fallback_alloc_df['distance_m']) & (fallback_alloc_df['distance_m'] >= 0)].copy()
            fallback_alloc_df['distance_km'] = fallback_alloc_df['distance_m'] / 1000.0
            fallback_alloc_df['parcel_delivery_cost'] = COST_PER_KM_EURO * fallback_alloc_df['distance_km']
            alloc_for_apl = fallback_alloc_df
        else:
            alloc_for_apl = pd.DataFrame()
    except Exception as e:
        print(f"  Could not load fallback allocations: {e}")
        alloc_for_apl = pd.DataFrame()
else:
    alloc_for_apl = valid_allocations_df

if alloc_for_apl.empty:
    print("  No valid allocations to evaluate APL placements.")
    apl_candidates_df = pd.DataFrame(columns=[
        'node_id', 'total_delivery_cost', 'num_parcels_year',
        'mean_cost_per_parcel', 'potential_savings',
        'net_benefit'
    ])
else:
    # -----------------------------------------------------------------
    # Ensure cost column exists in alloc_for_apl
    # -----------------------------------------------------------------
    if 'parcel_delivery_cost' not in alloc_for_apl.columns:
        if 'distance_km' not in alloc_for_apl.columns:
            # make sure distance_m is numeric
            alloc_for_apl['distance_m'] = pd.to_numeric(alloc_for_apl['distance_m'], errors='coerce')
            alloc_for_apl = alloc_for_apl[np.isfinite(alloc_for_apl['distance_m']) & (alloc_for_apl['distance_m'] >= 0)]
            alloc_for_apl['distance_km'] = alloc_for_apl['distance_m'] / 1000.0
        alloc_for_apl['parcel_delivery_cost'] = COST_PER_KM_EURO * alloc_for_apl['distance_km']
    # 1) total annual parcels per node
    parcels_per_node_df = alloc_for_apl.groupby('node_id').size().reset_index(name='num_parcels_year')

    # 2) compute total delivery cost per node from alloc_for_apl
    node_costs_df = alloc_for_apl.groupby('node_id')['parcel_delivery_cost'].sum().reset_index()
    node_costs_df.rename(columns={'parcel_delivery_cost': 'total_delivery_cost'}, inplace=True)

    apl_eval_df = pd.merge(node_costs_df, parcels_per_node_df, on='node_id', how='inner')

    # safety: avoid divide‑by‑zero
    apl_eval_df['mean_cost_per_parcel'] = apl_eval_df.apply(
        lambda r: r['total_delivery_cost'] / r['num_parcels_year'] if r['num_parcels_year'] > 0 else 0, axis=1
    )

    # 3) Locker’s yearly capacity (not limited by node volume)
    max_parcels_locker_year = ALP_CAPACITY_PER_DAY * 365

    # 4) Potential savings assume the locker redirects its full capacity
    apl_eval_df['potential_savings'] = max_parcels_locker_year * apl_eval_df['mean_cost_per_parcel']

    # 5) net benefit after paying for the locker
    apl_eval_df['net_benefit'] = apl_eval_df['potential_savings'] - ALP_ANNUAL_COST_EURO

    # 6) keep only nodes with positive net benefit
    apl_candidates_df = apl_eval_df[apl_eval_df['net_benefit'] > 0].copy()

    if apl_candidates_df.empty:
        print("  No nodes offer a positive net benefit for an APL.")
    else:
        # -------------------------------------------------------------------
        # Diversified selection: keep lockers at least D_METERS apart while
        # still maximising net benefit.
        # -------------------------------------------------------------------
        D_METERS = 15_000  # minimum spacing between lockers in metres

        # Attach coordinates to the candidate table
        apl_candidates_df = apl_candidates_df.merge(
            nodes_df[['node_id', 'x_rd', 'y_rd']], on='node_id', how='left')

        # Greedy selection – walk the list in descending net‑benefit order
        qualified_ids = []
        qualified_rows = []

        for _, row in apl_candidates_df.sort_values('net_benefit', ascending=False).iterrows():
            x, y = row['x_rd'], row['y_rd']
            if pd.isna(x) or pd.isna(y):
                continue  # skip if we don’t know the coordinates
            # Check distance to every already‑kept locker
            far_enough = True
            for q in qualified_rows:
                dist = np.hypot(x - q['x_rd'], y - q['y_rd'])
                if dist < D_METERS:
                    far_enough = False
                    break
            if far_enough:
                qualified_ids.append(row['node_id'])
                qualified_rows.append(row)
            if len(qualified_ids) == 20:
                break

        # Build the final dataframe of picked locations
        top_apl_node_ids = qualified_ids
        top_20_apl_df = apl_candidates_df[apl_candidates_df['node_id'].isin(top_apl_node_ids)] \
                           .sort_values('net_benefit', ascending=False)

        if top_20_apl_df.empty:
            print("  Diversification removed all candidates – try reducing D_METERS.")
        else:
            print("\nTop 20 diversified APL locations (≥ "
                  f"{D_METERS/1000:.1f} km apart):")
            print(top_20_apl_df[['node_id', 'num_parcels_year',
                                 'mean_cost_per_parcel', 'potential_savings',
                                 'net_benefit']].to_string(index=False))

            # Save to CSV
            top_20_apl_df.to_csv('top_20_apl_candidates.csv', index=False)
            print("  Saved: top_20_apl_candidates.csv")

    # Give quick hint if fewer than 20 viable nodes
    if len(apl_candidates_df) < 20:
        print(f"  Only {len(apl_candidates_df)} nodes have a positive net benefit.")

# --- 6. Visualization ---
print("\nStep 6: Creating visualization...")
try:
    allocations_df = pd.read_parquet(ALLOCATIONS_FILE)
    print(f"Loaded parcel allocations from {ALLOCATIONS_FILE} ({len(allocations_df)} rows).")
except FileNotFoundError:
    print(f"ERROR: Allocations file not found: {ALLOCATIONS_FILE}")
    exit()
except Exception as e:
    print(f"ERROR: Could not read allocations file {ALLOCATIONS_FILE}: {e}")
    exit()

if 'distance_m' not in allocations_df.columns:
    print(f"ERROR: 'distance_m' column missing in {ALLOCATIONS_FILE}. Cannot calculate costs.")
    exit()

# Filter for valid allocations and calculate costs
valid_allocations = allocations_df[
    np.isfinite(allocations_df['distance_m']) & (allocations_df['distance_m'] >= 0)
].copy()

if valid_allocations.empty:
    print("No valid allocations with finite distances found in the simulation results.")
    node_total_costs_df = pd.DataFrame(columns=['node_id', 'total_delivery_cost'])
else:
    valid_allocations['distance_km'] = valid_allocations['distance_m'] / 1000.0
    valid_allocations['parcel_delivery_cost'] = COST_PER_KM_EURO * valid_allocations['distance_km']
    
    node_total_costs_df = valid_allocations.groupby('node_id')['parcel_delivery_cost'].sum().reset_index()
    node_total_costs_df.rename(columns={'parcel_delivery_cost': 'total_delivery_cost'}, inplace=True)
    print(f"Calculated total delivery costs for {len(node_total_costs_df)} unique customer nodes.")

nodes_df['node_id'] = nodes_df['node_id'].astype(node_total_costs_df['node_id'].dtype if not node_total_costs_df.empty else int)
nodes_for_plotting_df = pd.merge(nodes_df, node_total_costs_df, on='node_id', how='left')
nodes_for_plotting_df['total_delivery_cost'] = nodes_for_plotting_df['total_delivery_cost'].fillna(0)

plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(figsize=(18, 15))
ax.set_facecolor('white')
ax.set_xticks([])
ax.set_yticks([])

print("  Plotting road network edges...")
plotted_edges_count = 0
for _, edge in edges_df.iterrows():
    from_node_coords = nodes_df[nodes_df['node_id'] == edge['from_node']]
    to_node_coords = nodes_df[nodes_df['node_id'] == edge['to_node']]

    if not from_node_coords.empty and not to_node_coords.empty:
        from_x, from_y = from_node_coords[['x_rd', 'y_rd']].iloc[0]
        to_x, to_y = to_node_coords[['x_rd', 'y_rd']].iloc[0]
        ax.plot([from_x, to_x], [from_y, to_y],
                color='#999999',
                linewidth=1.2,
                alpha=0.8,
                zorder=1)
        plotted_edges_count += 1
print(f"    ...{plotted_edges_count} edges plotted.")

print("  Plotting customer nodes...")
nodes_with_cost = nodes_for_plotting_df[nodes_for_plotting_df['total_delivery_cost'] > 0]
nodes_zero_cost = nodes_for_plotting_df[nodes_for_plotting_df['total_delivery_cost'] == 0]

if not nodes_zero_cost.empty:
    ax.scatter(
        nodes_zero_cost['x_rd'],
        nodes_zero_cost['y_rd'],
        s=3,
        color='lightgray',
        alpha=0.25,
        zorder=2,
        label='Nodes (No Delivery Cost)'
    )
    print(f"    ...{len(nodes_zero_cost)} nodes with zero delivery cost plotted.")

if not nodes_with_cost.empty:
    min_cost = nodes_with_cost['total_delivery_cost'].min()
    max_cost = nodes_with_cost['total_delivery_cost'].quantile(0.99)
    if max_cost <= min_cost: max_cost = nodes_with_cost['total_delivery_cost'].max()

    norm = mcolors.LogNorm(vmin=max(1, min_cost), vmax=max_cost) if min_cost > 0 else mcolors.Normalize(vmin=min_cost, vmax=max_cost)
    cmap = plt.colormaps.get_cmap('YlOrRd')

    scatter_nodes = ax.scatter(
        nodes_with_cost['x_rd'],
        nodes_with_cost['y_rd'],
        s=50,
        c=nodes_with_cost['total_delivery_cost'],
        cmap=cmap,
        norm=norm,
        alpha=0.85,
        marker='o',
        edgecolors='black',
        linewidths=0.5,
        zorder=3,
        label='Customer Nodes (by Cost)'
    )
    cbar = fig.colorbar(scatter_nodes, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
    cbar.set_label('Total Delivery Cost per Node (€)', fontsize=12)
    print(f"    ...{len(nodes_with_cost)} nodes with delivery costs plotted.")
else:
    print("    ...No customer nodes with delivery costs to plot.")

print("  Plotting service points...")
# ----------------------------------------------------------------------------
# Robust logic: always show open SPs even if coordinates are missing in
# service_points_df by falling back to the optimized configuration merge.
# ----------------------------------------------------------------------------
sp_open_plot_df = pd.DataFrame()

# Ensure service_points_df has sp_id as string
if not service_points_df.empty:
    service_points_df['sp_id'] = service_points_df['sp_id'].astype(str)

    # First attempt: direct filter from service_points_df
    sp_open_plot_df = service_points_df[service_points_df['sp_id'].isin(open_sp_ids)].copy()

# Fallback if we didn't find any or coordinates columns are missing
if (sp_open_plot_df.empty or
    not {'x_rd', 'y_rd'}.issubset(sp_open_plot_df.columns)):

    # Merge coordinates from service_points_df with optimized_df
    if not service_points_df.empty and {'x_rd', 'y_rd'}.issubset(service_points_df.columns):
        merged_df = pd.merge(
            optimized_df[optimized_df['opened']],            # open SPs only
            service_points_df[['sp_id', 'x_rd', 'y_rd']],    # coordinates
            left_on='sp_id_int',
            right_on='sp_id',
            how='left'
        )
    else:
        # As a last resort, rely solely on the coordinates in service_points_df (if any)
        merged_df = optimized_df[optimized_df['opened']].copy()
        if {'x_rd', 'y_rd'}.issubset(merged_df.columns):
            merged_df.rename(columns={'x_rd': 'x_rd', 'y_rd': 'y_rd'}, inplace=True)
        else:
            # No coordinates available at all
            merged_df['x_rd'] = np.nan
            merged_df['y_rd'] = np.nan

    # Harmonise column names
    if 'sp_id_int' in merged_df.columns:
        merged_df.rename(columns={'sp_id_int': 'sp_id'}, inplace=True)

    # If 'x_rd' / 'y_rd' exist, keep non‑NA rows; otherwise leave empty for next fallback
    if {'x_rd', 'y_rd'}.issubset(merged_df.columns):
        sp_open_plot_df = merged_df[['sp_id', 'x_rd', 'y_rd']].dropna(subset=['x_rd', 'y_rd'])
    else:
        sp_open_plot_df = pd.DataFrame()

# ------------------------------------------------------------------------
# Fallback #2: for any remaining open SPs without explicit coordinates,
# derive them via sp_to_network_node_map → nodes_df.
# ------------------------------------------------------------------------
if not sp_open_plot_df.empty:
    missing_sp_ids = [sp for sp in open_sp_ids if sp not in sp_open_plot_df['sp_id'].values]
else:
    missing_sp_ids = open_sp_ids.copy()

    if missing_sp_ids:
        node_coord_lookup = nodes_df.set_index('node_id')[['x_rd', 'y_rd']].to_dict('index')
        extra_rows = []
        for sp in missing_sp_ids:
            node_id = sp_to_network_node_map.get(sp)
            if node_id is None:
                continue
            coord = node_coord_lookup.get(node_id)
            if coord and not pd.isna(coord['x_rd']) and not pd.isna(coord['y_rd']):
                extra_rows.append({'sp_id': sp, 'x_rd': coord['x_rd'], 'y_rd': coord['y_rd']})
        if extra_rows:
            sp_open_plot_df = pd.concat([sp_open_plot_df, pd.DataFrame(extra_rows)], ignore_index=True)

 # Highlight the best APL nodes (gold stars)
if 'top_apl_node_ids' in locals() and top_apl_node_ids:
    apl_nodes_coords = nodes_df[nodes_df['node_id'].isin(top_apl_node_ids)]
    ax.scatter(
        apl_nodes_coords['x_rd'],
        apl_nodes_coords['y_rd'],
        s=600,  # increased size for visibility
        marker='*',
        color='green',
        edgecolors='black',
        linewidths=1.2,
        alpha=0.95,
        zorder=5,
        label='Top APL Nodes'
    )
    print(f"    ...{len(apl_nodes_coords)} top APL candidate nodes plotted.")
else:
    print("    ...0 top APL candidate nodes plotted.")
# Final plotting
if sp_open_plot_df.empty:
    print("    ...WARNING: No open service points found to plot.")
else:
    ax.scatter(
        sp_open_plot_df['x_rd'],
        sp_open_plot_df['y_rd'],
        s=120,
        color='blue',
        marker='P',
        edgecolors='black',
        linewidths=1,
        alpha=0.9,
        zorder=4,
        label='Open Service Points'
    )
    print(f"    ...{len(sp_open_plot_df)} open service points plotted.")

ax.set_title('Simulated Total Delivery Cost per Customer Node', fontsize=18, fontweight='bold')
ax.set_xlabel('X Coordinate (RD)', fontsize=12)
ax.set_ylabel('Y Coordinate (RD)', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.set_aspect('equal', adjustable='box')

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
if by_label:
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10, frameon=True, facecolor='white', framealpha=0.7)

plt.tight_layout(pad=1.5)
output_map_file = "delivery_cost_simulation_map_optimized.png"
plt.savefig(output_map_file, dpi=300)
print(f"\nMap saved to {output_map_file}")
plt.show()

# --- 7. Additional Cost Analysis Visualizations ---
print("\nStep 7: Generating additional cost analysis visualizations...")

if valid_allocations_df.empty:
    print("  No valid allocations data to generate additional plots.")
else:
    # Ensure 'sp_id' is treated as a categorical variable for plotting if it's numeric
    valid_allocations_df['sp_id'] = valid_allocations_df['sp_id'].astype(str)

    # 7.1: Bar Chart of Total Delivery Costs per Service Point
    plt.figure(figsize=(16, 8))
    total_cost_per_sp = valid_allocations_df.groupby('sp_id')['parcel_delivery_cost'].sum().sort_values(ascending=False)
    
    # Limit to top N service points if there are too many, for readability
    top_n_sp = 25 
    if len(total_cost_per_sp) > top_n_sp:
        total_cost_per_sp_plot = total_cost_per_sp.head(top_n_sp)
        plot_title_suffix = f" (Top {top_n_sp})"
    else:
        total_cost_per_sp_plot = total_cost_per_sp
        plot_title_suffix = ""

    total_cost_per_sp_plot.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title(f'Total Delivery Costs per Service Point{plot_title_suffix}', fontsize=16, fontweight='bold')
    plt.xlabel('Service Point ID', fontsize=12)
    plt.ylabel('Total Delivery Cost (€)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('total_delivery_costs_per_sp_optimized.png', dpi=300)
    print("  Saved: total_delivery_costs_per_sp_optimized.png")
    plt.show()

    # 7.2: Box Plot of Parcel Delivery Costs per Service Point
    # For readability, if there are many service points, consider plotting only a subset
    # For example, the top N by total cost, or a random sample.
    # Here, we'll use the same top_n_sp IDs from the bar chart for consistency.
    
    sp_ids_for_boxplot = total_cost_per_sp_plot.index.tolist()
    boxplot_data = valid_allocations_df[valid_allocations_df['sp_id'].isin(sp_ids_for_boxplot)]

    if not boxplot_data.empty:
        plt.figure(figsize=(18, 10))
        # Create a boxplot. Pandas can do this directly or use seaborn for more options.
        # Using matplotlib's boxplot for directness here.
        # We need to prepare data in a list of lists/arrays for matplotlib's boxplot if plotting multiple SPs.
        data_to_plot = [boxplot_data[boxplot_data['sp_id'] == sp]['parcel_delivery_cost'].dropna().values for sp in sp_ids_for_boxplot]
        
        # Filter out empty arrays which can cause errors in boxplot
        filtered_data_to_plot = [arr for arr in data_to_plot if len(arr) > 0]
        filtered_sp_ids = [sp for sp, arr in zip(sp_ids_for_boxplot, data_to_plot) if len(arr) > 0]

        if filtered_data_to_plot:
            bp = plt.boxplot(filtered_data_to_plot, patch_artist=True, medianprops={'color': 'black'})
            
            # Color the boxes
            colors = plt.cm.get_cmap('Pastel2', len(filtered_sp_ids))
            for patch, color in zip(bp['boxes'], colors(np.arange(len(filtered_sp_ids)))):
                patch.set_facecolor(color)

            plt.title(f'Distribution of Parcel Delivery Costs per Service Point{plot_title_suffix}', fontsize=16, fontweight='bold')
            plt.xlabel('Service Point ID', fontsize=12)
            plt.ylabel('Parcel Delivery Cost (€)', fontsize=12)
            plt.xticks(ticks=np.arange(1, len(filtered_sp_ids) + 1), labels=filtered_sp_ids, rotation=45, ha='right', fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig('parcel_delivery_cost_distribution_per_sp_optimized.png', dpi=300)
            print("  Saved: parcel_delivery_cost_distribution_per_sp_optimized.png")
            plt.show()
        else:
            print("  Skipped boxplot: No data after filtering for selected SP IDs.")
    else:
        print("  Skipped boxplot: No data for the selected service points.")


print("\nVisualization script finished.")

