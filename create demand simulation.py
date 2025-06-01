import random
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
import imp_data
from typing import Dict, Any # Add this line


data = imp_data.load_maastricht_data()  # Importing the data module which contains helper data
# ----------------------------------------------------------------------

# -------------------------------------------------------------------


config: Dict[str, Any] = {
        'SP_TO_NETWORK_NODE_FILE_NAME': 'sp_to_network_node.json',
        'SERVICE_POINT_COVERAGE_FILE_NAME': 'service_point_coverage.json',
        'DEMAND_SIM_CONFIG': {
            'W_DIST_PICKUP': 0.7, 'W_SOCIO_PICKUP': 0.3,
            'W_DIST_ATHOME': 0.3, 'W_SOCIO_ATHOME': 0.7,
            'DEFAULT_SOCIO_SCORE_IF_MISSING': 0.5
        }
}

# ----------------------------------------------------------------------
# 1.  Load helper data from imp_data.py
# ----------------------------------------------------------------------

distance_cache = {}  # Global cache for distances between nodes

def get_dynamic_pickup_probability(distance_meters):
        if distance_meters <= 200: return 0.8
        elif distance_meters > 2000: return 0.0
        else: return 0.8 - (((distance_meters - 200.0) / 1800.0) * 0.8)

def _select_node_for_delivery_parcel(base_candidate_info_list, parcel_status_for_weighting='at home'):
    if not base_candidate_info_list: return None
    weighted_candidates = []
    for cand_info in base_candidate_info_list:
        dist_m = cand_info['distance_m']; norm_socio = cand_info['norm_socio_score']
        dist_component = 1.0 / (1.0 + dist_m / 1000.0) if dist_m != float('inf') and dist_m >=0 else 0.0
        logit_w = ((config['DEMAND_SIM_CONFIG']['W_DIST_ATHOME'] * dist_component) + (config['DEMAND_SIM_CONFIG']['W_SOCIO_ATHOME'] * norm_socio)) if parcel_status_for_weighting == 'at home' else ((config['DEMAND_SIM_CONFIG']['W_DIST_PICKUP'] * dist_component) + (config['DEMAND_SIM_CONFIG']['W_SOCIO_PICKUP'] * (1.0-norm_socio)))
        if logit_w > 1e-9: weighted_candidates.append({'node_id': cand_info['node_id'], 'distance_m': dist_m, 'logit_weight': logit_w, 'norm_socio_score': norm_socio})
    if not weighted_candidates: return random.choice(base_candidate_info_list) if base_candidate_info_list else None
    total_logit_weight = sum(c['logit_weight'] for c in weighted_candidates)
    if total_logit_weight <= 1e-9: return random.choice(weighted_candidates) if weighted_candidates else None
    probabilities = [c['logit_weight'] / total_logit_weight for c in weighted_candidates]
    return weighted_candidates[np.random.choice(len(weighted_candidates), p=probabilities)]

def _assign_parcels_of_type(num_parcels_to_assign, parcel_status, base_candidate_info_list, sp_id_str, day_val, parcel_idx_start, sp_network_node):
    assigned_parcels_for_type = []
    if num_parcels_to_assign == 0 or not base_candidate_info_list: return assigned_parcels_for_type, parcel_idx_start # This is a valid return
    type_specific_candidates = []
    for cand_info in base_candidate_info_list:
        dist_m = cand_info['distance_m']; norm_socio = cand_info['norm_socio_score']
        dist_component = 1.0 / (1.0 + dist_m / 1000.0) if dist_m != float('inf') and dist_m >=0 else 0.0
        logit_w = ((config['DEMAND_SIM_CONFIG']['W_DIST_PICKUP'] * dist_component) + (config['DEMAND_SIM_CONFIG']['W_SOCIO_PICKUP'] * (1.0-norm_socio))) if parcel_status == 'pick up' else ((config['DEMAND_SIM_CONFIG']['W_DIST_ATHOME'] * dist_component) + (config['DEMAND_SIM_CONFIG']['W_SOCIO_ATHOME'] * norm_socio))
        if logit_w > 1e-9: type_specific_candidates.append({'node_id': cand_info['node_id'], 'distance_m': dist_m, 'logit_weight': logit_w})

    if not type_specific_candidates: # Fallback 1
        if base_candidate_info_list:
            chosen_nodes = np.random.choice([c['node_id'] for c in base_candidate_info_list], size=num_parcels_to_assign, replace=True)
            prob_fallback = 1.0 / len(base_candidate_info_list) if base_candidate_info_list else 0
            for i, assigned_node_id in enumerate(chosen_nodes):
                fallback_dist_key = tuple(sorted((sp_network_node, assigned_node_id))); fallback_dist = distance_cache.get(fallback_dist_key, float('inf'))
                assigned_parcels_for_type.append({'day': day_val, 'sp_id': sp_id_str, 'parcel_id': f"{sp_id_str}_{parcel_idx_start + i + 1}", 'node_id': assigned_node_id, 'status': parcel_status, 'draw_prob': prob_fallback, 'distance_m': fallback_dist, 'origin_source': 'pickups_data' if parcel_status == 'pick up' else 'deliveries_data'})
            parcel_idx_start += num_parcels_to_assign
        return assigned_parcels_for_type, parcel_idx_start # This is also a valid return

        total_logit_weight = sum(c['logit_weight'] for c in type_specific_candidates) # This line is indented incorrectly
        if total_logit_weight <= 1e-9: # Fallback 2
            chosen_nodes = np.random.choice([c['node_id'] for c in type_specific_candidates], size=num_parcels_to_assign, replace=True)
            prob_fallback = 1.0 / len(type_specific_candidates) if type_specific_candidates else 0
            for i, assigned_node_id in enumerate(chosen_nodes):
                node_detail = next((c for c in type_specific_candidates if c['node_id'] == assigned_node_id), {'distance_m': float('inf')})
                assigned_parcels_for_type.append({'day': day_val, 'sp_id': sp_id_str, 'parcel_id': f"{sp_id_str}_{parcel_idx_start + i + 1}", 'node_id': assigned_node_id, 'status': parcel_status, 'draw_prob': prob_fallback, 'distance_m': node_detail['distance_m'], 'origin_source': 'pickups_data' if parcel_status == 'pick up' else 'deliveries_data'})
            parcel_idx_start += num_parcels_to_assign
            return assigned_parcels_for_type, parcel_idx_start # This is also a valid return
    
    # ... rest of the function ...
    return assigned_parcels_for_type, parcel_idx_start # Main valid return


# Placeholder for deliveries data

def simulate_parcel_dispatch_for_day_local(day_to_simulate_value, deliveries_data_df, pickups_data_df, service_points_nodes_map_local, sp_to_net_node_map_local, graph_obj_local, node_socio_map_local, current_seed, iteration_num, total_iterations):
    day_specific_seed = current_seed + int(day_to_simulate_value) if isinstance(day_to_simulate_value, (int, float)) else current_seed
    np.random.seed(day_specific_seed)
    all_allocations_list = []
    daily_deliveries_raw = deliveries_data_df[deliveries_data_df['day'] == day_to_simulate_value].copy()
    daily_pickups_raw = pd.DataFrame()
    if not pickups_data_df.empty and 'day' in pickups_data_df.columns: daily_pickups_raw = pickups_data_df[pickups_data_df['day'] == day_to_simulate_value].copy()
    if not daily_deliveries_raw.empty: daily_deliveries_raw['sp_id'] = daily_deliveries_raw['sp_id'].astype(str)
    if not daily_pickups_raw.empty: daily_pickups_raw['sp_id'] = daily_pickups_raw['sp_id'].astype(str)
    agg_deliveries = daily_deliveries_raw.groupby('sp_id')['parcels'].sum().reset_index().rename(columns={'parcels': 'parcels_from_delivery'}) if not daily_deliveries_raw.empty else pd.DataFrame(columns=['sp_id', 'parcels_from_delivery'])
    agg_pickups = daily_pickups_raw.groupby('sp_id')['parcels'].sum().reset_index().rename(columns={'parcels': 'parcels_from_pickup'}) if not daily_pickups_raw.empty and 'parcels' in daily_pickups_raw.columns else pd.DataFrame(columns=['sp_id', 'parcels_from_pickup'])
    if not agg_deliveries.empty and not agg_pickups.empty: daily_combined_demand = pd.merge(agg_deliveries, agg_pickups, on='sp_id', how='outer')
    elif not agg_deliveries.empty: daily_combined_demand = agg_deliveries
    elif not agg_pickups.empty: daily_combined_demand = agg_pickups
    else: return pd.DataFrame()
    for col_name in ['parcels_from_delivery', 'parcels_from_pickup']:
        if col_name not in daily_combined_demand.columns: daily_combined_demand[col_name] = 0
        else: daily_combined_demand[col_name] = daily_combined_demand[col_name].fillna(0).astype(int)
    if daily_combined_demand.empty: return pd.DataFrame()
    parcel_sp_idx_counter = defaultdict(int)
    for _, row in daily_combined_demand.iterrows():
        sp_id = str(row['sp_id']); parcels_direct_delivery = int(row['parcels_from_delivery']); parcels_direct_pickup = int(row['parcels_from_pickup'])
        if (parcels_direct_delivery + parcels_direct_pickup) == 0: continue
        sp_network_node = sp_to_net_node_map_local.get(sp_id)
        if sp_network_node is None: continue
        cluster_node_ids = service_points_nodes_map_local.get(sp_id, [])
        if not cluster_node_ids: continue
        base_candidate_info = []
        for node_id_int in cluster_node_ids:
            node_id = int(node_id_int); distance_m_val = float('inf')
            cache_key = tuple(sorted((sp_network_node, node_id)))
            if cache_key in distance_cache: distance_m_val = distance_cache[cache_key]
            else:
                try:
                    if graph_obj_local.has_node(sp_network_node) and graph_obj_local.has_node(node_id): distance_m_val = nx.shortest_path_length(graph_obj_local, source=sp_network_node, target=node_id, weight='weight') if sp_network_node != node_id else 0.0
                    distance_cache[cache_key] = distance_m_val
                except (nx.NetworkXNoPath, nx.NodeNotFound): distance_cache[cache_key] = float('inf')
            norm_socio_val = node_socio_map_local.get(node_id, config['DEMAND_SIM_CONFIG']['DEFAULT_SOCIO_SCORE_IF_MISSING'])
            base_candidate_info.append({'node_id': node_id, 'distance_m': distance_m_val, 'norm_socio_score': norm_socio_val})
        if not base_candidate_info: continue
        if parcels_direct_pickup > 0:
            assigned_pickups, parcel_sp_idx_counter[sp_id] = _assign_parcels_of_type(parcels_direct_pickup, 'pick up', base_candidate_info, sp_id, day_to_simulate_value, parcel_sp_idx_counter[sp_id], sp_network_node)
            all_allocations_list.extend(assigned_pickups)
        for _ in range(parcels_direct_delivery):
            chosen_node_details = _select_node_for_delivery_parcel(base_candidate_info, parcel_status_for_weighting='at home')
            if chosen_node_details is None: continue
            assigned_node_id = chosen_node_details['node_id']; distance_to_sp = chosen_node_details['distance_m']
            pickup_prob = get_dynamic_pickup_probability(distance_to_sp)
            final_status = 'pick up' if np.random.rand() < pickup_prob else 'at home'
            parcel_sp_idx_counter[sp_id] += 1; parcel_id = f"{sp_id}_{parcel_sp_idx_counter[sp_id]}"
            draw_prob_val = chosen_node_details.get('logit_weight',0) / sum(c.get('logit_weight',0) for c in base_candidate_info if c.get('logit_weight',0) > 0) if sum(c.get('logit_weight',0) for c in base_candidate_info if c.get('logit_weight',0) > 0) > 0 else 0
            all_allocations_list.append({'day': day_to_simulate_value, 'sp_id': sp_id, 'parcel_id': parcel_id, 'node_id': assigned_node_id, 'status': final_status, 'draw_prob': draw_prob_val, 'distance_m': distance_to_sp, 'origin_source': 'deliveries_data'})
    return pd.DataFrame(all_allocations_list)

def create_daily_sp_df_local(sim_allocations_df):
    if sim_allocations_df.empty or not all(col in sim_allocations_df.columns for col in ['sp_id', 'day', 'status', 'distance_m']):
        return pd.DataFrame(columns=['LocationID', 'day', 'pred_pickup', 'pred_home', 'distance'])
    sim_allocations_df['status'] = sim_allocations_df['status'].astype(str)
    daily_counts = sim_allocations_df.groupby(['sp_id', 'day', 'status']).size().unstack(fill_value=0)
    if 'pick up' in daily_counts.columns: daily_counts.rename(columns={'pick up': 'pred_pickup'}, inplace=True)
    else: daily_counts['pred_pickup'] = 0
    if 'at home' in daily_counts.columns: daily_counts.rename(columns={'at home': 'pred_home'}, inplace=True)
    else: daily_counts['pred_home'] = 0
    daily_sp_intermediate = daily_counts[['pred_pickup', 'pred_home']].reset_index()
    if daily_sp_intermediate.empty: return pd.DataFrame(columns=['LocationID', 'day', 'pred_pickup', 'pred_home', 'distance'])
    at_home_parcels = sim_allocations_df[sim_allocations_df['status'] == 'at home']
    if not at_home_parcels.empty:
        sp_avg_home_distance = at_home_parcels.groupby('sp_id')['distance_m'].mean().reset_index()
        sp_avg_home_distance.rename(columns={'distance_m': 'distance'}, inplace=True)
    else:
        sp_id_dtype = sim_allocations_df['sp_id'].dtype if 'sp_id' in sim_allocations_df else object
        sp_avg_home_distance = pd.DataFrame({'sp_id': pd.Series(dtype=sp_id_dtype), 'distance': pd.Series(dtype=float)})
    daily_sp_df_res = pd.merge(daily_sp_intermediate, sp_avg_home_distance, on='sp_id', how='left')
    daily_sp_df_res['distance'].fillna(0, inplace=True)
    daily_sp_df_res.rename(columns={'sp_id': 'LocationID'}, inplace=True)
    return daily_sp_df_res[['LocationID', 'day', 'pred_pickup', 'pred_home', 'distance']]

if __name__ == "__main__":
    import json
    from pathlib import Path
    from sklearn.preprocessing import MinMaxScaler # Required for socio-economic score calculation

    print("Starting demand simulation process...")

    # --- Configuration ---
    RANDOM_SEED = 42 # Define a seed for reproducibility
    OUTPUT_CSV_FILENAME = "simulated_daily_sp_output.csv"
    DATA_DIR = "." # Assuming JSON config files are in the same directory as the script

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    print(f"Using random seed: {RANDOM_SEED}")

    # --- Retrieve globally loaded data ---
    nodes_df = data["nodes"].copy()
    edges_df = data["edges"].copy()
    cbs_df = data["cbs"].copy()
    deliveries_df = data["deliveries"].copy()
    pickups_df = data.get("pickups", pd.DataFrame()).copy()
    # service_points_df = data["service_points"].copy() # Not directly used by sim funcs but good to have

    # Basic preprocessing for deliveries_df and pickups_df
    deliveries_df['parcels'] = deliveries_df['parcels'].fillna(0).astype(int)
    if not pickups_df.empty:
        if 'parcels' in pickups_df.columns:
            pickups_df['parcels'] = pickups_df['parcels'].fillna(0).astype(int)
        elif 'pickups' in pickups_df.columns: # Handle if original column name was 'pickups'
            pickups_df.rename(columns={'pickups': 'parcels'}, inplace=True)
            pickups_df['parcels'] = pickups_df['parcels'].fillna(0).astype(int)
        else:
            print("Warning: 'parcels' column (or 'pickups') not found in pickups_df. Assuming 0 parcels.")
            pickups_df['parcels'] = 0
    else: # Ensure pickups_df has expected columns even if empty
        pickups_df = pd.DataFrame(columns=['day', 'sp_id', 'parcels'])
    print("Base data loaded and preprocessed.")

    # --- Load Pre-calculated Maps from JSON files ---
    sp_coverage_path = Path(DATA_DIR) / config['SERVICE_POINT_COVERAGE_FILE_NAME']
    sp_to_node_path = Path(DATA_DIR) / config['SP_TO_NETWORK_NODE_FILE_NAME']

    try:
        with open(sp_coverage_path, 'r') as f:
            coverage_data_json = json.load(f)
        service_point_nodes_map_main = coverage_data_json.get('service_point_nodes', {})
        service_point_nodes_map_main = {str(k): [int(n) for n in v] for k, v in service_point_nodes_map_main.items()}
        print(f"Loaded service point coverage from: {sp_coverage_path}")
    except FileNotFoundError:
        print(f"ERROR: Service point coverage file not found at {sp_coverage_path}.")
        exit()
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {sp_coverage_path}.")
        exit()

    try:
        with open(sp_to_node_path, 'r') as f:
            sp_to_network_node_map_main = json.load(f)
        sp_to_network_node_map_main = {str(k): int(v) for k, v in sp_to_network_node_map_main.items()}
        print(f"Loaded SP to network node map from: {sp_to_node_path}")
    except FileNotFoundError:
        print(f"ERROR: SP to network node map file not found at {sp_to_node_path}.")
        exit()
    except json.JSONDecodeError:
        print(f"ERROR: Could not decode JSON from {sp_to_node_path}.")
        exit()

    # --- Create Graph ---
    G = nx.Graph()
    for _, node_row in nodes_df.iterrows():
        G.add_node(int(node_row['node_id']), pos=(float(node_row['x_rd']), float(node_row['y_rd'])))
    for _, edge_row in edges_df.iterrows():
        G.add_edge(int(edge_row['from_node']), int(edge_row['to_node']), weight=float(edge_row['length_m']))
    print(f"Graph G created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # --- Socioeconomic Score Calculation ---
    print("Calculating socioeconomic scores...")
    nodes_df['cbs_square'] = nodes_df['cbs_square'].astype(str)
    cbs_df['cbs_square'] = cbs_df['cbs_square'].astype(str)
    SOCIO_COL_INCOME = 'median_income_k€'
    SOCIO_COL_HOME_VALUE = 'avg_home_value_k€'

    for col_name_socio in [SOCIO_COL_INCOME, SOCIO_COL_HOME_VALUE]:
        if col_name_socio not in cbs_df.columns:
            print(f"Warning: Socio-economic column '{col_name_socio}' not found in cbs_df. Filling with 0.")
            cbs_df[col_name_socio] = 0
            
    nodes_ext_df = pd.merge(nodes_df[['node_id', 'cbs_square']], cbs_df[['cbs_square', SOCIO_COL_INCOME, SOCIO_COL_HOME_VALUE]], on='cbs_square', how='left')
    
    for col_name_socio in [SOCIO_COL_INCOME, SOCIO_COL_HOME_VALUE]:
        fill_value = nodes_ext_df[col_name_socio].median()
        if pd.isna(fill_value): fill_value = 0
        nodes_ext_df[col_name_socio].fillna(fill_value, inplace=True)
        
    nodes_ext_df['raw_socio_score'] = nodes_ext_df[SOCIO_COL_INCOME] + nodes_ext_df[SOCIO_COL_HOME_VALUE]
    scaler = MinMaxScaler()
    if not nodes_ext_df['raw_socio_score'].empty and nodes_ext_df['raw_socio_score'].nunique() > 1:
        nodes_ext_df['normalized_socio_score'] = scaler.fit_transform(nodes_ext_df[['raw_socio_score']])
    else:
        print("Warning: Not enough unique values in 'raw_socio_score' to normalize. Defaulting to 0.5.")
        nodes_ext_df['normalized_socio_score'] = 0.5
    node_to_norm_socio_map = nodes_ext_df.set_index('node_id')['normalized_socio_score'].to_dict()
    print("Socioeconomic scores calculated.")

    # --- Run Demand Simulation ---
    print("Running demand simulation loop...")
    all_simulated_allocations_dfs = []
    
    unique_days_in_deliveries = deliveries_df['day'].unique() if not deliveries_df.empty and 'day' in deliveries_df.columns else np.array([])
    unique_days_in_pickups = pickups_df['day'].unique() if not pickups_df.empty and 'day' in pickups_df.columns else np.array([])
    days_to_simulate_list = sorted(np.union1d(unique_days_in_deliveries, unique_days_in_pickups))

    if not days_to_simulate_list:
        print("ERROR: No days found in deliveries or pickups data to simulate.")
        exit()

    print(f"Simulating for {len(days_to_simulate_list)} unique day(s): {days_to_simulate_list[:5]}... (up to 5 shown)")
    current_sim_seed = RANDOM_SEED # Use a base seed for the simulation runs

    for i, day_value in enumerate(days_to_simulate_list):
        # Optional: print progress for each day
        # print(f"  Simulating day: {day_value} ({i+1}/{len(days_to_simulate_list)})")
        daily_allocations_df = simulate_parcel_dispatch_for_day_local(
            day_to_simulate_value=day_value,
            deliveries_data_df=deliveries_df,
            pickups_data_df=pickups_df,
            service_points_nodes_map_local=service_point_nodes_map_main,
            sp_to_net_node_map_local=sp_to_network_node_map_main,
            graph_obj_local=G,
            node_socio_map_local=node_to_norm_socio_map,
            current_seed=current_sim_seed, # Pass the seed
            iteration_num=i + 1,
            total_iterations=len(days_to_simulate_list)
        )
        if not daily_allocations_df.empty:
            all_simulated_allocations_dfs.append(daily_allocations_df)
    
    if not all_simulated_allocations_dfs:
        print("Simulation finished but produced no parcel allocations.")
        full_allocations_df = pd.DataFrame() # Ensure it's an empty df
    else:
        full_allocations_df = pd.concat(all_simulated_allocations_dfs, ignore_index=True)
        print(f"Simulation complete. Total parcels allocated: {len(full_allocations_df)}")

    # --- Create daily_sp DataFrame from simulation results ---
    if not full_allocations_df.empty:
        daily_sp_df_output = create_daily_sp_df_local(full_allocations_df)
        print("`daily_sp` DataFrame generated from simulation results.")

        # --- Save to CSV ---
        try:
            output_path = Path(DATA_DIR) / OUTPUT_CSV_FILENAME
            daily_sp_df_output.to_csv(output_path, index=False)
            print(f"Successfully saved simulated `daily_sp` data to: {output_path}")
        except Exception as e:
            print(f"Error saving `daily_sp` DataFrame to CSV: {e}")
    else:
        print("No allocations from simulation, so `daily_sp` DataFrame cannot be created or saved.")

    print("Demand simulation script finished.")


