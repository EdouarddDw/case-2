from pathlib import Path
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
from typing import Union, Dict, List, Tuple, Any # Added Union and other common types
import imp_data
import json

data = imp_data.load_maastricht_data()
nodes = data["nodes"]
edges = data["edges"]
service_points = data["service_points"]
pickup = data["pickups"]



#test head of pickup
print("Pickup DataFrame head:")
print(pickup.head())
# ---------------------------------------------------------------------------
# Build a clean integer ID column while tolerating any rogue / blank values.
# ---------------------------------------------------------------------------
service_points["sp_id_int"] = (
    pd.to_numeric(service_points["sp_id"], errors="coerce")  # → floats or NA
      .astype("Int64")                                       # nullable int
)

# Drop any rows where the ID could not be parsed
service_points = service_points.dropna(subset=["sp_id_int"]).copy()
service_points["sp_id_int"] = service_points["sp_id_int"].astype(int)

# Fast lookup by clean integer ID
sp_lookup = service_points.set_index("sp_id_int")

cbs = data["cbs"]  # This should contain housing data including home values

#create a distance matrix for service points
distance_matrix = pd.DataFrame(index=service_points['sp_id_int'], columns=service_points['sp_id_int'])
def calculate_distance(sp1: pd.Series, sp2: pd.Series) -> int:
    return math.sqrt((sp1['x_rd'] - sp2['x_rd']) ** 2 + (sp1['y_rd'] - sp2['y_rd']) ** 2)
for i, sp1 in service_points.iterrows():
    for j, sp2 in service_points.iterrows():
        if i <= j:  # Avoid redundant calculations
            distance = calculate_distance(sp1, sp2)
            distance_matrix.at[sp1['sp_id_int'], sp2['sp_id_int']] = distance
            distance_matrix.at[sp2['sp_id_int'], sp1['sp_id_int']] = distance  # Symmetric matrix
            
# Save the distance matrix to a CSV file
distance_matrix_file = Path("data/service_point_distance_matrix.csv")
distance_matrix.to_csv(distance_matrix_file, index=True)

coverage_path = Path("service_point_coverage.json")
with coverage_path.open() as f:
    coverage = json.load(f)["service_point_nodes"]

# Fast look‑up table for node coordinates keyed by integer node_id
nodes_by_id = nodes.set_index("node_id")

avg_distance_records: List[Dict[str, Union[float, int, None]]] = []
for sp_id_str, node_ids in coverage.items():
    # Cast IDs to integers to avoid the dreaded float‑as‑string “1001.0” headache
    sp_id = int(sp_id_str)

    # Retrieve service‑point coordinates (convert column to int once on the fly)
    sp_row = sp_lookup.loc[sp_id]
    sp_x, sp_y = sp_row["x_rd"], sp_row["y_rd"]

    dist_sum = 0.0
    valid_nodes = 0
    for node_id in node_ids:
        node_int = int(node_id)
        if node_int not in nodes_by_id.index:
            # Skip any orphaned node references gracefully
            continue

        node = nodes_by_id.loc[node_int]
        dist = math.hypot(sp_x - node["x_rd"], sp_y - node["y_rd"])
        dist_sum += dist
        valid_nodes += 1

    avg_dist = dist_sum / valid_nodes if valid_nodes else None
    avg_distance_records.append(
        {
            "sp_id": sp_id,
            "nodes_covered": valid_nodes,
            "avg_distance_m": avg_dist,
        }
    )

avg_distance_df = pd.DataFrame(avg_distance_records)


# add average distance to service points
avg_distance_df['sp_id_int'] = pd.to_numeric(avg_distance_df['sp_id'], errors='coerce').astype('Int64')

service_points = service_points.merge(
    avg_distance_df[['sp_id_int', 'nodes_covered', 'avg_distance_m']],
    on='sp_id_int',
    how='left',
    validate='one_to_one'
)
print("Average distance DataFrame:")
print(avg_distance_df.head())

print("Distance matrix (excerpt):")
print(distance_matrix.head())



# add a binary variable to each service point indicating if it is opened or not (intitial value is true)
service_points['opened'] = True

# add a variable to each service point indicating capacity (inittial value is the max pickup  of that service point)
max_pickup = pickup.groupby('sp_id')['pickups'].max().reset_index()
print("Max pickups per service point:")
print(max_pickup.head())

# Merge capacity using the same clean integer key
max_pickup['sp_id_int'] = pd.to_numeric(max_pickup['sp_id'], errors='coerce').astype('Int64')

service_points = service_points.merge(
    max_pickup[['sp_id_int', 'pickups']],
    on='sp_id_int',
    how='left',
    validate='one_to_one'
).rename(columns={'pickups': 'capacity'})

# Initialize the closed_cost column
service_points['closed_cost'] = 0.0

#test head of service points capacity
print("Service Points DataFrame with capacity:")
print(service_points.head())

#convert demand for closed stores
def convert_demand_for_closed_stores(service_points, distance_matrix):
    # if a service point is closed, attribute its demand to the nearest open service point
    # Store original deliveries and pickups in base columns
    service_points['base_deliveries'] = service_points['total_deliveries']
    service_points['base_pickups'] = service_points['total_pickups']
    service_points['total_deliveries'] = 0.0
    service_points['total_pickups'] = 0.0

    for i, sp in service_points.iterrows():
        if not sp['opened']:
            # Find the nearest open service point
            distances = distance_matrix.loc[sp['sp_id_int']]
            open_service_points = service_points[service_points['opened']]
            if open_service_points.empty:
                continue
            nearest_id = distances[open_service_points['sp_id_int'].values].idxmin()
            # Map sp_id_int back to the positional index in the original DataFrame
            nearest_row_idx = service_points[service_points["sp_id_int"] == nearest_id].index[0]
            at_home = 0
            # if the distance between them is over 4km transfer deliveries and pickups both to the nearest open service point deliveries
            if distances[nearest_id] > 4000:
                service_points.at[nearest_row_idx, 'total_deliveries'] +=  sp['base_deliveries'] +  sp['base_pickups']
            else:
                rate = distance_matrix.at[sp['sp_id_int'], nearest_id] / 4000
                at_home = sp['base_deliveries'] * (1 - rate)
                closed_cost = distances[nearest_id] + (sp['avg_distance_m'] / 1000) * 1.5 * at_home
                service_points.at[i, 'closed_cost'] = closed_cost
                #pickups and deliveries are transferred to the nearest open service point
                service_points.at[nearest_row_idx, 'total_pickups'] +=  sp['base_pickups'] * rate
                # change service point
                #update capacity of the nearest open service point
                service_points.at[nearest_row_idx, 'capacity'] +=  sp['base_pickups'] * rate
        else:
            # For open service points, retain their original deliveries and pickups
            service_points.at[i, 'total_deliveries'] += sp['base_deliveries']
            service_points.at[i, 'total_pickups'] += sp['base_pickups']
    return service_points
    

    
# calculate cost
def calculate_cost(service_point: pd.Series) -> float:
    total_cost = 0.0
    if service_point['opened']:
        total_cost += 50000
        total_cost += service_point['total_deliveries'] * (service_point['avg_distance_m'] / 1000) * 1.5
        total_cost += service_point['capacity'] * 0.1 * 365
    else:
        total_cost += service_point.get('closed_cost', 0.0)
    return total_cost


        
# Simulated Annealing algorithm

#mutate function (create a copy of the service points and change one random service point to closed or open)
def mutate(service_points: pd.DataFrame) -> pd.DataFrame:
    mutated_sp = service_points.copy()
    random_sp = mutated_sp.sample(1).iloc[0]
    
    # Toggle the 'opened' state
    mutated_sp.loc[random_sp.name, 'opened'] = not random_sp['opened']
    
    # Recalculate capacity if the service point is closed using the function convert_demand_for_closed_stores
    if not mutated_sp.loc[random_sp.name, 'opened']:
        # Set capacity to 0 if the service point is closed
        mutated_sp.loc[random_sp.name, 'capacity'] = 0
        # Convert demand for closed stores
        mutated_sp = convert_demand_for_closed_stores(mutated_sp, distance_matrix)
    
    return mutated_sp

# Simulated Annealing function
def simulated_annealing(
    initial_service_points: pd.DataFrame,
    initial_temperature: float = 1000000.0,
    cooling_rate: float = 0.99 ,
    max_iterations: int = 100000
) -> pd.DataFrame:
    current_solution = initial_service_points.copy()
    current_cost = sum(calculate_cost(sp) for _, sp in current_solution.iterrows())
    
    best_solution = current_solution.copy()
    best_cost = current_cost
    
    temperature = initial_temperature
    
    for iteration in range(max_iterations):
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Current Cost: {current_cost}, Best Cost: {best_cost}, Temperature: {temperature}")
        # Mutate the solution
        new_solution = mutate(current_solution)
        new_cost = sum(calculate_cost(sp) for _, sp in new_solution.iterrows())
        
        # Calculate acceptance probability
        if new_cost < current_cost:
            acceptance_probability = 1.0
        else:
            acceptance_probability = math.exp((current_cost - new_cost) / temperature)
        
        # Accept the new solution with a certain probability
        if random.random() < acceptance_probability:
            current_solution = new_solution
            current_cost = new_cost
            
            # Update the best solution found so far
            if current_cost < best_cost:
                best_solution = current_solution.copy()
                best_cost = current_cost
        
        # Cool down the temperature
        temperature = 100000/ (1 + iteration)
        
    return best_solution, best_cost
    
#call the simulated annealing function
best_solution, best_cost = simulated_annealing(service_points)
# Print the best solution and its cost
initial_cost = sum(calculate_cost(sp) for _, sp in service_points.iterrows())

print("Best Solution:")
print(best_solution[['sp_id', 'opened', 'capacity', 'total_deliveries', 'total_pickups']])
print(f"Initial Cost: {initial_cost}")

print(f"Best Cost: {best_cost}")
