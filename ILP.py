#ILP try out:
"""
maastricht_ilp.py
ILP for redesigning the Post&L Maastricht last-mile network
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from imp_data import load_maastricht_data           # local helper library
from pulp import (LpProblem, LpMinimize, LpVariable, LpBinary, LpInteger,
                  lpSum, PULP_CBC_CMD, value)

# ----------------------------------------------------------------------
# 1. Data load
# ----------------------------------------------------------------------
dfs = load_maastricht_data()                        # takes two Excel paths if needed
sp_df      = dfs["service_points"].copy()
nodes_df   = dfs["nodes"].copy()
deliveries = dfs["deliveries"].copy()               # At-home deliveries
pickups    = dfs["pickups"].copy()                  # Parcels already slated for pick-up
                                               
# pick three representative days to keep the demo nimble               
rep_days   = sorted(deliveries["day"].unique())[:3]  
T          = list(range(len(rep_days)))             # 0,1,2 … for PuLP

# ----------------------------------------------------------------------
# 2. Build distance matrix (Euclidean quick-start)
# ----------------------------------------------------------------------
coords_nodes = nodes_df[["node_id", "x_rd", "y_rd"]].set_index("node_id")
coords_sps   = sp_df[["sp_id", "x_rd", "y_rd"]].set_index("sp_id")

# --- Add check and handling for non-finite values in coords_nodes ---
if coords_nodes[['x_rd', 'y_rd']].isnull().values.any():
    print("Warning: NaN values found in node coordinates. Attempting to fill with 0.")
    coords_nodes[['x_rd', 'y_rd']] = coords_nodes[['x_rd', 'y_rd']].fillna(0)

if np.isinf(coords_nodes[['x_rd', 'y_rd']]).values.any():
    print("Warning: Inf values found in node coordinates. Attempting to replace with 0.")
    coords_nodes[['x_rd', 'y_rd']] = coords_nodes[['x_rd', 'y_rd']].replace([np.inf, -np.inf], 0)

# --- Add check and handling for non-finite values in coords_sps ---
if coords_sps[['x_rd', 'y_rd']].isnull().values.any():
    print("Warning: NaN values found in service point coordinates. Attempting to fill with 0.")
    coords_sps[['x_rd', 'y_rd']] = coords_sps[['x_rd', 'y_rd']].fillna(0)

if np.isinf(coords_sps[['x_rd', 'y_rd']]).values.any():
    print("Warning: Inf values found in service point coordinates. Attempting to replace with 0.")
    coords_sps[['x_rd', 'y_rd']] = coords_sps[['x_rd', 'y_rd']].replace([np.inf, -np.inf], 0)


# kd-tree for fast nearest-neighbour distance lookup
from scipy.spatial import cKDTree
tree_data = coords_nodes[['x_rd', 'y_rd']].values 
if not np.all(np.isfinite(tree_data)):
    # This error should ideally not be hit if cleaning above is effective
    raise ValueError("Non-finite values still present in node tree_data after cleaning. Please check data.")
tree = cKDTree(tree_data)

dist_km = pd.DataFrame(index=coords_sps.index, columns=coords_nodes.index, dtype=float)
for row in coords_sps.itertuples(): 
    sp_id = row.Index 
    x = row.x_rd      
    y = row.y_rd      
    # Ensure x and y are finite before querying
    if not (np.isfinite(x) and np.isfinite(y)):
        print(f"Warning: Non-finite coordinates for SP {sp_id}: ({x}, {y}). Skipping distance calculation for this SP or assigning inf.")
        dist_km.loc[sp_id] = np.inf # Or handle as appropriate
        continue 
    dists, _ = tree.query([x, y], k=len(coords_nodes))
    dist_km.loc[sp_id] = dists / 1000.0             # convert to km

# ----------------------------------------------------------------------
# 3. Prepare demand table per (zone, day)
# ----------------------------------------------------------------------
# merge deliveries and pickups into common shape

# --- Debug: Check what columns exist in both DataFrames ---
print("Deliveries DataFrame columns:", deliveries.columns.tolist())
print("Pickups DataFrame columns:", pickups.columns.tolist())
print("Deliveries shape:", deliveries.shape)
print("Pickups shape:", pickups.shape)

# --- Handle different column names for deliveries and pickups ---
# Based on imp_data.py: deliveries has 'parcels', pickups has 'pickups'
if 'parcels' in deliveries.columns:
    deliveries.rename(columns={"parcels": "qty"}, inplace=True)
    deliveries["qty"] = deliveries["qty"].fillna(0)
else:
    print("Warning: 'parcels' column not found in deliveries DataFrame")
    if deliveries.empty:
        deliveries = pd.DataFrame(columns=['qty', 'day', 'sp_id'])
    elif 'qty' not in deliveries.columns:
        deliveries['qty'] = 0

# Note: based on imp_data.py, pickups has 'pickups' column, not 'parcels'
if 'pickups' in pickups.columns:
    pickups.rename(columns={"pickups": "qty"}, inplace=True)
    pickups["qty"] = pickups["qty"].fillna(0)
elif 'parcels' in pickups.columns:  # fallback in case the structure changes
    pickups.rename(columns={"parcels": "qty"}, inplace=True)
    pickups["qty"] = pickups["qty"].fillna(0)
else:
    print("Warning: Neither 'pickups' nor 'parcels' column found in pickups DataFrame")
    if pickups.empty:
        pickups = pd.DataFrame(columns=['qty', 'day', 'sp_id'])
    elif 'qty' not in pickups.columns:
        pickups['qty'] = 0

deliveries["type"] = "home"
pickups["type"] = "pick"

demand = pd.concat([deliveries, pickups], ignore_index=True)
demand = demand[demand["day"].isin(rep_days)]

# Filter demand to only include sp_ids that exist in our distance matrix
# This fixes the KeyError issues
available_sp_ids = set(dist_km.index.astype(str))  # SP IDs that have distance data
available_node_ids = set(dist_km.columns.astype(str))  # Node IDs that have distance data

print(f"Available SP IDs in distance matrix: {len(available_sp_ids)}")
print(f"Sample available SP IDs: {list(available_sp_ids)[:10]}")
print(f"Unique SP IDs in demand before filtering: {demand['sp_id'].nunique()}")
print(f"Sample demand SP IDs: {demand['sp_id'].unique()[:10].tolist()}")

# Convert sp_id to string for consistent comparison
demand["sp_id"] = demand["sp_id"].astype(str)
demand = demand[demand["sp_id"].isin(available_sp_ids)]
print(f"Unique SP IDs in demand after filtering: {demand['sp_id'].nunique()}")

demand["j"] = demand["sp_id"]  # treat sp_id as demand zone

# --- Ensure 'qty' is integer type after potential NaN filling and before loop ---
demand["qty"] = demand["qty"].astype(int)

# Update J and I to only include sp_ids that exist in both demand and distance matrix
J = demand["j"].unique().tolist()
I = [str(sp_id) for sp_id in sp_df["sp_id"].tolist() if str(sp_id) in available_sp_ids]

print(f"Final I (service points): {len(I)}")
print(f"Final J (demand zones): {len(J)}")
print(f"Sample I: {I[:5]}")
print(f"Sample J: {J[:5]}")

# pivot to qty[j,t,kind]
qty = defaultdict(lambda: defaultdict(lambda: {"home": 0, "pick": 0}))
for _, row in demand.iterrows():
    t = rep_days.index(row["day"])
    qty[row["j"]][t][row["type"]] += int(row["qty"])

# ----------------------------------------------------------------------
# 4. Static parameters (tweak as needed)
# ----------------------------------------------------------------------
F_FIXED          = 50_000                          # €/yr per service point
C_CAP_PER_UNIT   = 0.10                            # €/parcel day capacity
C_KM             = 1.5                             # €/km for at-home leg
RHO_CITY, RHO_SP = 0.01, 0.02

def alpha_walk(d_m):
    # piecewise definition 200 m to 2 km
    if d_m <= 200:
        return 0.8
    if d_m >= 2000:
        return 0.0
    return 0.8 * (1 - (d_m - 200) / 1800)

# Build alpha dictionary only for valid (i,j) pairs
alpha = {}
for i in I:
    for j in J:
        if i in dist_km.index and j in dist_km.columns:
            try:
                distance_km = dist_km.at[i, j]
                if pd.notna(distance_km) and np.isfinite(distance_km):
                    alpha[(i, j)] = alpha_walk(distance_km * 1000)  # convert to meters
                else:
                    alpha[(i, j)] = 0.0  # if distance is invalid, no walking
            except (KeyError, IndexError):
                print(f"Warning: Could not find distance for SP {i} to zone {j}")
                alpha[(i, j)] = 0.0
        else:
            alpha[(i, j)] = 0.0

print(f"Built alpha dictionary with {len(alpha)} entries")

# ----------------------------------------------------------------------
# 5. Create the PuLP model
# ----------------------------------------------------------------------
mdl = LpProblem("Maastricht_SP_ILP", LpMinimize)

# Decision variables
y  = LpVariable.dicts("open", I, 0, 1, LpBinary)
k  = LpVariable.dicts("capacity", I, lowBound=0, cat=LpInteger)
a  = LpVariable.dicts("assign", (I, J), 0, 1, LpBinary)
x  = LpVariable.dicts("pickup", (I, J, T), lowBound=0)
u  = LpVariable.dicts("home",   (I, J, T), lowBound=0)
b  = LpVariable.dicts("bounce", (I, T),    lowBound=0)

# Objective
mdl += lpSum(F_FIXED * y[i] + 365 * C_CAP_PER_UNIT * k[i] for i in I) + \
       lpSum(C_KM * dist_km.at[i, j] * u[i][j][t]
              for i in I for j in J for t in T)

# 1 zone assignment
for j in J:
    mdl += lpSum(a[i][j] for i in I) == 1
for i in I: 
    for j in J:
        mdl += a[i][j] <= y[i]

# 2 demand split
for i in I:
    for j in J:
        for t in T:
            d_home = qty[j][t]["home"]
            d_pick = qty[j][t]["pick"]
            total  = d_home + d_pick
            if total == 0:
                continue
            mdl += x[i][j][t] >= alpha[i, j] * total * a[i][j]    # lower bound
            mdl += u[i][j][t] >= (1 - alpha[i, j]) * total * a[i][j]

# 2.1 demand satisfaction constraint - ensure all demand is captured
for j in J:
    for t in T:
        d_home = qty[j][t]["home"]
        d_pick = qty[j][t]["pick"]
        total = d_home + d_pick
        if total > 0:
            mdl += lpSum(x[i][j][t] + u[i][j][t] for i in I) == total

# 3 capacity per day
for i in I: # Corrected loop structure
    for t in T: # Corrected loop structure
        mdl += lpSum(x[i][j][t] for j in J) <= k[i] - b[i][t]

# 4 bounce KPIs
mdl += lpSum(b[i][t] for i in I for t in T) <= RHO_CITY * \
       lpSum(qty[j][t]["pick"] + qty[j][t]["home"]
             for j in J for t in T)
for i in I:
    mdl += lpSum(b[i][t] for t in T) <= RHO_SP * \
           lpSum(x[i][j][t] for j in J for t in T)

# ----------------------------------------------------------------------
# 6. Solve
# ----------------------------------------------------------------------
print("Solving…")
mdl.solve(PULP_CBC_CMD(msg=True))

print("\n----- results -----")
print(f"total cost   {value(mdl.objective):,.0f} € per year")
print(f"open points  {[i for i in I if y[i].value() > 0.5]}")
print("capacity per point")
for i in I:
    if y[i].value() > 0.5:
        print(f"  {i:>4}: {int(k[i].value())} parcels day")