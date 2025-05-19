import pandas as pd
from pathlib import Path
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
from pyproj import Transformer
from shapely.geometry import LineString




#found node 3921 at coords 50.878244, 5.691430
# and found node 7859 at coords 50.866568, 5.704254

file = Path("data_Maastricht_2024.xlsx")          # relative path in your project
xls  = pd.ExcelFile(file)                         # keeps the file open only once

print("Sheets:", xls.sheet_names)

# Helper to convert any column header to snake_case
def snake(col):
    """
    Convert a column label to snake_case.
    If the label is not a string (e.g. an int date column like 20240101) leave it unchanged.
    """
    if not isinstance(col, str):
        return col
    return (
        col.strip()           # trim white‑space
           .replace("-", "_") # replace dashes
           .replace(" ", "_") # replace blanks
           .lower()           # lower‑case
    )

# ---------------------------------------------------------------------------
# 1. Nodes
nodes_df = pd.read_excel(xls, sheet_name="Nodes")
nodes_df.rename(columns=lambda c: snake(c), inplace=True)

# Guarantee predictable names + dtypes
nodes_df.rename(
    columns={
        "node_id": "node_id",
        # accept either original 'X'/'Y' or the lowercase 'x'/'y'
        "X": "x_rd",
        "x": "x_rd",
        "Y": "y_rd",
        "y": "y_rd",
        "square": "cbs_square",
    },
    inplace=True,
)
nodes_df = nodes_df.astype(
    {
        "node_id": "int32",
        "x_rd": "float64",
        "y_rd": "float64",
        "cbs_square": "string",
    }
)

# ---------------------------------------------------------------------------
# Compute scale and offset using TWO reference nodes so both position and
# scale are correct.
# Reference nodes: 3921 (50.878244, 5.691430) and 7859 (50.866568, 5.704254)
refs = {
    3921: (5.691430, 50.878244),   # lon, lat
    7859: (5.704254, 50.866568),
}

transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)

# Convert reference lon/lat to RD‑New metres
ref_rd = {nid: transformer.transform(lon, lat) for nid, (lon, lat) in refs.items()}

# Fetch local coordinates from dataset
local_ref = {
    nid: nodes_df.loc[nodes_df.node_id == nid, ["x_rd", "y_rd"]].iloc[0]
    for nid in refs
}

# Use the two points to solve for scale and offset
n1, n2 = 3921, 7859
x_real1, y_real1 = ref_rd[n1]
x_real2, y_real2 = ref_rd[n2]
x_loc1,  y_loc1  = local_ref[n1]
x_loc2,  y_loc2  = local_ref[n2]

scale_x = (x_real2 - x_real1) / (x_loc2 - x_loc1)
scale_y = (y_real2 - y_real1) / (y_loc2 - y_loc1)

dx = x_real1 - scale_x * x_loc1
dy = y_real1 - scale_y * y_loc1

print(f"Scale factors:  X={scale_x:.6f}  Y={scale_y:.6f}")
print(f"Offset (m):    dx={dx:.1f}  dy={dy:.1f}")

# Apply scale and offset to nodes
nodes_df["x_rd_abs"] = nodes_df.x_rd * scale_x + dx
nodes_df["y_rd_abs"] = nodes_df.y_rd * scale_y + dy

# ---------------------------------------------------------------------------
# 2. Edges
edges_df = pd.read_excel(xls, sheet_name="Edges")
edges_df.rename(columns=lambda c: snake(c), inplace=True)

# Friendly names where we know them
edges_df.rename(
    columns={
        "v1": "from_node",
        "v2": "to_node",
        "dist": "length_m",
        "max_speed": "speed_kmh",
        "name": "road_name"
    },
    inplace=True,
)

# Safe dtype coercion
dtype_edges = {
    "from_node": "int32",
    "to_node": "int32",
    "length_m": "float64",
    "speed_kmh": "float32",
}
for col, dt in dtype_edges.items():
    if col in edges_df.columns:
        edges_df[col] = edges_df[col].astype(dt)

# ---------------------------------------------------------------------------
# 3. CBS 500 m grid squares
cbs_df = pd.read_excel(xls, sheet_name="CBS Squares")
cbs_df.rename(columns=lambda c: snake(c), inplace=True)
if "square" in cbs_df.columns:
    cbs_df.rename(columns={"square": "cbs_square"}, inplace=True)
cbs_df["cbs_square"] = cbs_df["cbs_square"].astype("string")

# ---------------------------------------------------------------------------
# 4. Service‑point locations
sp_df = pd.read_excel(xls, sheet_name="Service Point Locations")
sp_df.rename(columns=lambda c: snake(c), inplace=True)
# harmonise service‑point ID column
if "sp_id" not in sp_df.columns and "location_id" in sp_df.columns:
    sp_df.rename(columns={"location_id": "sp_id"}, inplace=True)
sp_df.rename(
    columns={
        "location_id": "sp_id",
        "X": "x_rd",
        "x": "x_rd",
        "Y": "y_rd",
        "y": "y_rd",
        "population": "population",
        "total_deliveries": "total_deliveries",
        "total_pickups": "total_pickups",
    },
    inplace=True,
)
dtype_sp = {"sp_id": "int32", "x_rd": "float64", "y_rd": "float64"}
for col, dt in dtype_sp.items():
    if col in sp_df.columns:
        sp_df[col] = sp_df[col].astype(dt)

# apply the previously calculated offset and scale to service‑point coordinates
sp_df["x_rd_abs"] = sp_df.x_rd * scale_x + dx
sp_df["y_rd_abs"] = sp_df.y_rd * scale_y + dy

# ---------------------------------------------------------------------------
# Add customer‑per‑package ratios
sp_df["cust_per_pickup"]   = sp_df["population"] / sp_df["total_pickups"].replace(0, pd.NA)
sp_df["cust_per_delivery"] = sp_df["population"] / sp_df["total_deliveries"].replace(0, pd.NA)

# ---------------------------------------------------------------------------
# 5. 2024 at‑home deliveries (daily per square)
deliveries_df = pd.read_excel(xls, sheet_name="At-home Deliveries")
deliveries_df.rename(columns=lambda c: snake(c), inplace=True)

# ---------------------------------------------------------------------------
# 6. 2024 pick‑ups at service points (daily)
pickups_df = pd.read_excel(xls, sheet_name="Service Point Locations")
pickups_df.rename(columns=lambda c: snake(c), inplace=True)

# Quick confirmation prints
print(
    f"nodes_df {nodes_df.shape} | edges_df {edges_df.shape} | cbs_df {cbs_df.shape} | "
    f"sp_df {sp_df.shape} | deliveries_df {deliveries_df.shape} | pickups_df {pickups_df.shape}"
)



    
# ---------------------------------------------------------------------------
# Plot all nodes on top of an OpenStreetMap basemap
node_gdf = gpd.GeoDataFrame(
    nodes_df,
    geometry=gpd.points_from_xy(nodes_df.x_rd_abs, nodes_df.y_rd_abs),
    crs="EPSG:28992",
)

# Re‑project to Web Mercator for the tile server
node_web = node_gdf.to_crs(3857)

 # --- Highlight service‑point nodes ---
if "sp_id" not in sp_df.columns:
    if "id" in sp_df.columns:
        sp_df.rename(columns={"id": "sp_id"}, inplace=True)
    elif "node_id" in sp_df.columns:
        sp_df.rename(columns={"node_id": "sp_id"}, inplace=True)
    elif "location_id" in sp_df.columns:
        sp_df.rename(columns={"location_id": "sp_id"}, inplace=True)
    else:
        raise KeyError("Service‑point ID column not found in Service Point Locations sheet")
highlight_ids = sp_df["sp_id"].astype("int64").unique()

node_highlight = node_web[node_web["node_id"].isin(highlight_ids)].copy()
node_rest      = node_web[~node_web["node_id"].isin(highlight_ids)]

# attach pickup vs delivery percentage and total volume
sp_df["pickup_vs_delivery_pct"] = sp_df["total_pickups"] / (sp_df["total_pickups"] + sp_df["total_deliveries"])
sp_df["total_volume"] = sp_df["total_pickups"] + sp_df["total_deliveries"]

node_highlight = node_highlight.merge(
    sp_df[["sp_id", "pickup_vs_delivery_pct", "total_volume"]],
    left_on="node_id",
    right_on="sp_id",
    how="left",
)

size_min, size_max = 100, 1000
vol_min = node_highlight["total_volume"].min()
vol_max = node_highlight["total_volume"].max()
node_highlight["dot_size"] = size_min + (
    (node_highlight["total_volume"] - vol_min) / (vol_max - vol_min)
) * (size_max - size_min)

# --- Build edge geometries ---
edge_lines = []

# detect column names for start/end nodes
from_col = next((c for c in ["from_node", "from", "source", "start", "start_node"] if c in edges_df.columns), None)
to_col   = next((c for c in ["to_node", "to", "target", "end", "end_node"]        if c in edges_df.columns), None)
if from_col is None or to_col is None:
    raise KeyError("Could not find start/end node columns in Edges sheet")

for _, row in edges_df.iterrows():
    from_id = row[from_col]
    to_id   = row[to_col]

    from_pt = nodes_df.loc[nodes_df.node_id == from_id, ["x_rd_abs", "y_rd_abs"]].iloc[0]
    to_pt   = nodes_df.loc[nodes_df.node_id == to_id,   ["x_rd_abs", "y_rd_abs"]].iloc[0]

    edge_lines.append(
        LineString(
            [(from_pt.x_rd_abs, from_pt.y_rd_abs), (to_pt.x_rd_abs, to_pt.y_rd_abs)]
        )
    )
edges_gdf = gpd.GeoDataFrame(edges_df, geometry=edge_lines, crs="EPSG:28992")
edges_web = edges_gdf.to_crs(3857)

fig, ax = plt.subplots(figsize=(9, 9))



 # overlay service‑point nodes
node_highlight.plot(
    ax=ax,
    column="pickup_vs_delivery_pct",
    cmap="RdBu",
    markersize=node_highlight["dot_size"],
    edgecolor="white",
    linewidth=0.5,
    alpha=0.8,
    zorder=4,
)

# --- Manual legend for point sizes ---
from matplotlib.lines import Line2D

# Create a manual legend for point sizes
size_legend = [100, 500, 1000]  # sample volumes
handles = [
    Line2D([0], [0], marker='o', color='w', label=f'{s} parcels',
           markersize=(size_min + (s - vol_min) / (vol_max - vol_min) * (size_max - size_min))**0.5,
           markerfacecolor='gray', alpha=0.5)
    for s in size_legend
]
ax.legend(handles=handles, title="Total Volume", loc="lower right")


cx.add_basemap(ax, crs=3857, source=cx.providers.CartoDB.Positron)

# --- Add colorbar with label ---
sm = plt.cm.ScalarMappable(cmap="RdBu", norm=plt.Normalize(vmin=0, vmax=1))
sm._A = []
cbar = plt.colorbar(sm, ax=ax, orientation="vertical", shrink=0.7)
cbar.set_label("Pickup Ratio")

ax.set_title(
    "Pickup vs Delivery Percentage by Service Point\n(Color = Pickup %, Size = Total Volume)",
    fontsize=12
)
ax.set_axis_off()
plt.tight_layout()
plt.savefig("pickup_vs_delivery_percentage_map.png", dpi=300)
print("Saved map with pickup vs delivery percentages to pickup_vs_delivery_percentage_map.png")