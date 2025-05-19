import pandas as pd
from pathlib import Path
import geopandas as gpd
import contextily as cx
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from pyproj import Transformer
from shapely.geometry import LineString, box




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
           .replace(".", "_") # replace dots
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
        "x": "x_rd",
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
cbs_df = pd.read_excel("CBS_Squares_cleaned.xlsx", header=1)
cbs_df.rename(columns=lambda c: snake(c), inplace=True)
print("Renamed CBS columns:", cbs_df.columns.tolist())
if "square" in cbs_df.columns:
    cbs_df.rename(columns={"square": "cbs_square"}, inplace=True)
elif "cbs_imputed_knn_prioritized" in cbs_df.columns:
    cbs_df.rename(columns={"cbs_imputed_knn_prioritized": "cbs_square"}, inplace=True)

cbs_df["cbs_square"] = cbs_df["cbs_square"].astype("string")

# ---------------------------------------------------------------------------
# Convert CBS local coordinates to absolute RD metres and build 500 m squares
# Rename columns if necessary
if "x_rd" not in cbs_df.columns:
    if "x" in cbs_df.columns:
        cbs_df.rename(columns={"x": "x_rd"}, inplace=True)
    else:
        raise KeyError("'x_rd' or 'x' column not found in CBS data")

if "y_rd" not in cbs_df.columns:
    if "y" in cbs_df.columns:
        cbs_df.rename(columns={"y": "y_rd"}, inplace=True)
    else:
        raise KeyError("'y_rd' or 'y' column not found in CBS data")

cbs_df = cbs_df.astype({"x_rd": "float64", "y_rd": "float64"})

# Apply same scale/offset as for nodes
cbs_df["x_rd_abs"] = cbs_df.x_rd * scale_x + dx
cbs_df["y_rd_abs"] = cbs_df.y_rd * scale_y + dy

# Build 500 m × 500 m square polygons centred on each grid‑square centroid
cbs_df["geometry"] = cbs_df.apply(
    lambda r: box(r.x_rd_abs - 250, r.y_rd_abs - 250, r.x_rd_abs + 250, r.y_rd_abs + 250),
    axis=1,
)
cbs_gdf = gpd.GeoDataFrame(cbs_df, geometry="geometry", crs="EPSG:28992")
cbs_web = cbs_gdf.to_crs(3857)


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
        "name": "sp_name",
        "x": "x_rd",
        "y": "y_rd",
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

node_highlight = node_web[node_web["node_id"].isin(highlight_ids)]
node_rest = node_web[~node_web["node_id"].isin(highlight_ids)]

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



#
# --- Income Stratification Map ---
# Re-apply snake_case renaming to ensure columns are correct
cbs_df.rename(columns=lambda c: snake(c), inplace=True)

# Pull these columns from cbs_df before converting to GeoDataFrame
low_income = cbs_df["percentage_low_income_households"]
high_income = cbs_df["percentage_high_income_households"]

# Normalize them
low_income_norm = (low_income - low_income.min()) / (low_income.max() - low_income.min())
high_income_norm = (high_income - high_income.min()) / (high_income.max() - high_income.min())

 # Build a scalar 0‑1 value that blends low‑ vs high‑income
color_values = (high_income_norm - low_income_norm + 1) / 2  # 0 = red, 1 = blue

  # Prepare colour bar (0 = red, 1 = blue)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
sm   = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu, norm=norm)
sm.set_array([])

 # Map each scalar to a HEX colour via the RdYlBu colormap
cbs_web["color"] = color_values.apply(lambda v: mcolors.to_hex(plt.cm.RdYlBu(v)))

# Plot with custom color blend
cbs_web.plot(
    ax=ax,
    color=cbs_web["color"],
    linewidth=0,
    alpha=0.7,
    zorder=0,
)


# overlay service‑point nodes
node_highlight.plot(
    ax=ax,
    markersize=50,
    color="crimson",
    edgecolor="white",
    linewidth=0.5,
    zorder=4,
)

# Draw the basemap *under* all vector layers
cx.add_basemap(ax, crs=3857, source=cx.providers.CartoDB.Positron, zorder=-3)

  # Add colour bar legend
cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Low income   ←   →   High income", fontsize=8)

  # Add legend entry for service points
ax.scatter([], [], color="crimson", edgecolor="white", linewidth=0.5, s=50, label="Service point")
ax.legend(loc="lower left", fontsize=8)

ax.set_title("Income Stratification by CBS Square\n(Red = Low Income, Blue = High Income)", fontsize=12)
ax.set_axis_off()
plt.tight_layout()
plt.savefig("income_stratification_map.png", dpi=300)
print("Saved map with income stratification to income_stratification_map.png")