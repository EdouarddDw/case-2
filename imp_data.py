#import data
import pandas as pd
from pathlib import Path
import geopandas as gpd
import contextily as cx

"""
Data‑loading utilities for the Maastricht last‑mile case study.

Usage
-----
>>> from case2_data import load_maastricht_data
>>> data = load_maastricht_data()
>>> nodes = data["nodes"]
"""
__all__ = ["load_maastricht_data"]

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

def load_maastricht_data(
    xlsx_path: str = "data_Maastricht_2024.xlsx",
    cbs_path: str = "CBS_Squares_cleaned.xlsx",
) -> dict:
    """
    Load all core dataframes required for the Maastricht last‑mile logistics case.

    Parameters
    ----------
    xlsx_path : str
        Path to the Excel workbook containing Nodes, Edges, Service Point Locations and demand sheets.
    cbs_path : str
        Path to the cleaned CBS squares spreadsheet.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary keyed by:
        'nodes', 'edges', 'cbs', 'service_points', 'deliveries', 'pickups'
    """
    file = Path(xlsx_path)          # relative path in your project
    xls  = pd.ExcelFile(file)                         # keeps the file open only once

    print("Sheets:", xls.sheet_names)

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
    cbs_df = pd.read_excel(cbs_path, header=1)
    cbs_df.rename(columns=lambda c: snake(c), inplace=True)
    if "square" in cbs_df.columns:
        cbs_df.rename(columns={"square": "cbs_square"}, inplace=True)
    elif "cbs_imputed_knn_prioritized" in cbs_df.columns:
        cbs_df.rename(columns={"cbs_imputed_knn_prioritized": "cbs_square"}, inplace=True)


    cbs_df["cbs_square"] = cbs_df["cbs_square"].astype("string")

    # Harmonize coordinate columns for CBS squares
    if "rd_x" in cbs_df.columns and "rd_y" in cbs_df.columns:
        cbs_df.rename(columns={"rd_x": "x_rd", "rd_y": "y_rd"}, inplace=True)
    elif "x" in cbs_df.columns and "y" in cbs_df.columns:
        cbs_df.rename(columns={"x": "x_rd", "y": "y_rd"}, inplace=True)
    else:
        raise KeyError(
            "CBS squares dataframe missing coordinate columns. "
            "Expected 'rd_x'/'rd_y' or 'x'/'y'."
        )

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

    # Ensure service‑point IDs are treated consistently as strings
    sp_df["sp_id"] = sp_df["sp_id"].astype(str)

    deliveries_raw = pd.read_excel(xls, sheet_name="At-home Deliveries")
    deliveries_df = deliveries_raw.melt(id_vars=deliveries_raw.columns[0], var_name="sp_id", value_name="deliveries")
    deliveries_df.rename(columns={deliveries_df.columns[0]: "day"}, inplace=True)
    deliveries_df["sp_id"] = deliveries_df["sp_id"].astype(str)

    pickups_raw = pd.read_excel(xls, sheet_name="Service Point Parcels Picked Up")
    pickups_df = pickups_raw.melt(id_vars=pickups_raw.columns[0], var_name="sp_id", value_name="pickups")
    pickups_df.rename(columns={pickups_df.columns[0]: "day"}, inplace=True)
    pickups_df["sp_id"] = pickups_df["sp_id"].astype(str)

    # Build square-to-square distance lookup
    edges_df.rename(columns=lambda c: snake(c), inplace=True)

    return {
        "nodes": nodes_df,
        "edges": edges_df,
        "cbs": cbs_df,
        "service_points": sp_df,
        "deliveries": deliveries_df,
        "pickups": pickups_df,
    }

if __name__ == "__main__":
    data = load_maastricht_data()
    print("Loaded dataframes:", list(data.keys()))