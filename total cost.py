import pandas as pd
import numpy as np
# from imp_data import load_maastricht_data # Only needed if recalculating fixed costs from scratch
import json

# --- 1. Configuration ---
print("Step 1: Configuration")
FIXED_COSTS_CSV = 'service_point_yearly_costs.csv'
PARCEL_ALLOCATIONS_PARQUET = 'all_days_parcel_allocations.parquet'
OUTPUT_TOTAL_COSTS_CSV = 'total_yearly_costs_per_sp.csv'
OUTPUT_COST_BREAKDOWN_PLOT = 'sp_cost_breakdown_plot.png'

COST_PER_KM_EURO_DELIVERY = 1.5 # Variable delivery cost rate
BASE_FIXED_COST_SP = 50000     # Base fixed cost for an SP per year
STORAGE_COST_PER_PACKAGE_DAY = 0.10 # Storage cost rule

# --- 2. Load Data ---
print("\nStep 2: Loading data")
try:
    parcel_allocations_df = pd.read_parquet(PARCEL_ALLOCATIONS_PARQUET)
    print(f"  Loaded parcel allocations from '{PARCEL_ALLOCATIONS_PARQUET}' ({len(parcel_allocations_df)} rows).")
except FileNotFoundError:
    print(f"ERROR: Parcel allocations file not found: '{PARCEL_ALLOCATIONS_PARQUET}'. This file is required.")
    exit()
except Exception as e:
    print(f"ERROR: Could not read '{PARCEL_ALLOCATIONS_PARQUET}': {e}")
    exit()

# --- 3. Calculate/Load Fixed Costs ---
print("\nStep 3: Loading or Calculating Fixed Costs per Service Point")
fixed_costs_df = None
try:
    fixed_costs_df = pd.read_csv(FIXED_COSTS_CSV)
    if 'service_point_id' in fixed_costs_df.columns:
        fixed_costs_df.set_index('service_point_id', inplace=True)
    else: # Try to find a suitable column if 'service_point_id' is not the name
        potential_id_cols = [col for col in fixed_costs_df.columns if 'sp_id' in col.lower() or 'id' in col.lower()]
        if potential_id_cols:
            fixed_costs_df.set_index(potential_id_cols[0], inplace=True)
            print(f"  Used '{potential_id_cols[0]}' as index for fixed costs.")
        else:
            raise ValueError("No suitable 'service_point_id' or similar column found for index in fixed costs CSV.")

    if 'yearly_cost_€' not in fixed_costs_df.columns:
        raise ValueError("'yearly_cost_€' column missing in fixed costs CSV.")
    fixed_costs_df = fixed_costs_df[['yearly_cost_€']].copy()
    fixed_costs_df.rename(columns={'yearly_cost_€': 'fixed_yearly_cost'}, inplace=True)
    fixed_costs_df.index = fixed_costs_df.index.astype(str)
    print(f"  Loaded fixed costs from '{FIXED_COSTS_CSV}' for {len(fixed_costs_df)} service points.")
except FileNotFoundError:
    print(f"WARNING: Fixed costs file '{FIXED_COSTS_CSV}' not found.")
    print("         Fixed costs will be missing. For a complete analysis, generate this file using 'fixed cost calc.py' or similar.")
    fixed_costs_df = pd.DataFrame(columns=['fixed_yearly_cost']) # Empty df with correct column
    fixed_costs_df.index.name = 'service_point_id' # Set index name
except ValueError as e:
    print(f"ERROR: Issue with '{FIXED_COSTS_CSV}': {e}")
    exit()
except Exception as e:
    print(f"ERROR: Could not read '{FIXED_COSTS_CSV}': {e}")
    exit()


# --- 4. Calculate Variable Delivery Costs ---
print("\nStep 4: Calculating Variable Yearly Delivery Costs per Service Point")
if 'distance_m' not in parcel_allocations_df.columns or 'sp_id' not in parcel_allocations_df.columns:
    print("ERROR: 'distance_m' or 'sp_id' column missing in parcel allocations data. Cannot calculate variable costs.")
    exit()

valid_allocations = parcel_allocations_df[
    np.isfinite(parcel_allocations_df['distance_m']) & (parcel_allocations_df['distance_m'] >= 0)
].copy()

if valid_allocations.empty:
    print("  No valid parcel allocations with finite distances found. Variable delivery costs will be zero.")
    variable_costs_per_sp_df = pd.DataFrame(columns=['variable_yearly_cost'])
    variable_costs_per_sp_df.index.name = 'service_point_id'
else:
    valid_allocations['distance_km'] = valid_allocations['distance_m'] / 1000.0
    valid_allocations['parcel_delivery_cost'] = COST_PER_KM_EURO_DELIVERY * valid_allocations['distance_km']
    valid_allocations['sp_id'] = valid_allocations['sp_id'].astype(str)
    total_variable_cost_sim_period_per_sp = valid_allocations.groupby('sp_id')['parcel_delivery_cost'].sum()
    
    num_simulated_days = 0
    if 'day' in valid_allocations.columns:
        num_simulated_days = valid_allocations['day'].nunique()
    
    if num_simulated_days == 0: # Attempt to load from source if 'day' was missing or yielded 0 unique days
        print("  Attempting to determine num_simulated_days from source deliveries data for extrapolation.")
        try:
            from imp_data import load_maastricht_data
            data_temp = load_maastricht_data()
            deliveries_temp_df = data_temp["deliveries"]
            num_simulated_days = deliveries_temp_df['day'].nunique()
            print(f"  Determined {num_simulated_days} unique simulated days from source deliveries data.")
        except Exception as e_imp:
            num_simulated_days = 0 
            print(f"  Could not determine num_simulated_days from source ({e_imp}). Extrapolation might be inaccurate.")

    if num_simulated_days > 0:
        average_daily_variable_cost_per_sp = total_variable_cost_sim_period_per_sp / num_simulated_days
        variable_yearly_cost_per_sp = average_daily_variable_cost_per_sp * 365
    else:
        print("  Warning: Number of simulated days is 0 or unknown. Using sum from period as 'yearly' variable cost (may be inaccurate).")
        variable_yearly_cost_per_sp = total_variable_cost_sim_period_per_sp

    variable_costs_per_sp_df = variable_yearly_cost_per_sp.reset_index()
    variable_costs_per_sp_df.rename(columns={'parcel_delivery_cost': 'variable_yearly_cost', 'sp_id': 'service_point_id'}, inplace=True)
    variable_costs_per_sp_df.set_index('service_point_id', inplace=True)
    print(f"  Calculated variable yearly delivery costs for {len(variable_costs_per_sp_df)} service points.")

# --- 4.5 Calculate Storage Costs ---
print("\nStep 4.5: Calculating Yearly Storage Costs per Service Point")
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
        storage_costs_per_sp_df.rename(columns={'num_daily_parcels': 'storage_yearly_cost', 'sp_id': 'service_point_id'}, inplace=True)
        storage_costs_per_sp_df.set_index('service_point_id', inplace=True)
        # Ensure index is string
        storage_costs_per_sp_df.index = storage_costs_per_sp_df.index.astype(str)
        print(f"  Calculated yearly storage costs for {len(storage_costs_per_sp_df)} service points.")

# --- 5. Combine Costs ---
print("\nStep 5: Combining fixed, variable, and storage costs")
total_costs_df = fixed_costs_df.copy()
total_costs_df = total_costs_df.join(variable_costs_per_sp_df, how='outer')
total_costs_df = total_costs_df.join(storage_costs_per_sp_df, how='outer')

total_costs_df['fixed_yearly_cost'].fillna(0, inplace=True)
total_costs_df['variable_yearly_cost'].fillna(0, inplace=True)
total_costs_df['storage_yearly_cost'].fillna(0, inplace=True)
print(f"  Combined costs for {len(total_costs_df)} unique service points.")

# --- 6. Calculate Total Costs ---
print("\nStep 6: Calculating total costs")
total_costs_df['total_yearly_cost'] = (
    total_costs_df['fixed_yearly_cost'] +
    total_costs_df['variable_yearly_cost'] +
    total_costs_df['storage_yearly_cost']
)
total_costs_df.index.name = 'service_point_id'

# --- 7. Output ---
print("\nStep 7: Outputting results")
print("\nDataFrame of Total Yearly Costs per Service Point (including storage):")
# Display relevant columns
display_cols = ['fixed_yearly_cost', 'variable_yearly_cost', 'storage_yearly_cost', 'total_yearly_cost']
print(total_costs_df[display_cols].head().to_string(float_format="%.2f"))


try:
    total_costs_df.reset_index().to_csv(OUTPUT_TOTAL_COSTS_CSV, index=False, float_format='%.2f')
    print(f"\n  Saved combined costs to '{OUTPUT_TOTAL_COSTS_CSV}'")
except Exception as e:
    print(f"ERROR: Could not save total costs CSV: {e}")

overall_fixed_cost = total_costs_df['fixed_yearly_cost'].sum()
overall_variable_cost = total_costs_df['variable_yearly_cost'].sum()
overall_storage_cost = total_costs_df['storage_yearly_cost'].sum()
overall_total_cost = total_costs_df['total_yearly_cost'].sum()

print("\nOverall Yearly Cost Summary (including storage):")
print(f"  Total Fixed Costs: €{overall_fixed_cost:,.2f}")
print(f"  Total Variable Delivery Costs: €{overall_variable_cost:,.2f}")
print(f"  Total Storage Costs: €{overall_storage_cost:,.2f}")
print(f"  Grand Total Yearly Costs: €{overall_total_cost:,.2f}")

# --- 8. Visualization (Optional) ---
print("\nStep 8: Generating cost breakdown visualization (optional)")
try:
    import matplotlib.pyplot as plt
    
    top_n_sp_plot = 20
    plot_df = total_costs_df.sort_values(by='total_yearly_cost', ascending=False).head(top_n_sp_plot)

    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(18, 10))
        # Define colors for the stacks
        colors = ['#4A90E2', '#F5A623', '#7ED321'] # Blue (fixed), Orange (variable), Green (storage)
        
        plot_df[['fixed_yearly_cost', 'variable_yearly_cost', 'storage_yearly_cost']].plot(
            kind='bar', stacked=True, ax=ax, color=colors
        )
        
        ax.set_title(f'Top {len(plot_df)} Service Points: Yearly Cost Breakdown', fontsize=16, fontweight='bold')
        ax.set_xlabel('Service Point ID', fontsize=12)
        ax.set_ylabel('Total Yearly Cost (€)', fontsize=12)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        plt.setp(ax.get_xticklabels(), ha='right') # Ensure labels are aligned correctly after rotation
        ax.tick_params(axis='y', labelsize=10)
        ax.legend(title='Cost Type', labels=['Fixed', 'Variable Delivery', 'Storage'], fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(OUTPUT_COST_BREAKDOWN_PLOT, dpi=300)
        print(f"  Saved cost breakdown plot to '{OUTPUT_COST_BREAKDOWN_PLOT}'")
        plt.show() 
    else:
        print("  Skipped cost breakdown plot: No data to plot after filtering.")

except ImportError:
    print("  Skipped visualization: matplotlib not found.")
except Exception as e:
    print(f"  Error during visualization: {e}")

print("\nScript finished.")