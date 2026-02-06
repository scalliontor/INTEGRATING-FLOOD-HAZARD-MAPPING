"""
📊 RISK CHANGE ANALYSIS BY DISTRICT (LEVEL 2)
Phân tích thay đổi nguy cơ ngập theo từng huyện/xã giữa Baseline và Kịch bản BĐKH.

Output:
- CSV: Bảng thống kê risk theo từng đơn vị hành chính
- PNG: Bản đồ thể hiện mức tăng/giảm risk (choropleth)
- GeoJSON: File vector với attributes risk đã tính
"""
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os

print("=" * 80)
print("📊 RISK CHANGE ANALYSIS BY DISTRICT")
print("=" * 80)

# CONFIG
BOUNDARY_SHP = '../AOI_level2/HaTinh_Districts_Level2.shp'
SCENARIO_DIR = 'output_scenarios'
OUTPUT_DIR = 'output_scenarios'

# Scenario files
SCENARIOS = {
    "Baseline": "00_Baseline.tif",
    "RCP45_2050": "01_RCP45_2050.tif",
    "RCP85_2050": "02_RCP85_2050.tif",
    "RCP45_2100": "03_RCP45_2100.tif",
    "RCP85_2100": "04_RCP85_2100.tif"
}

# 1. LOAD BOUNDARIES
print("\n[1/4] Loading District Boundaries...")
gdf = gpd.read_file(BOUNDARY_SHP)
print(f"   Found {len(gdf)} districts/communes")
print(f"   Columns: {list(gdf.columns)}")

# Identify name column - ADM2_NAME is the correct column for district names
name_col = 'ADM2_NAME'  # This is the standard column for administrative level 2 names

if name_col is None:
    # Use first non-geometry column
    name_col = [c for c in gdf.columns if c != 'geometry'][0]

print(f"   Using '{name_col}' as district name column")
print(f"   Districts: {list(gdf[name_col].values)}")

# Ensure same CRS as raster
with rasterio.open(f"{SCENARIO_DIR}/{SCENARIOS['Baseline']}") as src:
    raster_crs = src.crs
    
if gdf.crs != raster_crs:
    print(f"   Reprojecting from {gdf.crs} to {raster_crs}...")
    gdf = gdf.to_crs(raster_crs)

# 2. COMPUTE ZONAL STATISTICS
print("\n[2/4] Computing Zonal Statistics for each scenario...")

results = {}
for scenario_name, filename in SCENARIOS.items():
    filepath = f"{SCENARIO_DIR}/{filename}"
    print(f"   Processing: {scenario_name}...")
    
    # Calculate mean, count of each risk level
    stats = zonal_stats(
        gdf, 
        filepath, 
        stats=['mean', 'count', 'sum'],
        categorical=True,
        category_map={1: 'VeryLow', 2: 'Low', 3: 'Moderate', 4: 'High', 5: 'VeryHigh'}
    )
    
    # Extract mean risk level
    means = [s.get('mean', np.nan) for s in stats]
    
    # Calculate High Risk percentage (Level 4 + 5)
    high_risk_pcts = []
    for s in stats:
        total = s.get('count', 0)
        high = s.get('High', 0) + s.get('VeryHigh', 0)
        pct = (high / total * 100) if total > 0 else 0
        high_risk_pcts.append(pct)
    
    results[f"{scenario_name}_Mean"] = means
    results[f"{scenario_name}_HighRisk%"] = high_risk_pcts

# Add to GeoDataFrame
for col, values in results.items():
    gdf[col] = values

# 3. COMPUTE CHANGES
print("\n[3/4] Computing Risk Changes (vs Baseline)...")

# Change in Mean Risk Level
for scenario in ["RCP45_2050", "RCP85_2050", "RCP45_2100", "RCP85_2100"]:
    gdf[f"{scenario}_Change"] = gdf[f"{scenario}_Mean"] - gdf["Baseline_Mean"]
    gdf[f"{scenario}_HighRisk_Change"] = gdf[f"{scenario}_HighRisk%"] - gdf["Baseline_HighRisk%"]

# 4. SAVE RESULTS
print("\n[4/4] Saving Results...")

# Save CSV
csv_path = f"{OUTPUT_DIR}/District_Risk_Analysis.csv"
df_export = gdf.drop(columns=['geometry'])
df_export.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f"   ✅ Saved: {csv_path}")

# Save GeoJSON
geojson_path = f"{OUTPUT_DIR}/District_Risk_Analysis.geojson"
gdf.to_file(geojson_path, driver='GeoJSON')
print(f"   ✅ Saved: {geojson_path}")

# 5. CREATE CHOROPLETH MAPS
print("\n[5/5] Creating Choropleth Maps...")

# Custom colormap: Blue (decrease) -> White (no change) -> Red (increase)
cmap = LinearSegmentedColormap.from_list('risk_change', ['#2166ac', '#f7f7f7', '#b2182b'])

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

scenarios_to_plot = ["RCP45_2050", "RCP85_2050", "RCP45_2100", "RCP85_2100"]
titles = ["RCP 4.5 (2050)", "RCP 8.5 (2050)", "RCP 4.5 (2100)", "RCP 8.5 (2100)"]

vmin = -0.5  # Max decrease in mean risk level
vmax = 0.5   # Max increase in mean risk level

for i, (scenario, title) in enumerate(zip(scenarios_to_plot, titles)):
    ax = axes[i]
    
    col = f"{scenario}_Change"
    
    gdf.plot(
        column=col,
        ax=ax,
        legend=False,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolor='black',
        linewidth=0.5
    )
    
    # Add district labels
    for idx, row in gdf.iterrows():
        centroid = row.geometry.centroid
        name = row[name_col]
        change = row[col]
        
        # Label: Name and change value
        label = f"{name}\n({change:+.2f})"
        ax.annotate(
            label,
            xy=(centroid.x, centroid.y),
            ha='center',
            va='center',
            fontsize=7,
            fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
        )
    
    ax.set_title(f"Thay đổi Mức Nguy cơ Trung bình: {title}\n(so với Baseline 2020)", fontsize=12, fontweight='bold')
    ax.axis('off')

# Add colorbar
cbar_ax = fig.add_axes([0.92, 0.25, 0.02, 0.5])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A = []
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('Thay đổi Mức Nguy cơ TB (Risk Level Change)', fontsize=11)
cbar.ax.tick_params(labelsize=10)

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#2166ac', label='Giảm (Safer)'),
    mpatches.Patch(facecolor='#f7f7f7', edgecolor='black', label='Không đổi'),
    mpatches.Patch(facecolor='#b2182b', label='Tăng (Riskier)')
]

fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=11, frameon=True)

plt.suptitle('PHÂN TÍCH THAY ĐỔI NGUY CƠ NGẬP THEO HUYỆN\n(Kịch bản Biến đổi Khí hậu so với Hiện trạng 2020)', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.05, 0.9, 0.95])

map_path = f"{OUTPUT_DIR}/District_Risk_Change_Map.png"
plt.savefig(map_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {map_path}")

# 6. PRINT SUMMARY TABLE
print("\n" + "=" * 80)
print("📋 BẢNG TÓM TẮT THAY ĐỔI NGUY CƠ THEO HUYỆN")
print("=" * 80)
print(f"\n{'Huyện':<20} {'Baseline':<10} {'RCP85_2100':<12} {'Thay đổi':<10} {'Đánh giá':<15}")
print("-" * 70)

for idx, row in gdf.iterrows():
    name = row[name_col][:18]
    baseline = row['Baseline_Mean']
    future = row['RCP85_2100_Mean']
    change = row['RCP85_2100_Change']
    
    if change > 0.1:
        status = "⚠️ TĂNG MẠNH"
    elif change > 0:
        status = "↗️ Tăng nhẹ"
    elif change < -0.1:
        status = "✅ GIẢM MẠNH"
    elif change < 0:
        status = "↘️ Giảm nhẹ"
    else:
        status = "➡️ Không đổi"
    
    print(f"{name:<20} {baseline:.2f}       {future:.2f}        {change:+.3f}      {status}")

print(f"\n🎉 Analysis complete! Check {OUTPUT_DIR}/ for results.")
