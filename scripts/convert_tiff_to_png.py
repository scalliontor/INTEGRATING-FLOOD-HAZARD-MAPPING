"""
📊 TIFF TO PNG CONVERTER + VISUALIZATION ORGANIZER
Chuyển đổi tất cả TIFF thành PNG với scale cố định theo chuẩn khoa học.
Tập hợp tất cả visualizations vào một thư mục duy nhất.

Output: scripts/figures/
"""
import os
import shutil
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

print("=" * 80)
print("📊 TIFF → PNG CONVERTER (Scientific Standard)")
print("=" * 80)

# CONFIG
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Source directories
SCENARIO_DIR = 'output_scenarios'
FINAL_DIR = 'output_final'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_tiff(path):
    """Load TIFF and return data + profile"""
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile
        bounds = src.bounds
        # Mask nodata
        if src.nodata is not None:
            data = np.ma.masked_equal(data, src.nodata)
        else:
            data = np.ma.masked_equal(data, 0)
    return data, bounds

def save_probability_map(data, bounds, title, output_path, vmin=0, vmax=1, cmap='RdYlGn_r'):
    """Save probability map with fixed scale 0-1"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    im = ax.imshow(data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Xác suất Ngập (Flood Probability)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    # Add scale note
    ax.text(0.02, 0.02, f'Scale: {vmin} - {vmax} (Fixed)', transform=ax.transAxes, 
            fontsize=8, verticalalignment='bottom', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: {output_path}")

def save_difference_map(data, bounds, title, output_path, vmin=-0.05, vmax=0.05):
    """Save difference map with symmetric scale"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    im = ax.imshow(data, extent=extent, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Thay đổi Xác suất (ΔP = Climate - Baseline)', fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    # Add interpretation note
    ax.text(0.02, 0.02, 'Đỏ = Tăng rủi ro | Xanh = Giảm rủi ro', transform=ax.transAxes, 
            fontsize=9, verticalalignment='bottom', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: {output_path}")

def save_classified_map(data, bounds, title, output_path):
    """Save classified risk map with categorical colormap"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define categorical colormap for 5 risk levels
    colors = ['#1a9850', '#91cf60', '#fee08b', '#fc8d59', '#d73027']  # Green to Red
    cmap = ListedColormap(colors)
    bounds_levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds_levels, cmap.N)
    
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    im = ax.imshow(data, extent=extent, cmap=cmap, norm=norm, aspect='equal')
    
    # Custom colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, ticks=[1, 2, 3, 4, 5])
    cbar.ax.set_yticklabels(['1-Rất Thấp\n(P<0.2)', '2-Thấp\n(0.2-0.4)', 
                              '3-Trung bình\n(0.4-0.6)', '4-Cao\n(0.6-0.8)', 
                              '5-Rất Cao\n(P≥0.8)'])
    cbar.set_label('Mức Nguy cơ Ngập', fontsize=11)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: {output_path}")

# ============================================================================
# NHÓM A: PROBABILITY MAPS (0-1 scale)
# ============================================================================
print("\n[1/5] Converting Probability Maps (Scale: 0-1)...")

probability_maps = [
    (f'{SCENARIO_DIR}/00_Baseline.tif', 'A1_Baseline_Probability.png', 
     'Bản đồ Xác suất Ngập - HIỆN TRẠNG (Baseline 2020)'),
    (f'{SCENARIO_DIR}/01_RCP45_2050.tif', 'A2_RCP45_2050_Probability.png',
     'Bản đồ Xác suất Ngập - RCP 4.5 (Năm 2050)'),
    (f'{SCENARIO_DIR}/02_RCP85_2050.tif', 'A3_RCP85_2050_Probability.png',
     'Bản đồ Xác suất Ngập - RCP 8.5 (Năm 2050)'),
    (f'{SCENARIO_DIR}/03_RCP45_2100.tif', 'A4_RCP45_2100_Probability.png',
     'Bản đồ Xác suất Ngập - RCP 4.5 (Năm 2100)'),
    (f'{SCENARIO_DIR}/04_RCP85_2100.tif', 'A5_RCP85_2100_Probability.png',
     'Bản đồ Xác suất Ngập - RCP 8.5 (Năm 2100, Khắc nghiệt nhất)'),
]

for tiff_path, png_name, title in probability_maps:
    if os.path.exists(tiff_path):
        data, bounds = load_tiff(tiff_path)
        # Normalize if classified (values 1-5) to probability (0-1)
        if data.max() <= 5:
            # This is classified, convert level to midpoint probability
            # Level 1 -> 0.1, Level 2 -> 0.3, Level 3 -> 0.5, Level 4 -> 0.7, Level 5 -> 0.9
            prob_map = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.7, 5: 0.9}
            data_prob = np.zeros_like(data, dtype=float)
            for level, prob in prob_map.items():
                data_prob[data == level] = prob
            data_prob = np.ma.masked_equal(data_prob, 0)
            save_probability_map(data_prob, bounds, title, f'{OUTPUT_DIR}/{png_name}')
        else:
            save_probability_map(data, bounds, title, f'{OUTPUT_DIR}/{png_name}')
    else:
        print(f"   ⚠️ Not found: {tiff_path}")

# ============================================================================
# NHÓM B: DIFFERENCE MAPS (ΔP, symmetric scale)
# ============================================================================
print("\n[2/5] Creating & Converting Difference Maps (Scale: -0.05 to +0.05)...")

# Load baseline for comparison
baseline_path = f'{SCENARIO_DIR}/00_Baseline.tif'
if os.path.exists(baseline_path):
    baseline_data, bounds = load_tiff(baseline_path)
    
    # Create difference maps for extreme scenarios
    diff_scenarios = [
        (f'{SCENARIO_DIR}/02_RCP85_2050.tif', 'B1_Risk_Difference_RCP85_2050.png',
         'Thay đổi Nguy cơ: RCP 8.5 (2050) so với Hiện trạng'),
        (f'{SCENARIO_DIR}/04_RCP85_2100.tif', 'B2_Risk_Difference_RCP85_2100.png',
         'Thay đổi Nguy cơ: RCP 8.5 (2100) so với Hiện trạng'),
    ]
    
    for tiff_path, png_name, title in diff_scenarios:
        if os.path.exists(tiff_path):
            scenario_data, _ = load_tiff(tiff_path)
            # Calculate difference
            diff_data = scenario_data.astype(float) - baseline_data.astype(float)
            # For classified data (1-5), scale difference appropriately
            if baseline_data.max() <= 5:
                diff_data = diff_data / 5.0  # Normalize to ~0-1 range
            save_difference_map(diff_data, bounds, title, f'{OUTPUT_DIR}/{png_name}',
                               vmin=-0.1, vmax=0.1)
        else:
            print(f"   ⚠️ Not found: {tiff_path}")
else:
    print(f"   ⚠️ Baseline not found: {baseline_path}")

# ============================================================================
# NHÓM C: CLASSIFIED RISK MAP (Categorical)
# ============================================================================
print("\n[3/5] Converting Classified Risk Maps (Categorical)...")

classified_path = f'{FINAL_DIR}/Classified_Risk_Levels.tif'
if os.path.exists(classified_path):
    data, bounds = load_tiff(classified_path)
    save_classified_map(data, bounds, 
                       'Bản đồ Phân cấp Nguy cơ Ngập - 5 Mức\n(Dựa trên Xác suất Ngập)',
                       f'{OUTPUT_DIR}/C1_Classified_Risk_Levels.png')
else:
    print(f"   ⚠️ Not found: {classified_path}")

# ============================================================================
# NHÓM D: COPY EXISTING PNG VISUALIZATIONS
# ============================================================================
print("\n[4/5] Copying existing PNG visualizations...")

png_files_to_copy = [
    # From output_final
    (f'{FINAL_DIR}/SHAP_Summary_Bar.png', 'D1_SHAP_Summary_Bar.png'),
    (f'{FINAL_DIR}/SHAP_Summary_Dot.png', 'D2_SHAP_Summary_Dot.png'),
    (f'{FINAL_DIR}/SHAP_Dependence_Rain7D.png', 'D3_SHAP_Dependence_Rain7D.png'),
    (f'{FINAL_DIR}/SHAP_Dependence_Relief.png', 'D4_SHAP_Dependence_Relief.png'),
    (f'{FINAL_DIR}/Feature_Importance.png', 'D5_Feature_Importance.png'),
    (f'{FINAL_DIR}/Validation_Classification_Report.png', 'D6_Validation_Report.png'),
    (f'{FINAL_DIR}/Risk_Distribution_Shift.png', 'D7_Risk_Distribution_Shift.png'),
    (f'{FINAL_DIR}/Risk_Difference_Hist.png', 'D8_Risk_Difference_Histogram.png'),
    # From output_scenarios
    (f'{SCENARIO_DIR}/Climate_Scenario_Comparison.png', 'E1_Climate_Scenario_Comparison.png'),
    (f'{SCENARIO_DIR}/District_Risk_Change_Map.png', 'E2_District_Risk_Change_Map.png'),
]

for src, dst in png_files_to_copy:
    if os.path.exists(src):
        shutil.copy2(src, f'{OUTPUT_DIR}/{dst}')
        print(f"   ✅ Copied: {dst}")
    else:
        print(f"   ⚠️ Not found: {src}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n[5/5] Generating Summary...")

# Count files
png_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])

print(f"\n{'=' * 80}")
print(f"🎉 HOÀN THÀNH! Tất cả visualizations đã được tập hợp vào: {OUTPUT_DIR}/")
print(f"{'=' * 80}")
print(f"\n📊 TỔNG CỘNG: {png_count} files PNG")

# List all files with categories
print("\n📁 DANH SÁCH FILE:")
print("-" * 60)

categories = {
    'A': 'Probability Maps (Bản đồ Xác suất)',
    'B': 'Difference Maps (Bản đồ Thay đổi)',
    'C': 'Classified Maps (Bản đồ Phân cấp)',
    'D': 'SHAP & Validation (Giải thích Mô hình)',
    'E': 'Climate & District (Kịch bản & Huyện)'
}

for prefix, category in categories.items():
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.startswith(prefix)])
    if files:
        print(f"\n[{prefix}] {category}:")
        for f in files:
            size = os.path.getsize(f'{OUTPUT_DIR}/{f}') / 1024
            print(f"   • {f} ({size:.0f} KB)")

print(f"\n✅ Sẵn sàng để đưa vào báo cáo/paper!")
