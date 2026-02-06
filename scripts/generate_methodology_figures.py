"""
📊 METHODOLOGY INTERMEDIATE FIGURES GENERATOR
Tạo các hình minh họa quy trình xây dựng nhãn và dữ liệu huấn luyện.

Output:
- F1_Flood_Event_Representative.png  (Flood mask Event 9 - Đại Hồng Thủy 2020)
- F2_Flood_Frequency_Map.png         (Tần suất ngập từ 19 events)
- F3_Hard_Negative_Overlay.png       (So sánh Flood vs Hard Negative)
- F4_Terrain_Panel.png               (DEM, Slope, TWI panel)
"""
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os

print("=" * 80)
print("📊 METHODOLOGY INTERMEDIATE FIGURES GENERATOR")
print("=" * 80)

# CONFIG
OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

FLOOD_STACK_PATH = '../flood_baseline/HaTinh_Flood_Stack_19Events_FullLogic.tif'
STATIC_PATH = '../input/HaTinh_Static_Full_Features_11Bands.tif'
EVENT_ID = 9  # Đại Hồng Thủy 2020 (0-indexed, so band 10)

# ============================================================================
# FIGURE 1: FLOOD EVENT REPRESENTATIVE (Event 9 - 2020)
# ============================================================================
print("\n[1/4] Generating Flood Event Representative...")

with rasterio.open(FLOOD_STACK_PATH) as src:
    # Read Event 9 (band 10, 1-indexed)
    flood_event = src.read(EVENT_ID + 1)
    bounds = src.bounds
    nodata = src.nodata

# Mask nodata
flood_event = np.ma.masked_equal(flood_event, nodata if nodata else 255)
flood_event = np.ma.masked_equal(flood_event, 0)  # Also mask non-flood for clean display

fig, ax = plt.subplots(figsize=(12, 10))
extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

# Binary flood map: show flood as red
flood_display = np.where(flood_event == 1, 1, np.nan)
ax.imshow(np.ones_like(flood_event) * 0.9, extent=extent, cmap='gray', vmin=0, vmax=1)  # Background
ax.imshow(flood_display, extent=extent, cmap='Reds', vmin=0, vmax=1, alpha=0.8)

ax.set_title('Vùng Ngập từ Sentinel-1 SAR\nSự kiện: ĐẠI HỒNG THỦY 10/2020 (Event 9)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Easting (m)', fontsize=10)
ax.set_ylabel('Northing (m)', fontsize=10)
ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))

# Legend
legend_elements = [
    Patch(facecolor='darkred', label='Vùng ngập (Flood Extent)'),
    Patch(facecolor='lightgray', label='Không ngập / NoData')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Caption note
ax.text(0.02, 0.02, 
        'Nguồn: Sentinel-1 GRD VH\nPhương pháp: Threshold + Slope Mask + Object Filter',
        transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F1_Flood_Event_Representative.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   ✅ Saved: {OUTPUT_DIR}/F1_Flood_Event_Representative.png")

# ============================================================================
# FIGURE 2: FLOOD FREQUENCY MAP (Sum of 19 events)
# ============================================================================
print("\n[2/4] Generating Flood Frequency Map...")

with rasterio.open(FLOOD_STACK_PATH) as src:
    # Read all 19 bands and sum
    all_events = src.read()  # Shape: (19, H, W)
    bounds = src.bounds

# Replace nodata with 0 for summing
all_events = np.where((all_events == 255) | (all_events == src.nodata), 0, all_events)
flood_frequency = np.sum(all_events, axis=0)

# Mask zeros for display
flood_frequency_display = np.ma.masked_equal(flood_frequency, 0)

fig, ax = plt.subplots(figsize=(12, 10))
extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

im = ax.imshow(flood_frequency_display, extent=extent, cmap='hot_r', vmin=0, vmax=19)

cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('Số lần ngập quan sát được (Flood Frequency)', fontsize=11)
cbar.ax.tick_params(labelsize=10)

ax.set_title('Bản đồ Tần suất Ngập Lịch sử\n(Tổng hợp từ 19 sự kiện lũ 2016-2025)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Easting (m)', fontsize=10)
ax.set_ylabel('Northing (m)', fontsize=10)
ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))

# Caption note
ax.text(0.02, 0.02, 
        'F(x) = Σ flood_e(x) với e = 1..19\nGiá trị cao = Vùng ngập thường xuyên',
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F2_Flood_Frequency_Map.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   ✅ Saved: {OUTPUT_DIR}/F2_Flood_Frequency_Map.png")

# ============================================================================
# FIGURE 3: HARD NEGATIVE OVERLAY
# ============================================================================
print("\n[3/4] Generating Hard Negative Overlay...")

# Load static features for hard negative criteria
with rasterio.open(STATIC_PATH) as src:
    slope = src.read(2)      # Band 2: slope
    dist_water = src.read(8) # Band 8: dist_water
    bounds_static = src.bounds

# Load flood event 9
with rasterio.open(FLOOD_STACK_PATH) as src:
    flood_event = src.read(EVENT_ID + 1)

# Define hard negative zone: slope < 5° AND dist_water < 1000m AND NOT flooded
hard_neg_zone = (slope < 5) & (dist_water < 1000) & (flood_event == 0) & (flood_event != 255)
flood_zone = (flood_event == 1)

# Create overlay image
overlay = np.zeros((*flood_event.shape, 3), dtype=np.uint8)
overlay[:, :] = [220, 220, 220]  # Light gray background

# Flood = Red
overlay[flood_zone] = [255, 50, 50]

# Hard Negative = Blue
overlay[hard_neg_zone] = [50, 100, 255]

fig, ax = plt.subplots(figsize=(12, 10))
extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

ax.imshow(overlay, extent=extent)

ax.set_title('So sánh Vùng Ngập và Hard Negative Samples\n(Event 9 - Đại Hồng Thủy 2020)', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Easting (m)', fontsize=10)
ax.set_ylabel('Northing (m)', fontsize=10)
ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))

# Legend
legend_elements = [
    Patch(facecolor='#FF3232', label='Vùng NGẬP (Positive Samples)'),
    Patch(facecolor='#3264FF', label='Hard Negative (Trũng nhưng KHÔNG ngập)'),
    Patch(facecolor='#DCDCDC', label='Không đủ điều kiện / NoData')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Caption note
ax.text(0.02, 0.02, 
        'Hard Negative Zone: Slope < 5° VÀ Dist_Water < 1km VÀ Không ngập\n→ Buộc model học sự tương tác Mưa × Địa hình',
        transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F3_Hard_Negative_Overlay.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   ✅ Saved: {OUTPUT_DIR}/F3_Hard_Negative_Overlay.png")

# ============================================================================
# FIGURE 4: TERRAIN PANEL (DEM, Slope, TWI)
# ============================================================================
print("\n[4/4] Generating Terrain Panel...")

with rasterio.open(STATIC_PATH) as src:
    dem = src.read(1)        # Band 1: elevation
    slope = src.read(2)      # Band 2: slope
    twi = src.read(6)        # Band 6: TWI
    bounds = src.bounds

# Mask invalid values
dem = np.ma.masked_less_equal(dem, 0)
slope = np.ma.masked_less(slope, 0)
twi = np.ma.masked_less(twi, 0)

fig, axes = plt.subplots(1, 3, figsize=(18, 7))
extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

# DEM
im1 = axes[0].imshow(dem, extent=extent, cmap='terrain', vmin=0, vmax=1500)
axes[0].set_title('(a) Độ cao (DEM)', fontsize=12, fontweight='bold')
cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.6, pad=0.02)
cbar1.set_label('Độ cao (m)', fontsize=10)

# Slope
im2 = axes[1].imshow(slope, extent=extent, cmap='YlOrRd', vmin=0, vmax=45)
axes[1].set_title('(b) Độ dốc (Slope)', fontsize=12, fontweight='bold')
cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.6, pad=0.02)
cbar2.set_label('Độ dốc (°)', fontsize=10)

# TWI
im3 = axes[2].imshow(twi, extent=extent, cmap='Blues', vmin=0, vmax=20)
axes[2].set_title('(c) Chỉ số Ẩm Địa hình (TWI)', fontsize=12, fontweight='bold')
cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.6, pad=0.02)
cbar3.set_label('TWI = ln(A/tanβ)', fontsize=10)

for ax in axes:
    ax.set_xlabel('Easting (m)', fontsize=9)
    ax.set_ylabel('Northing (m)', fontsize=9)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    ax.tick_params(labelsize=8)

plt.suptitle('Các Biến Địa hình Tĩnh (Static Terrain Predictors)\nNguồn: SRTM DEM 30m + HydroSHEDS', 
             fontsize=14, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F4_Terrain_Panel.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"   ✅ Saved: {OUTPUT_DIR}/F4_Terrain_Panel.png")

# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'=' * 80}")
print(f"🎉 HOÀN THÀNH! Các hình minh họa Methodology đã được tạo.")
print(f"{'=' * 80}")

print(f"""
📁 OUTPUT FILES (dùng cho Section "Construction of Flood Labels"):

| File | Mô tả | Dùng cho |
|------|-------|----------|
| F1_Flood_Event_Representative.png | Flood mask Event 9 | Chứng minh label từ Sentinel-1 |
| F2_Flood_Frequency_Map.png | Tần suất ngập 19 events | Spatial persistence evidence |
| F3_Hard_Negative_Overlay.png | Flood vs Hard Negative | Hard Negative Mining proof |
| F4_Terrain_Panel.png | DEM, Slope, TWI | Static predictor visualization |

✅ Tất cả đã lưu vào: {OUTPUT_DIR}/
""")
