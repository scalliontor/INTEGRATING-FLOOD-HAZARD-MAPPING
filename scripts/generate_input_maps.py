"""
📊 INPUT FEATURES VISUALIZATION
Tạo bản đồ cho TẤT CẢ các biến đầu vào (11 Static + 4 Rain).

Output: figures/G_Inputs/
"""
import os
import numpy as np
import rasterio
import rasterio.warp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

print("=" * 80)
print("📊 INPUT FEATURES VISUALIZATION")
print("=" * 80)

# CONFIG
OUTPUT_DIR = 'figures/G_Inputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATIC_TIF = '../input/HaTinh_Static_Full_Features_11Bands.tif'
RAIN_TIF = '../input/HaTinh_Rain_Stack_CHIRPS_19Events_4Vars.tif'
BASELINE_EVENT_IDX = 17  # Event 17 = 2025 May flood

# Feature definitions
STATIC_FEATURES = [
    ('elev', 'Độ cao (Elevation)', 'terrain', 0, 1500, 'm'),
    ('slope', 'Độ dốc (Slope)', 'YlOrRd', 0, 45, '°'),
    ('aspect', 'Hướng dốc (Aspect)', 'hsv', 0, 360, '°'),
    ('curv', 'Độ cong (Curvature)', 'RdBu_r', -0.5, 0.5, '1/m'),
    ('relief', 'Độ chênh cao cục bộ (Relief 500m)', 'viridis', 0, 500, 'm'),
    ('twi', 'Chỉ số Ẩm Địa hình (TWI)', 'Blues', 0, 20, ''),
    ('flow_acc', 'Tích lũy Dòng chảy (Flow Accumulation)', 'hot_r', 0, 10000, 'pixels'),
    ('dist_water', 'Khoảng cách tới Sông (Dist to Water)', 'YlGnBu_r', 0, 10000, 'm'),
    ('water_mask', 'Mặt nạ Thủy phần (Water Mask)', 'Blues', 0, 1, ''),
    ('lulc', 'Sử dụng Đất (Land Use)', 'tab10', 0, 10, 'class'),
    ('precip_clim', 'Lượng mưa TB năm (Climatology)', 'YlGnBu', 1500, 3000, 'mm/year'),
]

RAIN_FEATURES = [
    ('Rain_3D', 'Mưa 3 ngày (Rain 3-Day)', 'Blues', 0, 300, 'mm'),
    ('Rain_7D', 'Mưa 7 ngày (Rain 7-Day)', 'Blues', 0, 500, 'mm'),
    ('Rain_Max', 'Mưa Cực đại 1 ngày (Rain Max)', 'Purples', 0, 200, 'mm'),
    ('Rain_AM14', 'Độ ẩm Tiền cảnh (Antecedent Moisture)', 'Greens', 0, 200, 'mm'),
]

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/3] Loading data...")

with rasterio.open(STATIC_TIF) as src:
    static_data = src.read()
    bounds = src.bounds
    H, W = src.shape
    meta = src.meta.copy()

print(f"   Static: {static_data.shape} (11 bands, {H}x{W})")

with rasterio.open(RAIN_TIF) as src_rain:
    rain_data = []
    start_band = BASELINE_EVENT_IDX * 4 + 1
    for i in range(4):
        rain_buffer = np.zeros((H, W), dtype=np.float32)
        rasterio.warp.reproject(
            source=rasterio.band(src_rain, int(start_band + i)),
            destination=rain_buffer,
            src_transform=src_rain.transform,
            src_crs=src_rain.crs,
            dst_transform=meta['transform'],
            dst_crs=meta['crs'],
            resampling=rasterio.enums.Resampling.bilinear
        )
        rain_data.append(rain_buffer)
    rain_data = np.array(rain_data)

print(f"   Rain: {rain_data.shape} (4 vars for Event {BASELINE_EVENT_IDX})")

# =============================================================================
# GENERATE STATIC FEATURE MAPS
# =============================================================================
print("\n[2/3] Generating Static Feature Maps...")

extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

for idx, (name, title, cmap, vmin, vmax, unit) in enumerate(STATIC_FEATURES):
    print(f"   Processing: {name}...")
    
    data = static_data[idx]
    data_display = np.ma.masked_less_equal(data, -9999)
    data_display = np.ma.masked_equal(data_display, 0) if name == 'water_mask' else data_display
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if name == 'lulc':
        # Categorical colormap for land use
        im = ax.imshow(data_display, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(f'Loại Sử dụng Đất', fontsize=11)
    elif name == 'water_mask':
        colors = ['#f7fbff', '#0868ac']
        cmap_custom = ListedColormap(colors)
        im = ax.imshow(data_display, extent=extent, cmap=cmap_custom, vmin=0, vmax=1)
        legend_elements = [
            Patch(facecolor='#f7fbff', label='Đất liền'),
            Patch(facecolor='#0868ac', label='Sông/Hồ')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    else:
        im = ax.imshow(data_display, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(f'{title} ({unit})' if unit else title, fontsize=11)
    
    ax.set_title(f'Biến Đầu vào: {title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/G{idx+1:02d}_{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: G{idx+1:02d}_{name}.png")

# =============================================================================
# GENERATE RAIN FEATURE MAPS
# =============================================================================
print("\n[3/3] Generating Rain Feature Maps...")

for idx, (name, title, cmap, vmin, vmax, unit) in enumerate(RAIN_FEATURES):
    print(f"   Processing: {name}...")
    
    data = rain_data[idx]
    data_display = np.ma.masked_less_equal(data, 0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(data_display, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(f'{title} ({unit})', fontsize=11)
    
    ax.set_title(f'Biến Mưa Động: {title}\n(Event 17 - Lũ 5/2025)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/G{idx+12:02d}_{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: G{idx+12:02d}_{name}.png")

# =============================================================================
# CREATE PANEL VIEWS
# =============================================================================
print("\n[BONUS] Creating Panel Views...")

# Static Panel (4x3 = 12, but we have 11)
fig, axes = plt.subplots(3, 4, figsize=(24, 18))
axes = axes.flatten()

for idx, (name, title, cmap, vmin, vmax, unit) in enumerate(STATIC_FEATURES):
    data = static_data[idx]
    data_display = np.ma.masked_less_equal(data, -9999)
    
    ax = axes[idx]
    im = ax.imshow(data_display, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'({chr(97+idx)}) {name}', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

# Hide last empty subplot
axes[11].axis('off')

plt.suptitle('Tất cả Biến Địa hình Tĩnh (11 Static Features)\nNguồn: SRTM DEM 30m, HydroSHEDS, ESA WorldCover', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/G00_All_Static_Panel.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Saved: G00_All_Static_Panel.png")

# Rain Panel (2x2)
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, (name, title, cmap, vmin, vmax, unit) in enumerate(RAIN_FEATURES):
    data = rain_data[idx]
    data_display = np.ma.masked_less_equal(data, 0)
    
    ax = axes[idx]
    im = ax.imshow(data_display, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'({chr(97+idx)}) {name}', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(unit, fontsize=10)

plt.suptitle('Tất cả Biến Mưa Động (4 Rain Variables)\nEvent 17 - Lũ tháng 5/2025 | Nguồn: CHIRPS Daily', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/G00_All_Rain_Panel.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Saved: G00_All_Rain_Panel.png")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'=' * 80}")
print("🎉 HOÀN THÀNH!")
print(f"{'=' * 80}")

files = os.listdir(OUTPUT_DIR)
print(f"\n📊 TỔNG CỘNG: {len(files)} files")
print(f"📁 Đường dẫn: {OUTPUT_DIR}/")

print("\n📋 DANH SÁCH:")
print("-" * 50)
for f in sorted(files):
    size = os.path.getsize(f'{OUTPUT_DIR}/{f}') / 1024
    print(f"   • {f} ({size:.0f} KB)")

print("\n✅ Sẵn sàng để đưa vào báo cáo!")
