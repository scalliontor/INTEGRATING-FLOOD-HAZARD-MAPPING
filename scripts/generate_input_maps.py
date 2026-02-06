"""
📊 INPUT FEATURES VISUALIZATION V2 (FIXED)
Tạo bản đồ cho TẤT CẢ các biến đầu vào (11 Static + 4 Rain) + TIF files.

Fixes:
- LULC: Correct ESA WorldCover colormap (Tree cover visible)
- All maps: Proper masking and color scales
- Add TIF files for each feature
"""
import os
import numpy as np
import rasterio
import rasterio.warp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

print("=" * 80)
print("📊 INPUT FEATURES VISUALIZATION V2 (FIXED)")
print("=" * 80)

# CONFIG
OUTPUT_DIR = 'figures/G_Inputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

STATIC_TIF = '../input/HaTinh_Static_Full_Features_11Bands.tif'
RAIN_TIF = '../input/HaTinh_Rain_Stack_CHIRPS_19Events_4Vars.tif'
BASELINE_EVENT_IDX = 17

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/4] Loading data...")

with rasterio.open(STATIC_TIF) as src:
    static_data = src.read()
    bounds = src.bounds
    H, W = src.shape
    meta = src.meta.copy()

print(f"   Static: {static_data.shape}")

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

print(f"   Rain: {rain_data.shape}")

extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

# =============================================================================
# HELPER: Save TIF
# =============================================================================
def save_tif(data, output_path, dtype=rasterio.float32):
    """Save data as GeoTIFF"""
    meta_out = meta.copy()
    meta_out.update(dtype=dtype, count=1, nodata=-9999 if dtype == rasterio.float32 else 255)
    with rasterio.open(output_path, 'w', **meta_out) as dst:
        dst.write(data.astype(dtype if dtype != rasterio.float32 else np.float32), 1)

# =============================================================================
# [2/4] STATIC FEATURES (Except LULC)
# =============================================================================
print("\n[2/4] Generating Static Feature Maps...")

STATIC_FEATURES = [
    (0, 'elev', 'Độ cao (Elevation)', 'terrain', 0, 1500, 'm'),
    (1, 'slope', 'Độ dốc (Slope)', 'YlOrRd', 0, 45, '°'),
    (2, 'aspect', 'Hướng dốc (Aspect)', 'hsv', 0, 360, '°'),
    (3, 'curv', 'Độ cong (Curvature)', 'RdBu_r', -0.5, 0.5, '1/m'),
    (4, 'relief', 'Độ chênh cao cục bộ (Relief 500m)', 'viridis', 0, 500, 'm'),
    (5, 'twi', 'Chỉ số Ẩm Địa hình (TWI)', 'Blues', 0, 20, ''),
    (6, 'flow_acc', 'Tích lũy Dòng chảy (Log10)', 'hot_r', 0, 5, 'log10(pixels)'),
    (7, 'dist_water', 'Khoảng cách tới Sông', 'YlGnBu_r', 0, 10000, 'm'),
    (8, 'water_mask', 'Mặt nạ Thủy phần', 'custom', 0, 1, ''),
    # Skip LULC (index 9) - handle separately
    (10, 'precip_clim', 'Lượng mưa TB năm', 'YlGnBu', 1500, 3000, 'mm/year'),
]

for band_idx, name, title, cmap, vmin, vmax, unit in STATIC_FEATURES:
    print(f"   Processing: {name}...")
    
    data = static_data[band_idx].copy()
    
    # Apply appropriate masking
    if name == 'elev':
        data = np.ma.masked_less_equal(data, -200)
    elif name == 'flow_acc':
        # Log transform for flow accumulation
        data = np.log10(np.maximum(data, 1))
        data = np.ma.masked_less_equal(data, 0)
    elif name == 'water_mask':
        data = np.ma.masked_equal(data, -9999)
    else:
        data = np.ma.masked_less_equal(data, -9999)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if name == 'water_mask':
        colors = ['#e8f4f8', '#0868ac']
        cmap_obj = ListedColormap(colors)
        im = ax.imshow(data, extent=extent, cmap=cmap_obj, vmin=0, vmax=1)
        legend_elements = [
            Patch(facecolor='#e8f4f8', edgecolor='gray', label='Đất liền'),
            Patch(facecolor='#0868ac', label='Sông/Hồ')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    else:
        im = ax.imshow(data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(f'{unit}' if unit else title, fontsize=11)
    
    ax.set_title(f'Biến Đầu vào: {title}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    plt.tight_layout()
    idx_display = band_idx + 1 if band_idx < 9 else band_idx  # Adjust for LULC
    plt.savefig(f'{OUTPUT_DIR}/G{idx_display:02d}_{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save TIF
    save_tif(static_data[band_idx], f'{OUTPUT_DIR}/G{idx_display:02d}_{name}.tif')
    print(f"   ✅ Saved: G{idx_display:02d}_{name}.png + .tif")

# =============================================================================
# [3/4] LULC (Special handling)
# =============================================================================
print("\n[3/4] Generating LULC Map (ESA WorldCover)...")

lulc = static_data[9].copy()

# ESA WorldCover classes
ESA_CLASSES = [
    (10, 'Tree cover', '#006400'),
    (20, 'Shrubland', '#FFBB22'),
    (30, 'Grassland', '#FFFF4C'),
    (40, 'Cropland', '#F096FF'),
    (50, 'Built-up', '#FA0000'),
    (60, 'Bare/sparse', '#B4B4B4'),
    (80, 'Permanent water', '#0064C8'),
    (90, 'Herbaceous wetland', '#0096A0'),
    (95, 'Mangroves', '#00CF75'),
]

fig, ax = plt.subplots(figsize=(12, 10))

# Create discrete colormap
colors = [c[2] for c in ESA_CLASSES]
bounds_vals = [c[0] - 5 for c in ESA_CLASSES] + [100]  # Create boundaries around class values
cmap = ListedColormap(colors)
norm = BoundaryNorm([5, 15, 25, 35, 45, 55, 75, 85, 92.5, 100], cmap.N)

lulc_display = np.ma.masked_less_equal(lulc, 0)
im = ax.imshow(lulc_display, extent=extent, cmap=cmap, norm=norm)

# Legend
legend_elements = [Patch(facecolor=c[2], edgecolor='black', linewidth=0.5, 
                         label=f'{c[0]}: {c[1]}') for c in ESA_CLASSES]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, title='ESA WorldCover 2021')

ax.set_title('Biến Đầu vào: Sử dụng Đất (LULC)\nNguồn: ESA WorldCover 2021', fontsize=14, fontweight='bold')
ax.set_xlabel('Easting (m)', fontsize=10)
ax.set_ylabel('Northing (m)', fontsize=10)
ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/G10_lulc.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# Save TIF
save_tif(lulc, f'{OUTPUT_DIR}/G10_lulc.tif')
print("   ✅ Saved: G10_lulc.png + .tif")

# =============================================================================
# [4/4] RAIN FEATURES
# =============================================================================
print("\n[4/4] Generating Rain Feature Maps...")

RAIN_FEATURES = [
    (0, 'Rain_3D', 'Mưa 3 ngày', 'Blues', 0, 300, 'mm'),
    (1, 'Rain_7D', 'Mưa 7 ngày', 'Blues', 0, 500, 'mm'),
    (2, 'Rain_Max', 'Mưa Cực đại 1 ngày', 'Purples', 0, 150, 'mm'),
    (3, 'Rain_AM14', 'Độ ẩm Tiền cảnh 14 ngày', 'Greens', 0, 200, 'mm'),
]

for idx, name, title, cmap, vmin, vmax, unit in RAIN_FEATURES:
    print(f"   Processing: {name}...")
    
    data = rain_data[idx].copy()
    data = np.ma.masked_less_equal(data, 0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(f'{unit}', fontsize=11)
    
    ax.set_title(f'Biến Mưa Động: {title}\n(Event 17 - Lũ tháng 5/2025)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/G{idx+12:02d}_{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save TIF
    save_tif(rain_data[idx], f'{OUTPUT_DIR}/G{idx+12:02d}_{name}.tif')
    print(f"   ✅ Saved: G{idx+12:02d}_{name}.png + .tif")

# =============================================================================
# PANEL VIEWS
# =============================================================================
print("\n[BONUS] Regenerating Panel Views...")

# Static Panel
fig, axes = plt.subplots(3, 4, figsize=(24, 18))
axes = axes.flatten()

panel_features = [
    (0, 'elev', 'terrain', 0, 1500),
    (1, 'slope', 'YlOrRd', 0, 45),
    (2, 'aspect', 'hsv', 0, 360),
    (3, 'curv', 'RdBu_r', -0.5, 0.5),
    (4, 'relief', 'viridis', 0, 500),
    (5, 'twi', 'Blues', 0, 20),
    (6, 'flow_acc', 'hot_r', 0, 10000),
    (7, 'dist_water', 'YlGnBu_r', 0, 10000),
    (8, 'water_mask', 'Blues', 0, 1),
    (9, 'lulc', 'tab10', 10, 95),
    (10, 'precip_clim', 'YlGnBu', 1500, 3000),
]

for i, (band_idx, name, cmap, vmin, vmax) in enumerate(panel_features):
    data = static_data[band_idx]
    data = np.ma.masked_less_equal(data, -9999)
    
    ax = axes[i]
    im = ax.imshow(data, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'({chr(97+i)}) {name}', fontsize=11, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

axes[11].axis('off')

plt.suptitle('Tất cả Biến Địa hình Tĩnh (11 Static Features)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/G00_All_Static_Panel.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Saved: G00_All_Static_Panel.png")

# Rain Panel
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for i in range(4):
    data = rain_data[i]
    data = np.ma.masked_less_equal(data, 0)
    
    ax = axes[i]
    im = ax.imshow(data, extent=extent, cmap=['Blues', 'Blues', 'Purples', 'Greens'][i])
    ax.set_title(f'({chr(97+i)}) {RAIN_FEATURES[i][1]}', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

plt.suptitle('Tất cả Biến Mưa Động (Event 17 - 2025)', fontsize=14, fontweight='bold', y=1.02)
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

files = sorted(os.listdir(OUTPUT_DIR))
png_count = len([f for f in files if f.endswith('.png')])
tif_count = len([f for f in files if f.endswith('.tif')])

print(f"\n📊 TỔNG CỘNG: {png_count} PNG + {tif_count} TIF")
print(f"📁 Đường dẫn: {OUTPUT_DIR}/")
