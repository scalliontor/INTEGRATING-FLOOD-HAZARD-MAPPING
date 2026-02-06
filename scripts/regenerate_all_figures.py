"""
📊 COMPLETE FIGURE GENERATION PIPELINE (2025 BASELINE)
Tạo toàn bộ hình ảnh cho báo cáo từ TIFF với baseline 2025.
Sử dụng "Nguy cơ" thay vì "Xác suất".

Output: scripts/figures/
"""
import os
import shutil
import numpy as np
import rasterio
import rasterio.warp
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import xgboost as xgb
from tqdm import tqdm

print("=" * 80)
print("📊 COMPLETE FIGURE GENERATION (2025 BASELINE)")
print("=" * 80)

# =============================================================================
# CONFIG
# =============================================================================
OUTPUT_DIR = 'figures'
SCENARIO_DIR = 'output_scenarios'
FINAL_DIR = 'output_final'

STATIC_TIF = '../input/HaTinh_Static_Full_Features_11Bands.tif'
RAIN_TIF = '../input/HaTinh_Rain_Stack_CHIRPS_19Events_4Vars.tif'
FLOOD_STACK = '../flood_baseline/HaTinh_Flood_Stack_19Events_FullLogic.tif'
MODEL_PATH = 'output_final/Flood_Model.json'

# EVENT 17 = 18_2025_Lu_Bat_Thuong_T5 (May 2025)
BASELINE_EVENT_IDX = 17
VALIDATION_EVENT_IDX = 9  # Event 9 = 2020 Đại Hồng Thủy (for validation only)

STATIC_NAMES = ['elev', 'slope', 'aspect', 'curv', 'relief', 'twi', 'flow_acc', 'dist_water', 'water_mask', 'lulc', 'precip_clim']
RAIN_NAMES = ['Rain_3D', 'Rain_7D', 'Rain_Max', 'Rain_AM14']
FEATURES = ['slope', 'aspect', 'curv', 'relief', 'twi', 'flow_acc', 'dist_water', 'lulc', 'precip_clim', 'Rain_3D', 'Rain_7D', 'Rain_Max', 'Rain_AM14']

SCENARIOS = {
    "00_Baseline": (1.0, "Hiện trạng 2025"),
    "01_RCP45_2050": (1.12, "RCP 4.5 (2050)"),
    "02_RCP85_2050": (1.15, "RCP 8.5 (2050)"),
    "03_RCP45_2100": (1.18, "RCP 4.5 (2100)"),
    "04_RCP85_2100": (1.30, "RCP 8.5 (2100)")
}

# =============================================================================
# STEP 0: CLEAR OLD FIGURES
# =============================================================================
print("\n[0/6] Clearing old figures...")
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)
print(f"   ✅ Cleared and recreated: {OUTPUT_DIR}/")

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n[1/6] Loading data...")

# Load model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# Load static
with rasterio.open(STATIC_TIF) as src:
    static_data = src.read()
    meta = src.meta.copy()
    H, W = src.shape
    bounds = src.bounds
    nodata = src.nodata if src.nodata else -9999
    valid_mask = (static_data[0] != nodata) & (static_data[0] > -200)

# Load rain (Event 17 = 2025 Baseline)
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

print(f"   ✅ Loaded: Static (11 bands), Rain (4 vars for Event {BASELINE_EVENT_IDX})")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def classify_risk(prob):
    """Classify probability into 5 risk levels"""
    result = np.zeros_like(prob, dtype=np.uint8)
    result[prob < 0.2] = 1
    result[(prob >= 0.2) & (prob < 0.4)] = 2
    result[(prob >= 0.4) & (prob < 0.6)] = 3
    result[(prob >= 0.6) & (prob < 0.8)] = 4
    result[prob >= 0.8] = 5
    return result

def generate_risk_map(rain_factor):
    """Generate risk map with given rain factor"""
    rows, cols = np.where(valid_mask)
    n_pixels = len(rows)
    X_flat = np.zeros((n_pixels, len(FEATURES)), dtype=np.float32)
    
    for i, fname in enumerate(FEATURES):
        if fname in STATIC_NAMES:
            idx = STATIC_NAMES.index(fname)
            X_flat[:, i] = static_data[idx, rows, cols]
        elif fname in RAIN_NAMES:
            idx = RAIN_NAMES.index(fname)
            X_flat[:, i] = rain_data[idx, rows, cols] * rain_factor
    
    prob_preds = model.predict_proba(X_flat)[:, 1]
    class_preds = classify_risk(prob_preds)
    
    prob_map = np.zeros((H, W), dtype=np.float32)
    class_map = np.zeros((H, W), dtype=np.uint8)
    prob_map[rows, cols] = prob_preds
    class_map[rows, cols] = class_preds
    
    return prob_map, class_map

def save_risk_map_png(data, title, output_path, vmin=0, vmax=1):
    """Save risk/probability map as PNG"""
    fig, ax = plt.subplots(figsize=(12, 10))
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    
    data_display = np.ma.masked_equal(data, 0)
    im = ax.imshow(data_display, extent=extent, cmap='RdYlGn_r', vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Mức độ Nguy cơ Ngập (Flood Risk Score)', fontsize=11)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def save_classified_map_png(data, title, output_path):
    """Save classified risk map as PNG"""
    fig, ax = plt.subplots(figsize=(12, 10))
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    
    colors_list = ['#1a9850', '#91cf60', '#fee08b', '#fc8d59', '#d73027']
    cmap = ListedColormap(colors_list)
    bounds_levels = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = BoundaryNorm(bounds_levels, cmap.N)
    
    data_display = np.ma.masked_equal(data, 0)
    im = ax.imshow(data_display, extent=extent, cmap=cmap, norm=norm)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, ticks=[1, 2, 3, 4, 5])
    cbar.ax.set_yticklabels(['1-Rất Thấp', '2-Thấp', '3-Trung bình', '4-Cao', '5-Rất Cao'])
    cbar.set_label('Mức Nguy cơ Ngập', fontsize=11)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def save_difference_map_png(data, title, output_path, vmin=-0.1, vmax=0.1):
    """Save difference map as PNG"""
    fig, ax = plt.subplots(figsize=(12, 10))
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    
    data_display = np.ma.masked_equal(data, 0)
    im = ax.imshow(data_display, extent=extent, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('Thay đổi Nguy cơ (ΔRisk = Kịch bản - Baseline)', fontsize=11)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Easting (m)', fontsize=10)
    ax.set_ylabel('Northing (m)', fontsize=10)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    ax.text(0.02, 0.02, 'Đỏ = Tăng nguy cơ | Xanh = Giảm nguy cơ', 
            transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# =============================================================================
# STEP 2: GENERATE SCENARIO MAPS (A GROUP)
# =============================================================================
print("\n[2/6] Generating Scenario Maps...")

all_prob_maps = {}
all_class_maps = {}

for name, (factor, label) in SCENARIOS.items():
    print(f"   Processing: {name} ({label}, Rain x{factor})...")
    prob_map, class_map = generate_risk_map(factor)
    all_prob_maps[name] = prob_map
    all_class_maps[name] = class_map
    
    # Save PNG
    title = f'Bản đồ Nguy cơ Ngập - {label}'
    if name == "00_Baseline":
        title = 'Bản đồ Nguy cơ Ngập - HIỆN TRẠNG (Baseline 2025)'
    save_risk_map_png(prob_map, title, f'{OUTPUT_DIR}/A{list(SCENARIOS.keys()).index(name)+1}_{name}_Risk.png')
    
    # Save TIF
    meta.update(dtype=rasterio.float32, count=1, nodata=0)
    with rasterio.open(f'{OUTPUT_DIR}/A{list(SCENARIOS.keys()).index(name)+1}_{name}_Risk.tif', 'w', **meta) as dst:
        dst.write(prob_map, 1)

print("   ✅ Saved 5 scenario maps (PNG + TIF)")

# =============================================================================
# STEP 3: GENERATE DIFFERENCE MAPS (B GROUP)
# =============================================================================
print("\n[3/6] Generating Difference Maps...")

baseline_prob = all_prob_maps["00_Baseline"]

diff_scenarios = [
    ("02_RCP85_2050", "RCP 8.5 (2050)"),
    ("04_RCP85_2100", "RCP 8.5 (2100)")
]

for i, (name, label) in enumerate(diff_scenarios, 1):
    diff_map = all_prob_maps[name] - baseline_prob
    title = f'Thay đổi Nguy cơ: {label} so với Hiện trạng 2025'
    save_difference_map_png(diff_map, title, f'{OUTPUT_DIR}/B{i}_Risk_Difference_{name[3:]}.png')
    
    # Save TIF
    with rasterio.open(f'{OUTPUT_DIR}/B{i}_Risk_Difference_{name[3:]}.tif', 'w', **meta) as dst:
        dst.write(diff_map, 1)

print("   ✅ Saved 2 difference maps (PNG + TIF)")

# =============================================================================
# STEP 4: GENERATE CLASSIFIED MAP (C GROUP)
# =============================================================================
print("\n[4/6] Generating Classified Risk Map...")

save_classified_map_png(all_class_maps["00_Baseline"], 
                        'Bản đồ Phân cấp Nguy cơ Ngập - 5 Mức\n(Hiện trạng 2025)',
                        f'{OUTPUT_DIR}/C1_Classified_Risk_2025.png')

# Save TIF
meta.update(dtype=rasterio.uint8, count=1, nodata=0)
with rasterio.open(f'{OUTPUT_DIR}/C1_Classified_Risk_2025.tif', 'w', **meta) as dst:
    dst.write(all_class_maps["00_Baseline"], 1)

print("   ✅ Saved classified map (PNG + TIF)")

# =============================================================================
# STEP 5: COPY EXISTING CHARTS & GENERATE METHODOLOGY FIGURES
# =============================================================================
print("\n[5/6] Copying existing charts and generating methodology figures...")

# Copy existing PNGs
png_files = [
    (f'{FINAL_DIR}/SHAP_Summary_Bar.png', 'D1_SHAP_Summary_Bar.png'),
    (f'{FINAL_DIR}/SHAP_Summary_Dot.png', 'D2_SHAP_Summary_Dot.png'),
    (f'{FINAL_DIR}/SHAP_Dependence_Rain7D.png', 'D3_SHAP_Dependence_Rain7D.png'),
    (f'{FINAL_DIR}/SHAP_Dependence_Relief.png', 'D4_SHAP_Dependence_Relief.png'),
    (f'{FINAL_DIR}/Feature_Importance.png', 'D5_Feature_Importance.png'),
    (f'{FINAL_DIR}/Validation_Classification_Report.png', 'D6_Validation_Report.png'),
    (f'{SCENARIO_DIR}/Climate_Scenario_Comparison.png', 'E1_Climate_Scenario_Comparison.png'),
    (f'{SCENARIO_DIR}/District_Risk_Change_Map.png', 'E2_District_Risk_Change_Map.png'),
]

for src, dst in png_files:
    if os.path.exists(src):
        shutil.copy2(src, f'{OUTPUT_DIR}/{dst}')
        print(f"   ✅ Copied: {dst}")

# Generate Methodology Figures (F Group)
print("\n   Generating methodology figures...")

# F1: Flood Event Representative (Event 9 for validation context)
with rasterio.open(FLOOD_STACK) as src:
    flood_event = src.read(VALIDATION_EVENT_IDX + 1)
    flood_bounds = src.bounds

fig, ax = plt.subplots(figsize=(12, 10))
extent = [flood_bounds.left, flood_bounds.right, flood_bounds.bottom, flood_bounds.top]
flood_display = np.where(flood_event == 1, 1, np.nan)
ax.imshow(np.ones_like(flood_event) * 0.9, extent=extent, cmap='gray', vmin=0, vmax=1)
ax.imshow(flood_display, extent=extent, cmap='Reds', vmin=0, vmax=1, alpha=0.8)
ax.set_title('Vùng Ngập từ Sentinel-1 SAR\nSự kiện: ĐẠI HỒNG THỦY 10/2020 (Dùng cho Kiểm định)', fontsize=14, fontweight='bold')
ax.set_xlabel('Easting (m)', fontsize=10)
ax.set_ylabel('Northing (m)', fontsize=10)
ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
legend_elements = [Patch(facecolor='darkred', label='Vùng ngập (Flood Extent)')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F1_Flood_Event_Validation.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Generated: F1_Flood_Event_Validation.png")

# F2: Flood Frequency Map
with rasterio.open(FLOOD_STACK) as src:
    all_events = src.read()

all_events = np.where((all_events == 255) | (all_events == src.nodata), 0, all_events)
flood_frequency = np.sum(all_events, axis=0)
flood_frequency_display = np.ma.masked_equal(flood_frequency, 0)

fig, ax = plt.subplots(figsize=(12, 10))
extent = [flood_bounds.left, flood_bounds.right, flood_bounds.bottom, flood_bounds.top]
im = ax.imshow(flood_frequency_display, extent=extent, cmap='hot_r', vmin=0, vmax=19)
cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label('Số lần ngập quan sát được (Flood Frequency)', fontsize=11)
ax.set_title('Bản đồ Tần suất Ngập Lịch sử\n(Tổng hợp từ 19 sự kiện lũ 2016-2025)', fontsize=14, fontweight='bold')
ax.set_xlabel('Easting (m)', fontsize=10)
ax.set_ylabel('Northing (m)', fontsize=10)
ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F2_Flood_Frequency_Map.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Generated: F2_Flood_Frequency_Map.png")

# F3: Hard Negative Overlay
slope_data = static_data[1]
dist_water_data = static_data[7]
with rasterio.open(FLOOD_STACK) as src:
    flood_event_validation = src.read(VALIDATION_EVENT_IDX + 1)

hard_neg_zone = (slope_data < 5) & (dist_water_data < 1000) & (flood_event_validation == 0) & (flood_event_validation != 255)
flood_zone = (flood_event_validation == 1)

overlay = np.zeros((*flood_event_validation.shape, 3), dtype=np.uint8)
overlay[:, :] = [220, 220, 220]
overlay[flood_zone] = [255, 50, 50]
overlay[hard_neg_zone] = [50, 100, 255]

fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(overlay, extent=extent)
ax.set_title('So sánh Vùng Ngập và Hard Negative Samples\n(Event 9 - Đại Hồng Thủy 2020)', fontsize=14, fontweight='bold')
ax.set_xlabel('Easting (m)', fontsize=10)
ax.set_ylabel('Northing (m)', fontsize=10)
ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
legend_elements = [
    Patch(facecolor='#FF3232', label='Vùng NGẬP (Positive Samples)'),
    Patch(facecolor='#3264FF', label='Hard Negative (Trũng nhưng KHÔNG ngập)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F3_Hard_Negative_Overlay.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Generated: F3_Hard_Negative_Overlay.png")

# F4: Terrain Panel
dem = static_data[0]
slope = static_data[1]
twi = static_data[5]

dem = np.ma.masked_less_equal(dem, 0)
slope = np.ma.masked_less(slope, 0)
twi = np.ma.masked_less(twi, 0)

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

im1 = axes[0].imshow(dem, extent=extent, cmap='terrain', vmin=0, vmax=1500)
axes[0].set_title('(a) Độ cao (DEM)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=axes[0], shrink=0.6, pad=0.02).set_label('Độ cao (m)', fontsize=10)

im2 = axes[1].imshow(slope, extent=extent, cmap='YlOrRd', vmin=0, vmax=45)
axes[1].set_title('(b) Độ dốc (Slope)', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=axes[1], shrink=0.6, pad=0.02).set_label('Độ dốc (°)', fontsize=10)

im3 = axes[2].imshow(twi, extent=extent, cmap='Blues', vmin=0, vmax=20)
axes[2].set_title('(c) Chỉ số Ẩm Địa hình (TWI)', fontsize=12, fontweight='bold')
plt.colorbar(im3, ax=axes[2], shrink=0.6, pad=0.02).set_label('TWI = ln(A/tanβ)', fontsize=10)

for ax in axes:
    ax.set_xlabel('Easting (m)', fontsize=9)
    ax.set_ylabel('Northing (m)', fontsize=9)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0,0))

plt.suptitle('Các Biến Địa hình Tĩnh (Static Terrain Predictors)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/F4_Terrain_Panel.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("   ✅ Generated: F4_Terrain_Panel.png")

# =============================================================================
# STEP 6: CREATE README
# =============================================================================
print("\n[6/6] Creating README...")

readme_content = """# 📁 FIGURES - Thư mục Hình ảnh Báo cáo

**Ngày tạo:** 06/02/2026  
**Baseline:** Event 17 (18_2025_Lu_Bat_Thuong_T5 - Lũ tháng 5/2025)  
**Dữ liệu đầu vào:**
- Static Features: `input/HaTinh_Static_Full_Features_11Bands.tif` (SRTM DEM 30m, HydroSHEDS, ESA WorldCover)
- Dynamic Rain: `input/HaTinh_Rain_Stack_CHIRPS_19Events_4Vars.tif` (CHIRPS Daily)
- Flood Labels: `flood_baseline/HaTinh_Flood_Stack_19Events_FullLogic.tif` (Sentinel-1 SAR)

---

## 📊 NHÓM A: BẢN ĐỒ NGUY CƠ NGẬP (Risk Maps)

| File | Mô tả | Dữ liệu nguồn | Scale |
|------|-------|---------------|-------|
| `A1_00_Baseline_Risk.png/.tif` | **Hiện trạng 2025** | Event 17 (Rain x1.0) | 0-1 |
| `A2_01_RCP45_2050_Risk.png/.tif` | Kịch bản RCP 4.5 năm 2050 | Rain x1.12 | 0-1 |
| `A3_02_RCP85_2050_Risk.png/.tif` | Kịch bản RCP 8.5 năm 2050 | Rain x1.15 | 0-1 |
| `A4_03_RCP45_2100_Risk.png/.tif` | Kịch bản RCP 4.5 năm 2100 | Rain x1.18 | 0-1 |
| `A5_04_RCP85_2100_Risk.png/.tif` | Kịch bản RCP 8.5 năm 2100 | Rain x1.30 | 0-1 |

**Cách đọc:** Giá trị 0 = Không có nguy cơ, 1 = Nguy cơ tối đa. Colormap: RdYlGn_r (Xanh = An toàn, Đỏ = Nguy hiểm).

---

## 📊 NHÓM B: BẢN ĐỒ THAY ĐỔI NGUY CƠ (Difference Maps)

| File | Mô tả | Công thức | Scale |
|------|-------|-----------|-------|
| `B1_Risk_Difference_RCP85_2050.png/.tif` | Thay đổi so với Baseline (2050) | ΔRisk = RCP85_2050 - Baseline | ±0.1 |
| `B2_Risk_Difference_RCP85_2100.png/.tif` | Thay đổi so với Baseline (2100) | ΔRisk = RCP85_2100 - Baseline | ±0.1 |

**Cách đọc:** Đỏ = Nguy cơ TĂNG, Xanh = Nguy cơ GIẢM. Colormap: RdBu_r (đối xứng).

---

## 📊 NHÓM C: BẢN ĐỒ PHÂN CẤP (Classified)

| File | Mô tả | Giá trị |
|------|-------|---------|
| `C1_Classified_Risk_2025.png/.tif` | Phân cấp 5 mức nguy cơ | 1=Rất Thấp, 2=Thấp, 3=TB, 4=Cao, 5=Rất Cao |

**Ngưỡng phân cấp:** <0.2 / 0.2-0.4 / 0.4-0.6 / 0.6-0.8 / ≥0.8

---

## 📊 NHÓM D: GIẢI THÍCH MÔ HÌNH (SHAP & Validation)

| File | Mô tả | Nguồn |
|------|-------|-------|
| `D1_SHAP_Summary_Bar.png` | Feature Importance | SHAP |
| `D2_SHAP_Summary_Dot.png` | Beeswarm Plot | SHAP |
| `D3_SHAP_Dependence_Rain7D.png` | Tương tác Rain 7D | SHAP |
| `D4_SHAP_Dependence_Relief.png` | Tương tác Relief | SHAP |
| `D5_Feature_Importance.png` | XGBoost Gain | XGBoost |
| `D6_Validation_Report.png` | Kiểm định với Lũ 2020 | Sentinel-1 |

---

## 📊 NHÓM E: KỊCH BẢN & HUYỆN (Climate & District)

| File | Mô tả |
|------|-------|
| `E1_Climate_Scenario_Comparison.png` | So sánh diện tích nguy cơ cao giữa các kịch bản |
| `E2_District_Risk_Change_Map.png` | Thay đổi nguy cơ theo cấp Huyện |

---

## 📊 NHÓM F: HÌNH MINH HỌA PHƯƠNG PHÁP (Methodology)

| File | Mô tả | Dùng cho Section |
|------|-------|------------------|
| `F1_Flood_Event_Validation.png` | Flood mask Event 9 (2020) | Xây dựng nhãn từ Sentinel-1 |
| `F2_Flood_Frequency_Map.png` | Tần suất ngập 19 events | Observed flood frequency |
| `F3_Hard_Negative_Overlay.png` | Flood vs Hard Negative | Hard Negative Mining |
| `F4_Terrain_Panel.png` | DEM, Slope, TWI | Static terrain predictors |

---

## 🔧 GHI CHÚ KỸ THUẬT

- **Resolution:** 300 DPI (đủ cho in ấn)
- **Coordinate System:** EPSG:32648 (UTM Zone 48N)
- **Pixel Size:** 30m x 30m
- **TIF Files:** GeoTIFF với metadata đầy đủ, có thể mở bằng QGIS/ArcGIS

---

*Tạo tự động bởi script `regenerate_all_figures.py`*
"""

with open(f'{OUTPUT_DIR}/README.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)
print(f"   ✅ Created: {OUTPUT_DIR}/README.md")

# =============================================================================
# SUMMARY
# =============================================================================
print(f"\n{'=' * 80}")
print("🎉 HOÀN THÀNH!")
print(f"{'=' * 80}")

files = os.listdir(OUTPUT_DIR)
png_count = len([f for f in files if f.endswith('.png')])
tif_count = len([f for f in files if f.endswith('.tif')])

print(f"\n📊 TỔNG CỘNG: {png_count} PNG + {tif_count} TIF + 1 README.md")
print(f"📁 Đường dẫn: {OUTPUT_DIR}/")
print("\n✅ Sẵn sàng để đưa vào báo cáo!")
