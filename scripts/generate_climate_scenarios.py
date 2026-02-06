"""
🌍 GENERATE CLIMATE CHANGE SCENARIOS (RCP 4.5 & 8.5)
Generate Flood Risk Maps for 2050 and 2100 under different emission scenarios.

Scenarios (Based on MONRE 2020 for North Central Coast / Ha Tinh - Rainfall Rx5day):
1. Baseline (2020 Historic Flood): 1.0
2. RCP 4.5 (2050): +12% -> 1.12
3. RCP 8.5 (2050): +15% -> 1.15
4. RCP 4.5 (2100): +18% -> 1.18
5. RCP 8.5 (2100): +30% -> 1.30

Outputs:
- Classified GeoTIFFs (5 Risk Levels) for each scenario.
- Comparison plots.
"""
import rasterio
import rasterio.warp
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import glob
from tqdm import tqdm

print("=" * 80)
print("🌍 CLIMATE SCENARIO GENERATION: RCP 4.5 & 8.5")
print("=" * 80)

# CONFIG
MODEL_PATH = 'output_final/Flood_Model.json'
STATIC_TIF = '../input/HaTinh_Static_Full_Features_11Bands.tif'
RAIN_TIF = '../input/HaTinh_Rain_Stack_CHIRPS_19Events_4Vars.tif'
OUTPUT_DIR = 'output_scenarios'
EVENT_IDX = 9  # DAI HONG THUY 2020

# SCENARIO FACTORS (Rain Multipliers)
SCENARIOS = {
    "00_Baseline": 1.0,
    "01_RCP45_2050": 1.12,
    "02_RCP85_2050": 1.15,
    "03_RCP45_2100": 1.18,
    "04_RCP85_2100": 1.30
}

# Features List (Order must match training)
STATIC_NAMES = ['elev', 'slope', 'aspect', 'curv', 'relief', 'twi', 'flow_acc', 'dist_water', 'water_mask', 'lulc', 'precip_clim']
RAIN_NAMES = ['Rain_3D', 'Rain_7D', 'Rain_Max', 'Rain_AM14']
FEATURES = [
    'slope', 'aspect', 'curv', 'relief', 'twi', 'flow_acc', 'dist_water', 
    'lulc', 'precip_clim', 
    'Rain_3D', 'Rain_7D', 'Rain_Max', 'Rain_AM14'
]

# 1. SETUP OUTPUT
if os.path.exists(OUTPUT_DIR):
    print(f"🧹 Clearing old data in {OUTPUT_DIR}...")
    files = glob.glob(f"{OUTPUT_DIR}/*")
    for f in files:
        os.remove(f)
else:
    os.makedirs(OUTPUT_DIR)

# 2. LOAD MODEL
print("🔄 Loading Model...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# load static ONE TIME
print("🔄 Loading Static Raster...")
with rasterio.open(STATIC_TIF) as src_static:
    static_data = src_static.read() # (11, H, W)
    meta = src_static.meta.copy()
    H, W = src_static.shape
    
    # Valid mask (Just ignore nodata, KEEP RIVERS)
    nodata = src_static.nodata if src_static.nodata else -9999
    # We want to keep rivers. Rivers usually have valid elevation (even if negative or 0).
    # Static data usually covers the whole AOI.
    # Just mask absolutely invalid values.
    valid_mask = (static_data[0] != nodata) & (static_data[0] > -200)

# 3. PREPARE RAIN BASELINE
print("🔄 Preparing Baseline Rain Data...")
with rasterio.open(RAIN_TIF) as src_rain:
    rain_data_resampled = []
    start_band = EVENT_IDX * 4 + 1
    
    for i in range(4):
        # Read & Resize
        rain_band_idx = int(start_band + i)
        rain_buffer = np.zeros((H, W), dtype=np.float32)
        rasterio.warp.reproject(
            source=rasterio.band(src_rain, rain_band_idx),
            destination=rain_buffer,
            src_transform=src_rain.transform,
            src_crs=src_rain.crs,
            dst_transform=src_static.transform,
            dst_crs=src_static.crs,
            resampling=rasterio.enums.Resampling.bilinear
        )
        rain_data_resampled.append(rain_buffer)
    rain_data_resampled = np.array(rain_data_resampled)

# 4. GENERATE MAPS LOOP
print("\n🚀 Starting Batch Generation...")

results = {}

for name, factor in SCENARIOS.items():
    print(f"\n🌍 Processing: {name} (Rain x {factor})")
    
    # 4.1 Prepare Features (Apply Factor)
    # We construct X flat
    rows, cols = np.where(valid_mask)
    n_pixels = len(rows)
    X_flat = np.zeros((n_pixels, len(FEATURES)), dtype=np.float32)
    
    for i, fname in enumerate(FEATURES):
        if fname in STATIC_NAMES:
            idx = STATIC_NAMES.index(fname)
            X_flat[:, i] = static_data[idx, rows, cols]
        elif fname in RAIN_NAMES:
            idx = RAIN_NAMES.index(fname)
            # Apply Rain Factor Here
            X_flat[:, i] = rain_data_resampled[idx, rows, cols] * factor
            
    # 4.2 Predict Probability
    prob_preds = model.predict_proba(X_flat)[:, 1]
    
    # 4.3 Classify (5 Levels)
    # 1: <0.2, 2: 0.2-0.4, 3: 0.4-0.6, 4: 0.6-0.8, 5: >0.8
    class_preds = np.zeros_like(prob_preds, dtype=np.uint8)
    class_preds[prob_preds < 0.2] = 1
    class_preds[(prob_preds >= 0.2) & (prob_preds < 0.4)] = 2
    class_preds[(prob_preds >= 0.4) & (prob_preds < 0.6)] = 3
    class_preds[(prob_preds >= 0.6) & (prob_preds < 0.8)] = 4
    class_preds[prob_preds >= 0.8] = 5
    
    # 4.4 Save Classified TIF
    out_map = np.zeros((H, W), dtype=np.uint8)
    out_map[rows, cols] = class_preds
    
    meta.update(dtype=rasterio.uint8, count=1, nodata=0)
    out_name = f"{OUTPUT_DIR}/{name}.tif"
    with rasterio.open(out_name, 'w', **meta) as dst:
        dst.write(out_map, 1)
        
    print(f"   ✅ Saved: {out_name}")
    
    # Store for stats
    results[name] = {
        'HighRisk_Pix': np.sum(class_preds >= 4), # Level 4+5
        'VeryHigh_Pix': np.sum(class_preds == 5),
        'Total_Pix': n_pixels
    }

# 5. SUMMARY REPORT
print("\n📊 SCENARIO STATISTICS")
print(f"{'Scenario':<20} {'High Risk (ha)':<15} {'% Increase':<15}")
base_high = results["00_Baseline"]['HighRisk_Pix']

# Pixel size 30m -> 900m2 = 0.09 ha
PIXEL_HA = 0.09

for name, stats in results.items():
    area_ha = stats['HighRisk_Pix'] * PIXEL_HA
    delta = ((stats['HighRisk_Pix'] - base_high) / base_high) * 100
    print(f"{name:<20} {area_ha:,.0f} ha   {delta:+.2f}%")

print(f"\n🎉 Done! All scenarios in {OUTPUT_DIR}/")
