"""
🔧 FINAL PRODUCTION SCRIPT: TRAIN & GENERATE FLOOD MAPS
1. Train XGBoost model (Monotonic Constraints) on CSV data.
2. Load full raster datasets (Static + Rain).
3. Generate Flood Risk Maps (GeoTIFF) for:
   - Baseline (Historical Event)
   - Climate Change Scenario (+20% Rain)
4. Generate Analysis Reports & Graphs.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import rasterio
import rasterio.warp
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
import joblib
import os
import json
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

print("=" * 80)
print("🌊 HÀ TĨNH FLOOD RISK: FINAL EXECUTION PIPELINE")
print("=" * 80)

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
CSV_PATH = 'HaTinh_Training_Ready_Clean.csv'
STATIC_TIF = '../input/HaTinh_Static_Full_Features_11Bands.tif'
RAIN_TIF = '../input/HaTinh_Rain_Stack_CHIRPS_19Events_4Vars.tif'
OUTPUT_DIR = 'output_final'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Feature Names (Order mapping for raster bands)
# Static Bands: 1=elev, 2=slope, ..., 5=relief, ..., 11=precip_clim
STATIC_NAMES = ['elev', 'slope', 'aspect', 'curv', 'relief', 'twi', 'flow_acc', 'dist_water', 'water_mask', 'lulc', 'precip_clim']
# Rain Bands: 4 vars per event (3D, 7D, Max, AM14)
RAIN_NAMES = ['Rain_3D', 'Rain_7D', 'Rain_Max', 'Rain_AM14']

FEATURES = [
    'slope', 'aspect', 'curv', 'relief', 'twi', 'flow_acc', 'dist_water', 
    'lulc', 'precip_clim', 
    'Rain_3D', 'Rain_7D', 'Rain_Max', 'Rain_AM14'
]

# Monotonic Constraints
MONO_CONSTRAINTS = tuple(
    -1 if f == 'relief' else (1 if 'Rain' in f else 0) for f in FEATURES
)

# ==============================================================================
# 2. TRAINING MODEL
# ==============================================================================
print("\n[1/4] Training Model...")
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df):,} samples.")

model = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    tree_method='hist', device='cuda', random_state=42,
    monotone_constraints=MONO_CONSTRAINTS
)
model.fit(df[FEATURES], df['Label'])

# Save
model.save_model(f"{OUTPUT_DIR}/Flood_Model.json")
print(f"✅ Model saved to {OUTPUT_DIR}/Flood_Model.json")

# ==============================================================================
# 3. MAP GENERATION FUNCTION
# ==============================================================================
def generate_risk_map(event_idx, scenario_factor=1.0, out_name="Risk_Map.tif"):
    """
    Generates a full risk map for a specific event.
    event_idx: 0-based index of the event in rain stack
    scenario_factor: Multiplier for rain (1.0 = Baseline, 1.2 = +20%)
    """
    print(f"\n🌍 Generating Map: {out_name} (Rain x {scenario_factor})")
    
    # Open Rasters
    src_static = rasterio.open(STATIC_TIF)
    src_rain = rasterio.open(RAIN_TIF)
    
    # Read Static Data
    print("   Reading Static Data...")
    static_data = src_static.read() # (11, H, W)
    H, W = src_static.shape
    
    # Read Rain Data (Upscale if needed - assumes CHIRPS 5km needs resize likely, but here we read raw)
    # The Rain stack has resolution mismatch with Static usually.
    # We must match Rain to Static geometry.
    print("   Reading & Resampling Rain Data...")
    
    rain_data_resampled = []
    start_band = event_idx * 4 + 1
    
    # Build coordinate grid for the Static raster (Target)
    # Actually, simpler to use rasterio.reproject/read with out_shape
    
    for i in range(4):
        # Read from Rain TIF
        # We need to read into a buffer matching Static TIF dimensions
        rain_band_idx = int(start_band + i)
        
        # Allocate buffer
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
        
        # Apply Scenario Factor
        rain_buffer *= scenario_factor
        rain_data_resampled.append(rain_buffer)
        
    rain_data_resampled = np.array(rain_data_resampled) # (4, H, W)
    
    # Prepare Inference Data
    # Stack: Static features (mapped by name) + Rain features
    # FEATURES list defines order.
    # Static Indices in STATIC_NAMES
    
    print("   Predicting full raster (this may take a moment)...")
    
    # Valid mask (where data exists)
    nodata = src_static.nodata if src_static.nodata else -9999
    valid_mask = (static_data[0] != nodata) & (static_data[0] > -100)
    
    # Initialize output probability map
    prob_map = np.zeros((H, W), dtype=np.float32) - 1 # -1 for nodata
    
    # Flatten valid pixels for batch prediction
    rows, cols = np.where(valid_mask)
    if len(rows) == 0:
        print("❌ Error: No valid pixels found!")
        return
        
    n_pixels = len(rows)
    print(f"   Processing {n_pixels:,} pixels...")
    
    # Construct Feature Matrix X
    # X shape: (n_pixels, n_features)
    X_flat = np.zeros((n_pixels, len(FEATURES)), dtype=np.float32)
    
    # Fill Static Features
    for i, fname in enumerate(FEATURES):
        if fname in STATIC_NAMES:
            idx = STATIC_NAMES.index(fname)
            X_flat[:, i] = static_data[idx, rows, cols]
        elif fname in RAIN_NAMES:
            idx = RAIN_NAMES.index(fname)
            X_flat[:, i] = rain_data_resampled[idx, rows, cols]
            
    # Predict in batches to save RAM
    batch_size = 500000
    preds = []
    
    
    # XGBoost sklearn API expects numpy array for predict_proba
    # Also we want PROBABILITY, not class label
    preds = model.predict_proba(X_flat)[:, 1] 
    
    # Fill map
    prob_map[rows, cols] = preds
    
    # Export GeoTIFF
    meta = src_static.meta.copy()
    meta.update(dtype=rasterio.float32, count=1, nodata=-1)
    
    out_path = f"{OUTPUT_DIR}/{out_name}"
    with rasterio.open(out_path, 'w', **meta) as dst:
        dst.write(prob_map, 1)
        
    print(f"✅ Saved map to {out_path}")
    return prob_map, valid_mask

# ==============================================================================
# 4. EXECUTION
# ==============================================================================

# Select a representative flood event (e.g., Event 5 - a major one)
# Or select Event with max rainfall
# Select "DAI HONG THUY 2020" (Event 10 in list -> Index 9)
target_event_id = 9
print(f"Selected Event {target_event_id} (Historic Flood 2020) for mapping.")

# A. Baseline Map
map_base, mask = generate_risk_map(target_event_id, 1.0, "Baseline_Risk.tif")

# B. Climate Map (+20%)
map_climate, _ = generate_risk_map(target_event_id, 1.2, "Climate_Risk.tif")

# ==============================================================================
# 5. ANALYSIS & PLOTS
# ==============================================================================
print("\n[4/4] Generating Analysis...")

# 1. Pixel-wise Difference
diff_map = map_climate - map_base
# Only look at valid pixels
valid_diff = diff_map[mask]

# 2. Probability Distribution Plot
plt.figure(figsize=(10, 6))
plt.hist(map_base[mask], bins=100, alpha=0.5, label='Baseline', density=True, color='blue')
plt.hist(map_climate[mask], bins=100, alpha=0.5, label='Climate (+20% Rain)', density=True, color='red')
plt.xlabel('Flood Probability')
plt.ylabel('Density')
plt.title(f'Risk Shift Distribution (Event {target_event_id})')
plt.legend()
plt.savefig(f"{OUTPUT_DIR}/Risk_Distribution_Shift.png")

# 3. Difference Histogram
plt.figure(figsize=(10, 6))
plt.hist(valid_diff, bins=100, color='purple', range=(-0.05, 0.05)) # Limit range to see details
plt.axvline(0, color='black', linestyle='--')
plt.xlabel('Change in Probability (Climate - Baseline)')
plt.ylabel('Count')
plt.title('Pixel-wise Risk Change (\u0394P)')
plt.text(0.01, plt.ylim()[1]*0.9, f"Mean \u0394P: {valid_diff.mean():.4f}\n% P\u2191: {(valid_diff > 0).mean()*100:.1f}%")
plt.savefig(f"{OUTPUT_DIR}/Risk_Difference_Hist.png")

# 4. Feature Importance
imp = model.feature_importances_
plt.figure(figsize=(10, 8))
indices = np.argsort(imp)
plt.title('Feature Importances')
plt.barh(range(len(indices)), imp[indices], color='b', align='center')
plt.yticks(range(len(indices)), [FEATURES[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig(f"{OUTPUT_DIR}/Feature_Importance.png")

print(f"\n🎉 ALL TASKS COMPLETED. Outputs in {OUTPUT_DIR}/")
