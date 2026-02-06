"""
✅ MAP VALIDATION & CLASSIFICATION (5 LEVELS)
1. Compare Predicted Risk Map vs Actual Flood Extent (Ground Truth).
2. Classify Risk Probability into 5 Levels (Very Low -> Very High).
3. Generate Validation Metrics & Maps.
"""
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, jaccard_score

print("=" * 80)
print("🗺️ VALIDATION & CLASSIFICATION (5 LEVELS)")
print("=" * 80)

# CONFIG
RISK_MAP_PATH = 'output_final/Baseline_Risk.tif'
ACTUAL_FLOOD_STACK = '../flood_baseline/HaTinh_Flood_Stack_19Events_FullLogic.tif'
OUTPUT_DIR = 'output_final'
EVENT_IDX = 9  # Update to 9 (DAI HONG THUY 2020)

# 1. LOAD DATA
print("\n[1/3] Loading Maps...")
with rasterio.open(RISK_MAP_PATH) as src_risk:
    risk_prob = src_risk.read(1)
    meta = src_risk.meta.copy()
    valid_mask = risk_prob != -1 # -1 was nodata
    
with rasterio.open(ACTUAL_FLOOD_STACK) as src_flood:
    # Event ID 11 -> Band 12 (1-based indexing)
    print(f"   Loading Actual Flood for Event {EVENT_IDX} (Band {EVENT_IDX+1})...")
    actual_flood = src_flood.read(EVENT_IDX + 1)
    
    # Ensure shapes match (Resize actual if needed, but they should match if using same reference)
    if actual_flood.shape != risk_prob.shape:
        print("   ⚠️ Shape mismatch! Resizing actual flood to match risk map...")
        # Simple crop or resize implementation if needed, but let's assume match for now
        # based on previous script logic.
        pass

# Mask valid area for comparison
mask = valid_mask & (actual_flood != 255) # Assuming 255 is nodata for byte
y_true = actual_flood[mask]
y_prob = risk_prob[mask]

# Threshold prediction for binary comparison (Prob > 0.5)
y_pred = (y_prob > 0.5).astype(int)

# 2. VALIDATION METRICS
print("\n[2/3] Validation Metrics (Binary Comparison @ 0.5):")
acc = accuracy_score(y_true, y_pred)
iou = jaccard_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print(f"   ✅ Accuracy: {acc:.4f}")
print(f"   ✅ IoU (Intersection over Union): {iou:.4f}")
print(f"   ✅ AUC Score: {auc:.4f}")
print(f"   Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"   Precision: {tp / (tp+fp):.4f}")
# ... (Existing metric calculation above)

# 2.5 THRESHOLD OPTIMIZATION (NEW)
print("\n[2.5/3] Optimization: Finding Best Thresholds...")
thresholds = np.linspace(0, 1, 101)
f1_scores = []
j_scores = [] # Youden's J = Sensitivity + Specificity - 1

for t in thresholds:
    y_p = (y_prob > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_p).ravel()
    
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0
    j = sens + spec - 1
    
    f1_scores.append(f1)
    j_scores.append(j)

best_f1_idx = np.argmax(f1_scores)
best_j_idx = np.argmax(j_scores)

opt_thresh_f1 = thresholds[best_f1_idx]
opt_thresh_j = thresholds[best_j_idx]

print(f"   🚀 Optimal Threshold (Maximize F1-Score): {opt_thresh_f1:.2f} (F1={f1_scores[best_f1_idx]:.4f})")
print(f"   🚀 Optimal Threshold (Youden's J):        {opt_thresh_j:.2f} (J={j_scores[best_j_idx]:.4f})")

# Determine Jenks Natural Breaks (Simplified using Quantiles of Flood Pixels)
# Lets see distribution of Probabilities for ACTUAL FLOOD pixels
prob_flood = y_prob[y_true == 1]
p20 = np.percentile(prob_flood, 20)
p40 = np.percentile(prob_flood, 40)
p60 = np.percentile(prob_flood, 60)
p80 = np.percentile(prob_flood, 80)

print(f"\n   Stats of Flood Pixels (Probability Distribution):")
print(f"   - 20th Percentile (Easy flood): {p20:.2f}")
print(f"   - Median (Typical flood):       {np.median(prob_flood):.2f}")
print(f"   - 80th Percentile (Deep flood): {p80:.2f}")

# Recommendation
print(f"   💡 SUGGESTION: Instead of 0.2/0.4/0.6/0.8, maybe use: {opt_thresh_j:.2f} as the main cut-off?")

# 3. CLASSIFICATION (5 LEVELS)
print("\n[3/3] Classifying Risk into 5 Levels...")
# 1: Very Low (<0.2)
# 2: Low (0.2-0.4)
# 3: Moderate (0.4-0.6)
# 4: High (0.6-0.8)
# 5: Very High (>0.8)

classified_map = np.zeros_like(risk_prob, dtype=np.uint8)
classified_map[valid_mask & (risk_prob < 0.2)] = 1
classified_map[valid_mask & (risk_prob >= 0.2) & (risk_prob < 0.4)] = 2
classified_map[valid_mask & (risk_prob >= 0.4) & (risk_prob < 0.6)] = 3
classified_map[valid_mask & (risk_prob >= 0.6) & (risk_prob < 0.8)] = 4
classified_map[valid_mask & (risk_prob >= 0.8)] = 5
classified_map[~valid_mask] = 0 # NoData

# Save Classified Map
meta.update(dtype=rasterio.uint8, nodata=0)
out_cls_path = f"{OUTPUT_DIR}/Classified_Risk_Levels.tif"
with rasterio.open(out_cls_path, 'w', **meta) as dst:
    dst.write(classified_map, 1)
print(f"💾 Saved Classified Map to: {out_cls_path}")

# 4. VISUALIZATION
print("   Generating Comparison Plots...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# A. Risk Probability
im1 = axes[0].imshow(risk_prob, cmap='jet', vmin=0, vmax=1)
axes[0].set_title(f'Predicted Probability (AUC={auc:.3f})')
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# B. Actual Flood
im2 = axes[1].imshow(actual_flood, cmap='Blues', vmin=0, vmax=1)
axes[1].set_title(f'Actual Flood Event {EVENT_IDX}')
# plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# C. 5-Level Classification
levels_cmap = colors.ListedColormap(['white', '#00ff00', '#ffff00', '#ff9900', '#ff0000', '#990000'])
bounds = [0, 1, 2, 3, 4, 5, 6]
norm = colors.BoundaryNorm(bounds, levels_cmap.N)

im3 = axes[2].imshow(classified_map, cmap=levels_cmap, norm=norm)
axes[2].set_title('Risk Levels (Very Low -> Very High)')
cbar = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04, ticks=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
cbar.set_ticklabels(['NoData', 'Very Low', 'Low', 'Moderate', 'High', 'Very High'])

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/Validation_Classification_Report.png", dpi=300)
print(f"💾 Saved Plot to: {OUTPUT_DIR}/Validation_Classification_Report.png")
