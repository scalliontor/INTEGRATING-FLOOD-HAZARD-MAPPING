"""
📊 FINAL SHAP ANALYSIS
Load the trained model and generate SHAP values to explain model behavior.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import os

print("=" * 80)
print("📊 GENERATING SHAP EXPLANATIONS")
print("=" * 80)

# Configuration
CSV_PATH = 'HaTinh_Training_Ready_Clean.csv'
MODEL_PATH = 'output_final/Flood_Model.json'
OUTPUT_DIR = 'output_final'

FEATURES = [
    'slope', 'aspect', 'curv', 'relief', 'twi', 'flow_acc', 'dist_water', 
    'lulc', 'precip_clim', 
    'Rain_3D', 'Rain_7D', 'Rain_Max', 'Rain_AM14'
]

# 1. Load Data
print("🔄 Loading Data...")
df = pd.read_csv(CSV_PATH)
X = df[FEATURES]

# 2. Load Model
print("🔄 Loading Model...")
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# SHAP expects the model object or Booster. 
# XGBClassifier wrap requires typical sklearn handling, but TreeExplainer works fine with the booster.
# But for consistency with features, we pass the dataframe.

# 3. Compute SHAP
print("🧮 Computing SHAP values (this may take a minute)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 4. Plots

# A. Summary Plot (Bar)
print("   Generating Summary Bar Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.savefig(f"{OUTPUT_DIR}/SHAP_Summary_Bar.png", bbox_inches='tight', dpi=300)
plt.close()

# B. Summary Plot (Dot/Beeswarm)
print("   Generating Summary Dot Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, show=False)
plt.title("SHAP Summary (Impact on Flood Risk)")
plt.savefig(f"{OUTPUT_DIR}/SHAP_Summary_Dot.png", bbox_inches='tight', dpi=300)
plt.close()

# C. Dependence Plot for Rain_7D
print("   Generating Dependence Plot for Rain_7D...")
plt.figure(figsize=(8, 6))
shap.dependence_plot("Rain_7D", shap_values, X, show=False, interaction_index="relief")
plt.title("SHAP Dependence: Rain_7D vs Relief")
plt.savefig(f"{OUTPUT_DIR}/SHAP_Dependence_Rain7D.png", bbox_inches='tight', dpi=300)
plt.close()

# D. Dependence Plot for Relief
print("   Generating Dependence Plot for Relief...")
plt.figure(figsize=(8, 6))
shap.dependence_plot("relief", shap_values, X, show=False, interaction_index="Rain_7D")
plt.title("SHAP Dependence: Relief vs Rain_7D")
plt.savefig(f"{OUTPUT_DIR}/SHAP_Dependence_Relief.png", bbox_inches='tight', dpi=300)
plt.close()

print(f"✅ SHAP plots saved to {OUTPUT_DIR}/")
