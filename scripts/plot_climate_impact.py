"""
📊 CLIMATE SCENARIO IMPACT PLOT
Generate a comparison graph for High Risk Areas across different climate scenarios.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

print("=" * 80)
print("📊 GENERATING CLIMATE IMPACT GRAPH")
print("=" * 80)

OUTPUT_DIR = 'output_scenarios'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Data from previous analysis (Step 940)
data = {
    "Scenario": ["Baseline (2020)", "RCP 4.5 (2050)", "RCP 8.5 (2050)", "RCP 4.5 (2100)", "RCP 8.5 (2100)"],
    "Rain_Increase": ["+0%", "+12%", "+15%", "+18%", "+30%"],
    "High_Risk_Area_Ha": [105804, 106102, 106183, 106241, 106528]
}

df = pd.DataFrame(data)

# Calculate Increase
base_area = df.loc[0, "High_Risk_Area_Ha"]
df["Increase_Ha"] = df["High_Risk_Area_Ha"] - base_area
df["Increase_Pct"] = (df["Increase_Ha"] / base_area) * 100

# Plotting
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")
sns.set_palette("Reds_d")

# Create Bar Plot
bars = plt.bar(df["Scenario"], df["High_Risk_Area_Ha"], color=['#95a5a6', '#f1c40f', '#e67e22', '#e74c3c', '#c0392b'])

# Add Labels
plt.ylim(105000, 107000) # Zoom in to see the difference clearly (since difference is small)
plt.ylabel("High Risk Area (ha) - P > 0.6", fontsize=12)
plt.title("Impact of Climate Change on Flood Risk Area (Ha Tinh)", fontsize=16, fontweight='bold')

# Annotate bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    pct = df.loc[i, "Increase_Pct"]
    rain = df.loc[i, "Rain_Increase"]
    
    # Label: Area (ha)
    plt.text(bar.get_x() + bar.get_width()/2., height + 50,
             f'{int(height):,}\n({pct:+.2f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')
    
    # Label: Rain info inside bar
    plt.text(bar.get_x() + bar.get_width()/2., 105100,
             f'Rain\n{rain}',
             ha='center', va='bottom', fontsize=10, color='white', fontweight='bold')

plt.tight_layout()
out_path = f"{OUTPUT_DIR}/Climate_Scenario_Comparison.png"
plt.savefig(out_path, dpi=300)
print(f"✅ Saved Comparison Graph to: {out_path}")
