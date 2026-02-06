import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle  # <--- THƯ VIỆN ĐỂ LƯU .PKL

# ==========================================================
# 1. LOAD DỮ LIỆU & CẤU HÌNH
# ==========================================================
csv_path = 'HaTinh_Training_Ready_Clean.csv'
print(f"🔄 Đang đọc dữ liệu: {csv_path}")
df = pd.read_csv(csv_path)

# Feature Set CHUẨN (Đã bỏ water_mask và elev do multicollinearity r=0.918 với relief)
features = [
    'slope', 'aspect', 'curv', 'relief', 'twi', 'flow_acc', 'dist_water', 
    'lulc', 'precip_clim', 
    'Rain_3D', 'Rain_7D', 'Rain_Max', 'Rain_AM14'
]
target = 'Label'

X = df[features]
y = df[target]

# ==========================================================
# 2. TRAIN MODEL FINAL
# ==========================================================
print("🚀 Đang Train Model Final (GPU)...")

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    tree_method='hist',
    device='cuda', # GPU
    random_state=42
)

model.fit(X, y)

# ==========================================================
# 3. LƯU MODEL (SAVE) - CẢ 2 ĐỊNH DẠNG
# ==========================================================
print("\n💾 ĐANG LƯU MODEL...")

# Cách 1: Lưu dạng JSON (Chuẩn của XGBoost - Nhẹ, tương thích cao)
model.save_model('XGBoost_Flood_Model_Final.json')
print("   ✅ Đã lưu JSON: XGBoost_Flood_Model_Final.json")

# Cách 2: Lưu dạng PICKLE (.pkl) - Tiện dụng cho Python
pkl_filename = "XGBoost_Flood_Model_Final.pkl"
with open(pkl_filename, "wb") as f:
    pickle.dump(model, f)
print(f"   ✅ Đã lưu PICKLE: {pkl_filename}")

# ==========================================================
# 4. SHAP ANALYSIS
# ==========================================================
print("\n⚡ Đang tính SHAP Values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# A. SHAP SUMMARY
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
plt.title('SHAP Summary (No Water Mask)', fontsize=14)
plt.tight_layout()
plt.savefig('Final_SHAP_Summary.png', dpi=300)
plt.close()
print("   ✅ Đã lưu ảnh SHAP Summary")

# B. SHAP DEPENDENCE (MƯA)
plt.figure(figsize=(10, 6))
shap.dependence_plot("Rain_7D", shap_values, X, interaction_index=None, show=False)
plt.title('Quan hệ: Mưa 7 ngày vs Nguy cơ Ngập', fontsize=12)
plt.tight_layout()
plt.savefig('Final_SHAP_Rain7D.png', dpi=300)
plt.close()
print("   ✅ Đã lưu ảnh SHAP Dependence")

# ==========================================================
# 5. KỊCH BẢN BĐKH (MƯA TĂNG 20%)
# ==========================================================
print("\n🌍 Đang chạy mô phỏng BĐKH (Mưa tăng 20%)...")

X_scenario = X.copy()
rain_cols = ['Rain_3D', 'Rain_7D', 'Rain_Max', 'Rain_AM14']
for col in rain_cols:
    X_scenario[col] = X_scenario[col] * 1.2 

prob_baseline = model.predict_proba(X)[:, 1]
prob_scenario = model.predict_proba(X_scenario)[:, 1]

df['Risk_Increase'] = prob_scenario - prob_baseline
avg_increase = df.groupby('Event_Name')['Risk_Increase'].mean().sort_values(ascending=False)

print("\n--- TOP 5 SỰ KIỆN TĂNG RỦI RO MẠNH NHẤT ---")
print(avg_increase.head(5))

plt.figure(figsize=(14, 7))
sns.barplot(x=avg_increase.values, y=avg_increase.index, palette='Reds_r')
plt.title('Mức tăng Nguy cơ Ngập khi Mưa tăng 20%', fontsize=14)
plt.xlabel('Delta Probability')
plt.tight_layout()
plt.savefig('Climate_Change_Impact.png', dpi=300)
plt.close()
print("   ✅ Đã lưu ảnh BĐKH")

print("\n🎉 HOÀN THÀNH! BẠN ĐÃ CÓ FILE .PKL ĐỂ DÙNG SAU NÀY.")