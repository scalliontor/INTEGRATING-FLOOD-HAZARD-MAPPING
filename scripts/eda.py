"""
📊 FIXED eda.py
Sửa các lỗi:
1. Thêm debug NaN để tìm nguồn gốc
2. Kiểm tra balance per event
3. Thêm các visualization quan trọng
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================================
# 1. LOAD DỮ LIỆU
# ==========================================================
csv_path = 'HaTinh_EventBased_Training_Data_Final.csv'
print(f"🔄 Đang đọc file: {csv_path}")
df = pd.read_csv(csv_path)

print(f"Original Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ==========================================================
# 2. DEBUG NaN (TÌM NGUỒN GỐC)
# ==========================================================
print("\n🔍 PHÂN TÍCH NaN CHI TIẾT:")
print("-" * 50)

nan_counts = df.isnull().sum()
print("NaN count per column:")
for col in df.columns:
    cnt = nan_counts[col]
    if cnt > 0:
        pct = cnt / len(df) * 100
        print(f"   {col}: {cnt:,} ({pct:.2f}%)")
        
# Phân tích NaN theo Event
print("\nNaN by Event (first NaN column):")
nan_col = nan_counts[nan_counts > 0].index[0] if any(nan_counts > 0) else None
if nan_col:
    nan_by_event = df[df[nan_col].isnull()].groupby('Event_Name').size()
    print(nan_by_event.sort_values(ascending=False).head(10))

# ==========================================================
# 3. XỬ LÝ LÀM SẠCH (CLEANING)
# ==========================================================
print("\n🧹 CLEANING:")
print("-" * 50)

# Loại bỏ các dòng chứa NaN
df_clean = df.dropna().copy()

print(f"Original: {len(df):,}")
print(f"Clean: {len(df_clean):,}")
print(f"Removed: {len(df) - len(df_clean):,} ({(len(df) - len(df_clean))/len(df)*100:.1f}%)")

# ==========================================================
# 4. KIỂM TRA BALANCE SAU CLEAN
# ==========================================================
print("\n⚖️ CLASS BALANCE:")
print("-" * 50)

# Overall
print("Overall:")
print(df_clean['Label'].value_counts())

# Per Event
print("\nPer Event balance:")
balance_df = df_clean.groupby(['Event_Name', 'Label']).size().unstack(fill_value=0)
balance_df['Ratio'] = balance_df[1.0] / balance_df[0.0]
balance_df = balance_df.round(2)
print(balance_df)

# Cảnh báo nếu imbalanced
imbalanced = balance_df[(balance_df['Ratio'] < 0.5) | (balance_df['Ratio'] > 2.0)]
if len(imbalanced) > 0:
    print(f"\n⚠️ CẢNH BÁO: {len(imbalanced)} events bị imbalanced (ratio < 0.5 or > 2.0)!")
    print(imbalanced)

# ==========================================================
# 5. EDA 1: KIỂM TRA HARD NEGATIVES
# ==========================================================
print("\n📊 Đang vẽ biểu đồ Hard Negatives...")

conditions = [
    (df_clean['Label'] == 1),
    (df_clean['Label'] == 0) & (df_clean['Is_Hard_Neg'] == 1),
    (df_clean['Label'] == 0) & (df_clean['Is_Hard_Neg'] == 0)
]
choices = ['Positive (Flood)', 'Hard Negative (Risk Zone)', 'Random Negative (Safe Zone)']

df_clean['Group'] = np.select(conditions, choices, default='Unknown')

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='Group', y='slope', hue='Group', data=df_clean, showfliers=False, palette="Set2", legend=False)
plt.title('Kỳ vọng: Hard Negatives có độ dốc THẤP (gần giống Flood)')
plt.xticks(rotation=15)

plt.subplot(1, 2, 2)
sns.boxplot(x='Group', y='dist_water', hue='Group', data=df_clean, showfliers=False, palette="Set2", legend=False)
plt.title('Kỳ vọng: Hard Negatives nằm GẦN SÔNG')
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig('EDA_Hard_Negatives_Check.png', dpi=300)
plt.close()
print("✅ Đã lưu ảnh: EDA_Hard_Negatives_Check.png")

# ==========================================================
# 6. EDA 2: KIỂM TRA TÍN HIỆU MƯA
# ==========================================================
print("📊 Đang vẽ biểu đồ Tín hiệu Mưa...")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.boxplot(x='Label', y='Rain_7D', hue='Label', data=df_clean, showfliers=False, palette="coolwarm", legend=False)
plt.title('Mưa 7 ngày: Ngập (1) vs Không Ngập (0)')

plt.subplot(1, 2, 2)
sns.boxplot(x='Label', y='Rain_AM14', hue='Label', data=df_clean, showfliers=False, palette="coolwarm", legend=False)
plt.title('Độ ẩm đất trước lũ (AM14)')

plt.tight_layout()
plt.savefig('EDA_Rain_Signal_Check.png', dpi=300)
plt.close()
print("✅ Đã lưu ảnh: EDA_Rain_Signal_Check.png")

# ==========================================================
# 7. EDA 3: CORRELATION MATRIX
# ==========================================================
print("📊 Đang vẽ Correlation Matrix...")

# Features để phân tích tương quan (đã bỏ elev do r=0.918 với relief)
features = ['slope', 'aspect', 'curv', 'relief', 'twi', 'flow_acc', 'dist_water', 
            'lulc', 'precip_clim', 'Rain_3D', 'Rain_7D', 'Rain_Max', 'Rain_AM14']

corr_matrix = df_clean[features].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('EDA_Correlation_Matrix.png', dpi=300)
plt.close()
print("✅ Đã lưu ảnh: EDA_Correlation_Matrix.png")

# Cảnh báo multicollinearity
high_corr = []
for i in range(len(features)):
    for j in range(i+1, len(features)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr.append((features[i], features[j], corr_matrix.iloc[i, j]))

if high_corr:
    print("\n⚠️ CẢNH BÁO MULTICOLLINEARITY (|r| > 0.8):")
    for f1, f2, r in high_corr:
        print(f"   {f1} <-> {f2}: r = {r:.3f}")

# ==========================================================
# 8. LƯU LẠI FILE SẠCH
# ==========================================================
output_clean_csv = 'HaTinh_Training_Ready_Clean.csv'
df_clean.to_csv(output_clean_csv, index=False)
print(f"\n💾 Đã lưu file sạch: {output_clean_csv}")
print(f"   Rows: {len(df_clean):,}")
print("🚀 SẴN SÀNG TRAIN XGBOOST!")