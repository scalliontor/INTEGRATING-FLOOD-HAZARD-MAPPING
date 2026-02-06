import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================================
# 1. LOAD DỮ LIỆU
# ==========================================================
csv_path = 'HaTinh_Training_Ready_Clean.csv'
print(f"🔄 Đang đọc dữ liệu: {csv_path}")
df = pd.read_csv(csv_path)

# ==========================================================
# 2. VẼ PHÂN BỐ MƯA THEO SỰ KIỆN (Median Rain_7D)
# ==========================================================
print("📊 Đang vẽ biểu đồ Mưa theo Sự kiện...")

# Tính Median Mưa 7 ngày cho nhóm Ngập (Label=1) và Không Ngập (Label=0) theo từng sự kiện
rain_stats = df.groupby(['Event_Name', 'Label'])['Rain_7D'].median().unstack()

# Sắp xếp theo lượng mưa của nhóm Ngập để dễ nhìn
rain_stats = rain_stats.sort_values(by=1, ascending=False)

plt.figure(figsize=(14, 8))
# Vẽ đường biểu diễn
plt.plot(rain_stats.index, rain_stats[1], marker='o', color='red', linewidth=2, label='Median Rain (Flood Locations)')
plt.plot(rain_stats.index, rain_stats[0], marker='x', color='blue', linestyle='--', label='Median Rain (Non-Flood Locations)')

plt.title('Phân bố Mưa 7 ngày theo từng Sự kiện (Sắp xếp theo độ lớn)', fontsize=14)
plt.ylabel('Lượng mưa 7 ngày (mm)', fontsize=12)
plt.xlabel('Tên Sự kiện', fontsize=12)
plt.xticks(rotation=90) # Xoay tên sự kiện cho dễ đọc
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('EDA_Rain_Distribution_Per_Event.png', dpi=300)
plt.close()
print("✅ Đã lưu: EDA_Rain_Distribution_Per_Event.png")

# ==========================================================
# 3. VẼ TƯƠNG TÁC ĐỊA HÌNH (Heatmap: Elev x Slope)
# ==========================================================
print("📊 Đang vẽ Heatmap Tương tác Địa hình...")

# Chia Elevation và Slope thành các khoảng (Bins)
# Elevation: 20 khoảng từ thấp đến cao
e_bins = pd.qcut(df["elev"], q=20, duplicates="drop")
# Slope: 20 khoảng
s_bins = pd.qcut(df["slope"], q=20, duplicates="drop")

# Tính Tỷ lệ Ngập (Probability of Flood) trong từng ô lưới (Elev, Slope)
# Mean của Label (0/1) chính là xác suất ngập thực nghiệm
prob_matrix = df.groupby([e_bins, s_bins])["Label"].mean().unstack()

plt.figure(figsize=(10, 8))
sns.heatmap(prob_matrix, cmap="YlOrRd", annot=False, fmt=".2f", cbar_kws={'label': 'Xác suất Ngập Thực tế'})

# Đảo ngược trục Y để Elevation thấp nằm ở dưới (trực quan hơn)
plt.gca().invert_yaxis()

plt.title('Xác suất Ngập theo Độ cao & Độ dốc', fontsize=14)
plt.xlabel('Khoảng Độ dốc (Slope Bins)', fontsize=12)
plt.ylabel('Khoảng Độ cao (Elevation Bins)', fontsize=12)
plt.tight_layout()
plt.savefig('EDA_Terrain_Interaction_Heatmap.png', dpi=300)
plt.close()
print("✅ Đã lưu: EDA_Terrain_Interaction_Heatmap.png")

print("\n🚀 HOÀN THÀNH VẼ BIỂU ĐỒ!")