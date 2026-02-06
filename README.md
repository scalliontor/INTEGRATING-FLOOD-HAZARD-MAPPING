# 🌊 Event-Based Flood Risk Mapping - Ha Tinh Province

> **Data-driven Flood Risk Assessment theo từng sự kiện lũ**  
> Sentinel-1 + Dynamic Rainfall (CHIRPS) + XGBoost + SHAP

📅 **Cập nhật:** 2026-02-05  
📍 **Khu vực:** Hà Tĩnh, Việt Nam  
🛰️ **Dữ liệu:** Sentinel-1 SAR (2016-2025), CHIRPS Daily, SRTM, ESA WorldCover

---

# 📋 Tuyên Bố Dự Án (1 câu)

> Dự án xây dựng hệ thống đánh giá rủi ro ngập **theo sự kiện** cho tỉnh Hà Tĩnh bằng cách kết hợp nhãn ngập Sentinel-1 (conservative, event-based) với địa hình–mặt phủ và **đặc trưng mưa CHIRPS theo từng sự kiện**. Mô hình XGBoost được đánh giá bằng **Leave-One-Event-Out (LOEO)** để kiểm tra khả năng dự báo cho một trận lũ chưa từng thấy. Kết quả được giải thích bằng SHAP và mở rộng để mô phỏng kịch bản **mưa cực đoan tăng 20%**.

---

# Mục Lục

1. [Điểm Khác Biệt So Với Bài Cũ](#1-điểm-khác-biệt-so-với-bài-cũ)
2. [Cấu Trúc Dữ Liệu](#2-cấu-trúc-dữ-liệu)
3. [Cấu Trúc Thư Mục](#3-cấu-trúc-thư-mục)
4. [Chi Tiết Scripts](#4-chi-tiết-scripts)
5. [Pipeline Thực Thi](#5-pipeline-thực-thi)
6. [Feature Set](#6-feature-set)
7. [Danh Sách 19 Sự Kiện](#7-danh-sách-19-sự-kiện)
8. [Kết Quả Đạt Được](#8-kết-quả-đạt-được)
9. [Việc Còn Phải Làm](#9-việc-còn-phải-làm)

---

# 1. Điểm Khác Biệt So Với Bài Cũ

## 1.1. Bài cũ (Static Susceptibility - AHP+SVM)

| Đặc điểm | Bài cũ |
|----------|--------|
| Biến đầu vào | Chỉ **tĩnh** (DEM, slope, dist river...) |
| Mưa | Trung bình nhiều năm / Annual mean |
| Label | Proxy từ AHP/heuristic (không phải lũ thực) |
| Đầu ra | "Điểm nhạy cảm" không gắn với thời điểm |
| Câu hỏi không trả lời được | "Mưa tăng 20% thì rủi ro tăng ở đâu?" |

## 1.2. Bài mới (Dynamic Event-Based Risk)

| Đặc điểm | Bài mới |
|----------|---------|
| Biến đầu vào | **Tĩnh + Động** (Rain_3D, Rain_7D, Rain_Max, Rain_AM14) |
| Mưa | **Tổng mưa từng sự kiện** từ CHIRPS |
| Label | **Ngập quan sát thực tế** từ Sentinel-1 |
| Đầu ra | Xác suất ngập P(x\|event) + Expected frequency |
| Khả năng mới | Mô phỏng kịch bản BĐKH (Rain × 1.2) |

## 1.3. Điểm cốt lõi

Mỗi dòng dữ liệu có dạng:

```
(Pixel_i, Event_e)  →  [X_static(i), X_rain(i,e), y(i,e)]
```

**Cùng một pixel** có thể:
- Không ngập ở event mưa nhỏ
- Ngập ở event mưa lớn

⇒ Model buộc phải học **"mưa × địa hình"**, không thể chỉ học "vùng trũng".

---

# 2. Cấu Trúc Dữ Liệu

## 2.1. Labels từ Sentinel-1 (19 bands = 19 events)

```
flood_baseline/HaTinh_Flood_Stack_19Events_FullLogic.tif
├── Band 01: 01_2016_Lu_Ho_Ho (0/1)
├── Band 02: 02_2016_Lu_T11_Dot2 (0/1)
├── ...
└── Band 19: 19_2025_Lu_T11 (0/1)
```

**Quy trình tạo label (conservative):**
1. Sentinel-1 VH, IW mode, DESCENDING orbit
2. Mask góc nghiêng (31°-45°)
3. Min composite theo event window
4. Focal median 50m (lọc speckle)
5. Threshold VH < -19 dB
6. Slope mask < 10°
7. **Loại permanent water** (ESA class 80)
8. Remove blobs < 20 pixels

## 2.2. Static Features (11 bands)

```
input/HaTinh_Static_Full_Features_11Bands.tif
├── Band 01: elevation     → Độ cao (m)
├── Band 02: slope         → Độ dốc (°)
├── Band 03: aspect        → Hướng sườn (°)
├── Band 04: curv_lap      → Độ cong Laplacian
├── Band 05: relief_2km    → Chênh cao so với đáy thung lũng
├── Band 06: twi           → Topographic Wetness Index
├── Band 07: flow_acc      → Flow Accumulation
├── Band 08: dist_water    → Khoảng cách đến mặt nước (m)
├── Band 09: water_mask    → [ĐÃ BỎ khi train] Mask nước vĩnh cửu
├── Band 10: lulc          → Land Use Land Cover (class)
└── Band 11: precip_clim   → Mưa khí hậu (WorldClim BIO16)
```

## 2.3. Dynamic Rainfall Features (76 bands = 19 events × 4 vars)

```
input/HaTinh_Rain_Stack_CHIRPS_19Events_4Vars.tif
├── 01_2016_Lu_Ho_Ho_Rain_3D    → Tổng mưa 3 ngày (kích hoạt lũ nhanh)
├── 01_2016_Lu_Ho_Ho_Rain_7D    → Tổng mưa 7 ngày (tích nước hồ/sông)
├── 01_2016_Lu_Ho_Ho_Rain_Max   → Ngày mưa lớn nhất trong 7 ngày
├── 01_2016_Lu_Ho_Ho_Rain_AM14  → Độ ẩm đất trước lũ (14 ngày, có trọng số)
├── ...
└── 19_2025_Lu_T11_Rain_AM14
```

**Công thức Rain_AM14 (Antecedent Moisture):**
```
AM14 = Σ (Rain_day_i × 0.9^i)  với i = 1..14
```
Ngày hôm qua w=0.9, 14 ngày trước w≈0.2

---

# 3. Cấu Trúc Thư Mục

```
flood risk/
│
├── 📄 README.md                                    # File này
├── 📄 Event_based_Flood_Susceptibility_*.pdf       # Report PDF
│
├── 📁 scripts/                                     # ⭐ Code Python + Output
│   ├── create_dataset.py                           # Tạo dataset event-based
│   ├── eda.py                                      # Làm sạch + EDA cơ bản
│   ├── eda_adv.py                                  # EDA nâng cao
│   ├── train.py                                    # Train LOEO (GPU)
│   ├── final_analysis.py                           # SHAP + Climate scenario
│   │
│   ├── HaTinh_EventBased_Training_Data_Final.csv   # Dataset gốc (~190k rows)
│   ├── HaTinh_Training_Ready_Clean.csv             # Dataset đã clean (~156k rows)
│   ├── LOEO_Metrics_NoWaterMask.csv                # Kết quả LOEO
│   ├── XGBoost_Flood_Model_Final.json/.pkl         # Model đã train
│   │
│   └── *.png                                       # Các biểu đồ output
│       ├── EDA_Hard_Negatives_Check.png
│       ├── EDA_Rain_Signal_Check.png
│       ├── EDA_Rain_Distribution_Per_Event.png
│       ├── EDA_Terrain_Interaction_Heatmap.png
│       ├── Feature_Importance_NoWaterMask.png
│       ├── Final_SHAP_Summary.png
│       ├── Final_SHAP_Rain7D.png
│       └── Climate_Change_Impact.png
│
├── 📁 input/                                       # Dữ liệu đầu vào (GEE)
│   ├── hatinh_input_feature.js                     # GEE: Tạo Static 11 bands
│   ├── recip.js                                    # GEE: Tạo Rain 76 bands
│   ├── HaTinh_Static_Full_Features_11Bands.tif     # ~207MB
│   └── HaTinh_Rain_Stack_CHIRPS_19Events_4Vars.tif # ~113KB
│
├── 📁 flood_baseline/                              # Labels từ Sentinel-1
│   ├── flood_baseline.js                           # GEE: Tạo label stack
│   └── HaTinh_Flood_Stack_19Events_FullLogic.tif   # ~3MB
│
├── 📁 AOI_level2/                                  # Shapefile ranh giới huyện
│   └── HaTinh_Districts_Level2.*                   # 13 huyện/TX/TP
│
├── 📁 docs/                                        # Tài liệu tham khảo
└── 📁 venv/                                        # Python environment
```

---

# 4. Chi Tiết Scripts

## 4.1. `create_dataset.py` - Tạo Dataset Event-Based

### Mục đích
Trích xuất samples từ rasters theo từng event với **Hard Negative Mining**.

### Input
| File | Mô tả |
|------|-------|
| `flood_baseline/HaTinh_Flood_Stack_19Events_*.tif` | Labels (19 bands) |
| `input/HaTinh_Static_Full_Features_11Bands.tif` | Static features |
| `input/HaTinh_Rain_Stack_CHIRPS_19Events_4Vars.tif` | Rain features |

### Output
| File | Mô tả |
|------|-------|
| `HaTinh_EventBased_Training_Data_Final.csv` | ~190k rows |

### Chiến lược sampling

**Với mỗi event e (19 events):**

| Loại | Số lượng | Điều kiện |
|------|----------|-----------|
| **Positive** | 5000 | y(i,e) = 1 (ngập) |
| **Hard Negative** | 2500 | y=0, slope<5°, dist_water<1000m |
| **Random Negative** | 2500 | y=0, ngoài hard zone |

**Hard Negative** = vùng **thấp trũng, gần sông nhưng KHÔNG ngập** → buộc model học tinh tế hơn.

### Các cột trong dataset
```
Event_ID, Event_Name, X, Y, Label, Is_Hard_Neg,
elev, slope, aspect, curv, relief, twi, flow_acc, dist_water, water_mask, lulc, precip_clim,
Rain_3D, Rain_7D, Rain_Max, Rain_AM14
```

---

## 4.2. `eda.py` - Làm Sạch + EDA Cơ Bản

### Mục đích
Loại bỏ NaN và vẽ EDA kiểm tra chất lượng data.

### Quy trình
1. `dropna()` → giảm từ ~190k xuống ~156k rows
2. Vẽ **Hard Negatives Check**: Boxplot slope/dist_water theo 3 nhóm
3. Vẽ **Rain Signal Check**: Boxplot Rain_7D/AM14 theo Label

### Output
| File | Ý nghĩa |
|------|---------|
| `HaTinh_Training_Ready_Clean.csv` | Dataset sạch |
| `EDA_Hard_Negatives_Check.png` | Kỳ vọng: Hard Neg có slope thấp, gần sông |
| `EDA_Rain_Signal_Check.png` | Kỳ vọng: Ngập có mưa cao hơn |

---

## 4.3. `eda_adv.py` - EDA Nâng Cao

### Plot 1: Rain Distribution Per Event
Median Rain_7D của nhóm Ngập vs Không Ngập theo từng event.

### Plot 2: Terrain Interaction Heatmap
Xác suất ngập thực tế trong lưới (Elevation bins × Slope bins).

→ Giúp hiểu **interaction** giữa địa hình và ngập.

---

## 4.4. `train.py` - Train LOEO (GPU)

### Mục đích
Huấn luyện XGBoost với **Leave-One-Event-Out** validation.

### ⚠️ QUAN TRỌNG: Đã bỏ `water_mask`
```python
# BỎ 'water_mask' khỏi features để tránh Data Leakage
features = [
    'elev', 'slope', 'aspect', 'curv', 'relief', 'twi', 'flow_acc', 'dist_water', 
    'lulc', 'precip_clim',  # ← Đã bỏ 'water_mask'
    'Rain_3D', 'Rain_7D', 'Rain_Max', 'Rain_AM14'
]  # 14 features
```

### LOEO Validation
```
Với 19 events:
  For e in [1..19]:
    Train trên 18 events (tất cả trừ e)
    Test trên event e (chưa từng thấy)
    → Tính AUC, Precision, Recall, F1
```

### Hyperparameters
| Param | Giá trị |
|-------|---------|
| n_estimators | 500 |
| max_depth | 8 |
| learning_rate | 0.05 |
| tree_method | hist |
| device | **cuda** (GPU) |

### Output
| File | Mô tả |
|------|-------|
| `LOEO_Metrics_NoWaterMask.csv` | Metrics của 19 events |
| `Feature_Importance_NoWaterMask.png` | Importance trung bình |

---

## 4.5. `final_analysis.py` - SHAP + Climate Scenario

### Phần 1: Train Final Model
Train trên **toàn bộ data** (không split) để có model mạnh nhất.

### Phần 2: Lưu Model
| Format | File |
|--------|------|
| JSON (nhẹ, tương thích) | `XGBoost_Flood_Model_Final.json` |
| Pickle (tiện Python) | `XGBoost_Flood_Model_Final.pkl` |

### Phần 3: SHAP Analysis
```python
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
```

| Output | Mô tả |
|--------|-------|
| `Final_SHAP_Summary.png` | Beeswarm global importance |
| `Final_SHAP_Rain7D.png` | Dependence plot Rain_7D |

### Phần 4: Climate Scenario (Mưa +20%)
```python
for col in ['Rain_3D', 'Rain_7D', 'Rain_Max', 'Rain_AM14']:
    X_scenario[col] = X_scenario[col] * 1.2

prob_baseline = model.predict_proba(X)[:, 1]
prob_scenario = model.predict_proba(X_scenario)[:, 1]
ΔP = prob_scenario - prob_baseline
```

| Output | Mô tả |
|--------|-------|
| `Climate_Change_Impact.png` | Bar chart ΔP theo event |

---

# 5. Pipeline Thực Thi

```bash
cd scripts/

# Bước 1: Tạo dataset (nếu chưa có)
python create_dataset.py

# Bước 2: Clean + EDA cơ bản
python eda.py

# Bước 3: EDA nâng cao (optional)
python eda_adv.py

# Bước 4: Train LOEO
python train.py

# Bước 5: SHAP + Climate Scenario
python final_analysis.py
```

---

# 6. Feature Set

## 6.1. 14 Features đang dùng (đã bỏ water_mask)

| # | Tên | Loại | Nguồn | Mô tả |
|---|-----|------|-------|-------|
| 1 | elev | Static | SRTM | Độ cao (m) |
| 2 | slope | Static | SRTM | Độ dốc (°) |
| 3 | aspect | Static | SRTM | Hướng sườn (°) |
| 4 | curv | Static | SRTM | Độ cong Laplacian |
| 5 | relief | Static | SRTM | Chênh cao so với thung lũng |
| 6 | twi | Static | HydroSHEDS | Topographic Wetness Index |
| 7 | flow_acc | Static | HydroSHEDS | Flow Accumulation |
| 8 | dist_water | Static | ESA | Khoảng cách đến mặt nước |
| 9 | lulc | Static | ESA | Land Use (categorical) |
| 10 | precip_clim | Static | WorldClim | Mưa khí hậu BIO16 |
| 11 | Rain_3D | **Dynamic** | CHIRPS | Tổng mưa 3 ngày |
| 12 | Rain_7D | **Dynamic** | CHIRPS | Tổng mưa 7 ngày |
| 13 | Rain_Max | **Dynamic** | CHIRPS | Mưa max 1 ngày trong 7D |
| 14 | Rain_AM14 | **Dynamic** | CHIRPS | Độ ẩm đất trước lũ |

## 6.2. Tại sao bỏ water_mask?

**Vấn đề Data Leakage:**
- `water_mask` = permanent water (sông/hồ) từ ESA
- Label đã được tạo bằng cách **trừ đi** permanent water
- ⇒ water_mask = 1 → Label chắc chắn = 0 (shortcut!)

**Sau khi bỏ:**
- Feature importance trở về hợp lý
- `elev`, `dist_water`, `Rain_7D` lên top
- Model học đúng quan hệ thực

---

# 7. Danh Sách 19 Sự Kiện

| ID | Tên | T0 (Start) | Window |
|----|-----|-----------|--------|
| 01 | 2016_Lu_Ho_Ho | 2016-10-10 | 10-25/10 |
| 02 | 2016_Lu_T11_Dot2 | 2016-10-28 | 28/10-15/11 |
| 03 | 2017_Bao_So_2 | 2017-07-14 | 14-30/07 |
| 04 | 2017_ATND_Sau_Bao | 2017-10-05 | 05-25/10 |
| 05 | 2018_Mua_T7 | 2018-07-12 | 12-30/07 |
| 06 | 2019_Lu_Dau_Mua | 2019-08-30 | 30/08-15/09 |
| 07 | 2019_Lu_T10 | 2019-10-10 | 10-25/10 |
| 08 | 2020_Bao_So_5 | 2020-09-15 | 15-30/09 |
| 09 | 2020_Lu_Dau_T10 | 2020-10-02 | 02-14/10 |
| 10 | **2020_DAI_HONG_THUY** | 2020-10-15 | 15/10-05/11 |
| 11 | 2021_Lu_T9 | 2021-09-19 | 19/09-05/10 |
| 12 | 2021_Lu_T10_Dot1 | 2021-10-12 | 12-25/10 |
| 13 | 2021_Lu_T10_Dot2 | 2021-10-24 | 24/10-05/11 |
| 14 | 2022_Bao_Noru | 2022-09-24 | 24/09-15/10 |
| 15 | 2023_Lu_T9 | 2023-09-22 | 22/09-07/10 |
| 16 | 2023_Lu_Vu_Quang | 2023-10-25 | 25/10-15/11 |
| 17 | 2024_Sau_Bao_Soulik | 2024-09-15 | 15/09-05/10 |
| 18 | 2025_Lu_Bat_Thuong_T5 | 2025-05-15 | 15/05-05/06 |
| 19 | 2025_Lu_T11 | 2025-10-25 | 25/10-15/11 |

---

# 8. Kết Quả Đạt Được

## 8.1. LOEO Metrics (sau khi bỏ water_mask)

| Metric | Giá trị trung bình |
|--------|-------------------|
| **AUC** | ~0.90 |
| **Precision** | ~0.86 |
| **Recall** | ~0.84 |
| **F1** | ~0.85 |

## 8.2. Feature Importance (Top 5)

1. **lulc** - Land use (tác động mạnh)
2. **dist_water** - Khoảng cách sông
3. **elev** - Độ cao
4. **Rain_7D** - Mưa 7 ngày
5. **relief** - Chênh cao địa hình

## 8.3. Climate Scenario Observation

⚠️ **Lưu ý quan trọng:**
- Một số events có ΔP âm (giảm rủi ro) → không thể kết luận theo mean
- Cần aggregation đúng: chỉ tính trên **floodplain/lowland** hoặc tính **% diện tích > threshold**

---

# 9. Việc Còn Phải Làm

## ✅ Đã hoàn thành

- [x] Event-based labels sạch (19 events)
- [x] Static + Rainfall features stack
- [x] Dataset + Hard Negatives Mining
- [x] LOEO training + metrics mạnh
- [x] Xử lý Data Leakage (bỏ water_mask)
- [x] SHAP summary + dependence
- [x] EDA terrain interaction
- [x] Climate scenario (Rain +20%)

## 🔲 Còn thiếu (để thành paper)

- [ ] **SHAP interaction plots đúng chuẩn**
  - Rain_7D vs SHAP colored by elev/dist_water
  
- [ ] **Climate aggregation đúng policy**
  - Δ% area high-risk (P > 0.7) theo event
  - Hoặc chỉ tính trên lowland mask
  
- [ ] **Spatial outputs**
  - Map P(x|e) cho vài event tiêu biểu
  - Map Expected Frequency / Exceedance

- [ ] **Zonal statistics**
  - Top 10 huyện/xã rủi ro cao
  - Top 10 tăng mạnh nhất khi Rain +20%

---

# 10. Thư Viện Yêu Cầu

```txt
numpy pandas rasterio tqdm
xgboost scikit-learn
matplotlib seaborn shap
geopandas rasterstats (cho zonal)
```

---

*Cập nhật: 2026-02-05*
