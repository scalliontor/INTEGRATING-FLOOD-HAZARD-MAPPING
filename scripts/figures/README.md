# 📁 FIGURES - Thư mục Hình ảnh Báo cáo

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
