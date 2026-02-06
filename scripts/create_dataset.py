"""
📊 FIXED create_dataset.py
Sửa các lỗi:
1. Thêm random seed cho reproducibility
2. Handle edge cases khi n_hard = 0 hoặc n_rand = 0
3. Thêm validation và logging chi tiết
4. Fix valid_mask logic
"""
import rasterio
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# =========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# =========================================================
path_flood  = '/mnt/DA0054DE0054C365/linh_tinh/nghien_cuu_ve_xam_nhap_man/flood risk/flood_baseline/HaTinh_Flood_Stack_19Events_FullLogic.tif'
path_static = '/mnt/DA0054DE0054C365/linh_tinh/nghien_cuu_ve_xam_nhap_man/flood risk/input/HaTinh_Static_Full_Features_11Bands.tif'
path_rain   = '/mnt/DA0054DE0054C365/linh_tinh/nghien_cuu_ve_xam_nhap_man/flood risk/input/HaTinh_Rain_Stack_CHIRPS_19Events_4Vars.tif'
output_csv  = 'HaTinh_EventBased_Training_Data_Final.csv'

SAMPLES_PER_CLASS = 5000 
HARD_NEG_RATIO = 0.5 

# ⭐ FIX 1: Thêm random seed
np.random.seed(42)

# =========================================================
# 2. HÀM TRÍCH XUẤT DỮ LIỆU (ĐÃ SỬA)
# =========================================================
def extract_samples_v4():
    """
    Version 4: Sửa các lỗi từ v3:
    - Fix valid_mask logic (dùng band data thay vì read_masks)
    - Handle edge cases khi không đủ samples
    - Thêm logging chi tiết
    """
    src_flood = rasterio.open(path_flood)
    src_static = rasterio.open(path_static)
    src_rain = rasterio.open(path_rain)

    rain_inv_transform = ~src_rain.transform

    static_cols = ['elev', 'slope', 'aspect', 'curv', 'relief', 'twi', 'flow_acc', 'dist_water', 'water_mask', 'lulc', 'precip_clim']
    rain_cols = ['Rain_3D', 'Rain_7D', 'Rain_Max', 'Rain_AM14']
    
    event_names = src_flood.descriptions 
    if not any(event_names): 
        event_names = [f"Event_{i+1}" for i in range(src_flood.count)]

    print("⏳ Đang tải dữ liệu tĩnh...")
    
    # ⭐ FIX 2: Tạo valid_mask từ DỮ LIỆU THỰC thay vì read_masks
    # read_masks() trả về mask của file (có thể không đúng với NoData thực tế)
    elev_data = src_static.read(1)  # Band elevation
    
    # Kiểm tra NoData value
    if src_static.nodata is not None:
        valid_mask = elev_data != src_static.nodata
    else:
        # Nếu không có NoData defined, dùng finite check
        valid_mask = np.isfinite(elev_data)
    
    # Thêm điều kiện: loại bỏ elevation < -100 hoặc > 3000 (outliers)
    valid_mask = valid_mask & (elev_data >= -100) & (elev_data <= 3000)
    
    print(f"   Valid pixels: {np.sum(valid_mask):,} / {valid_mask.size:,} ({np.sum(valid_mask)/valid_mask.size*100:.1f}%)")
    
    slope_layer = src_static.read(2)
    dist_layer = src_static.read(8)
    
    # Vùng Hard: Dốc < 5 VÀ Cách sông < 1000m
    hard_zone_mask = (slope_layer < 5) & (dist_layer < 1000) & valid_mask

    all_data = [] 
    stats_log = []
    
    print(f"\n🔄 Đang xử lý {src_flood.count} sự kiện (V4 - Fixed)...")
    
    for idx, event_name in enumerate(tqdm(event_names)):
        flood_layer = src_flood.read(idx + 1)
        
        pos_mask = (flood_layer == 1) & valid_mask
        neg_mask = (flood_layer == 0) & valid_mask
        
        hard_neg_mask = neg_mask & hard_zone_mask
        rand_neg_mask = neg_mask & (~hard_zone_mask)
        
        pos_idx = np.where(pos_mask)
        hard_neg_idx = np.where(hard_neg_mask)
        rand_neg_idx = np.where(rand_neg_mask)
        
        n_pos = len(pos_idx[0])
        n_hard = len(hard_neg_idx[0])
        n_rand = len(rand_neg_idx[0])
        
        # ⭐ FIX 3: Skip event nếu không có positive
        if n_pos == 0: 
            print(f"   ⚠️ {event_name}: Không có flood pixels, bỏ qua!")
            continue 
        
        take_pos = min(n_pos, SAMPLES_PER_CLASS)
        
        target_neg = take_pos 
        target_hard = int(target_neg * HARD_NEG_RATIO)
        target_rand = target_neg - target_hard
        
        # ⭐ FIX 4: Handle edge case khi không đủ hard/rand samples
        take_hard = min(n_hard, target_hard)
        take_rand = min(n_rand, target_neg - take_hard)  # Bù vào rand nếu hard không đủ
        
        total_neg = take_hard + take_rand
        
        # ⭐ FIX 5: Safe random choice - handle empty arrays
        if take_pos > 0:
            p_choices = np.random.choice(n_pos, take_pos, replace=False)
        else:
            p_choices = np.array([], dtype=int)
            
        if take_hard > 0:
            h_choices = np.random.choice(n_hard, take_hard, replace=False)
        else:
            h_choices = np.array([], dtype=int)
            
        if take_rand > 0:
            r_choices = np.random.choice(n_rand, take_rand, replace=False)
        else:
            r_choices = np.array([], dtype=int)
        
        # Ghép indices
        rows = np.concatenate([
            pos_idx[0][p_choices], 
            hard_neg_idx[0][h_choices] if len(h_choices) > 0 else np.array([], dtype=int),
            rand_neg_idx[0][r_choices] if len(r_choices) > 0 else np.array([], dtype=int)
        ])
        cols = np.concatenate([
            pos_idx[1][p_choices], 
            hard_neg_idx[1][h_choices] if len(h_choices) > 0 else np.array([], dtype=int),
            rand_neg_idx[1][r_choices] if len(r_choices) > 0 else np.array([], dtype=int)
        ])
        
        labels = np.concatenate([
            np.ones(take_pos), 
            np.zeros(take_hard + take_rand)
        ])
        
        is_hard = np.concatenate([
            np.zeros(take_pos), 
            np.ones(take_hard), 
            np.zeros(take_rand)
        ])
        
        # Lấy tọa độ không gian
        xs, ys = src_flood.xy(rows, cols)
        
        # Static Features
        static_vals = []
        for b_idx in range(1, 12): 
            band_data = src_static.read(b_idx)
            static_vals.append(band_data[rows, cols])
            
        # Rain Features - Transform coordinates
        r_cols_raw, r_rows_raw = rain_inv_transform * (np.array(xs), np.array(ys))
        r_rows_rain = np.floor(r_rows_raw).astype(int)
        r_cols_rain = np.floor(r_cols_raw).astype(int)
        r_rows_rain = np.clip(r_rows_rain, 0, src_rain.height - 1)
        r_cols_rain = np.clip(r_cols_rain, 0, src_rain.width - 1)
        
        start_band_rain = idx * 4 + 1
        rain_vals = []
        for r_idx in range(4):
            rain_data = src_rain.read(start_band_rain + r_idx)
            rain_vals.append(rain_data[r_rows_rain, r_cols_rain])
            
        df_dict = {
            'Event_ID': idx,
            'Event_Name': event_name,
            'X': xs,
            'Y': ys,
            'Label': labels,
            'Is_Hard_Neg': is_hard
        }
        
        for i, col_name in enumerate(static_cols):
            df_dict[col_name] = static_vals[i]
            
        for i, col_name in enumerate(rain_cols):
            df_dict[col_name] = rain_vals[i]
            
        all_data.append(pd.DataFrame(df_dict))
        
        # Log thống kê
        stats_log.append({
            'Event': event_name,
            'Pos': take_pos,
            'Hard': take_hard,
            'Rand': take_rand,
            'Total': take_pos + total_neg
        })

    # In thống kê
    print("\n📊 THỐNG KÊ SAMPLING:")
    stats_df = pd.DataFrame(stats_log)
    print(stats_df.to_string(index=False))
    
    final_df = pd.concat(all_data, ignore_index=True)
    
    # ⭐ FIX 6: Validation cuối cùng
    print(f"\n🔍 VALIDATION:")
    print(f"   Total rows: {len(final_df):,}")
    print(f"   NaN count per column:")
    nan_counts = final_df.isnull().sum()
    for col, cnt in nan_counts.items():
        if cnt > 0:
            print(f"      {col}: {cnt} ({cnt/len(final_df)*100:.2f}%)")
    
    return final_df

if __name__ == "__main__":
    print("🚀 Bắt đầu tạo Dataset V4 (Fixed)...")
    df = extract_samples_v4()
    
    print(f"\n✅ Hoàn thành! Tổng số dòng: {len(df)}")
    
    # Lưu file
    df.to_csv(output_csv, index=False)
    print(f"💾 Đã lưu file: {output_csv}")
    
    # Thống kê balance
    print(f"\n📊 Class balance:")
    print(df['Label'].value_counts())