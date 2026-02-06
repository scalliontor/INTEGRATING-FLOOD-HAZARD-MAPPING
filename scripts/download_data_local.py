"""
Script download GPM IMERG & ERA5-Land rainfall data
Yêu cầu đã chạy: `earthengine authenticate`
"""
import ee
import geemap
import os

# Initialize
try:
    # geemap handles auth and project selection better
    geemap.ee_initialize()
    print("✅ GEE Initialized successfully via geemap!")
except Exception as e:
    print("❌ GEE Initialize failed. Msg:", e)
    print("👉 Đang thử authenticate lại...")
    try:
        geemap.ee_authenticate()
        geemap.ee_initialize()
    except:
        print("❌ Vẫn thất bại. Hãy kiểm tra lại Google Cloud Project.")
        exit(1)

# Cấu hình
AOI = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq('ADM1_NAME', 'Ha Tinh'))
CRS = 'EPSG:32648' # UTM Zone 48N
SCALE = 30 
OUT_DIR = "../input"

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# Danh sách sự kiện (từ recip.js)
events = [
  {'name': '01_2016_Lu_Ho_Ho',        'start': '2016-10-10', 'end': '2016-10-24'},
  {'name': '02_2016_Lu_T11_Dot2',     'start': '2016-11-12', 'end': '2016-11-26'},
  {'name': '03_2017_Bao_So_2',        'start': '2017-07-13', 'end': '2017-07-27'},
  {'name': '04_2017_ATND_Sau_Bao',    'start': '2017-10-06', 'end': '2017-10-20'},
  {'name': '05_2018_Mua_T7',          'start': '2018-07-15', 'end': '2018-07-29'},
  {'name': '06_2019_Lu_Dau_Mua',      'start': '2019-08-27', 'end': '2019-09-10'},
  {'name': '07_2019_Lu_T10',          'start': '2019-10-20', 'end': '2019-11-03'},
  {'name': '08_2020_Bao_So_5',        'start': '2020-10-15', 'end': '2020-10-29'},
  {'name': '09_2020_Lu_Dau_T10',      'start': '2020-10-03', 'end': '2020-10-17'},
  {'name': '10_2020_DAI_HONG_THUY',   'start': '2020-10-15', 'end': '2020-10-29'},
  {'name': '11_2021_Lu_T9',           'start': '2021-09-25', 'end': '2021-10-09'},
  {'name': '12_2021_Lu_T10_Dot1',     'start': '2021-10-06', 'end': '2021-10-20'},
  {'name': '13_2021_Lu_T10_Dot2',     'start': '2021-10-24', 'end': '2021-11-07'},
  {'name': '14_2022_Bao_Noru',        'start': '2022-09-25', 'end': '2022-10-09'},
  {'name': '15_2023_Lu_T9',           'start': '2023-09-18', 'end': '2023-10-02'},
  {'name': '16_2023_Lu_Vu_Quang',     'start': '2023-10-28', 'end': '2023-11-11'},
  {'name': '17_2024_Sau_Bao_Soulik',  'start': '2024-09-16', 'end': '2024-09-30'},
  {'name': '18_2025_Lu_Bat_Thuong_T5','start': '2025-05-10', 'end': '2025-05-24'},
  {'name': '19_2025_Lu_T11',          'start': '2025-11-05', 'end': '2025-11-19'}
]

def get_gpm_rain(event_name, start_date, end_date):
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    
    # GPM IMERG V07
    gpm = ee.ImageCollection('NASA/GPM_L3/IMERG_V07') \
        .filterDate(start, end) \
        .filterBounds(AOI) \
        .select('precipitation') # mm/hr
        
    # Metrics
    rain_3d = gpm.filterDate(start.advance(-3, 'day'), start) \
        .sum().multiply(0.5).rename(event_name + '_GPM_Rain_3D')
        
    rain_7d = gpm.filterDate(start.advance(-7, 'day'), start) \
        .sum().multiply(0.5).rename(event_name + '_GPM_Rain_7D')
        
    rain_max = gpm.filterDate(start.advance(-7, 'day'), start) \
        .max().multiply(0.5).rename(event_name + '_GPM_Rain_Max')
        
    return ee.Image.cat([rain_3d, rain_7d, rain_max]).clip(AOI)

def get_era5_rain(event_name, start_date, end_date):
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    
    # ERA5-Land
    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
        .filterDate(start.advance(-7, 'day'), end) \
        .filterBounds(AOI) \
        .select('total_precipitation_hourly') # meters
        
    def to_mm(img):
        return img.multiply(1000)
        
    rain_3d = era5.filterDate(start.advance(-3, 'day'), start) \
        .map(to_mm).sum().rename(event_name + '_ERA5_Rain_3D')
        
    rain_7d = era5.filterDate(start.advance(-7, 'day'), start) \
        .map(to_mm).sum().rename(event_name + '_ERA5_Rain_7D')
        
    rain_max = era5.filterDate(start.advance(-7, 'day'), start) \
        .map(to_mm).max().multiply(24).rename(event_name + '_ERA5_Rain_Max')
        
    return ee.Image.cat([rain_3d, rain_7d, rain_max]).clip(AOI)

# Download loop
print("\n🚀 Bắt đầu tạo ImageCollection...")

gpm_images = []
era5_images = []

for evt in events:
    year = int(evt['start'][:4])
    if year <= 2024:
        print(f"Processing {evt['name']}...")
        gpm_images.append(get_gpm_rain(evt['name'], evt['start'], evt['end']))
        era5_images.append(get_era5_rain(evt['name'], evt['start'], evt['end']))

# Export
print("\n⏳ Đang export ra Google Drive (vì geemap download direct có thể chậm/lỗi với ảnh lớn)...")
print("⚠️ Lưu ý: Script này sẽ submit task vào GEE server.")
print("   Bạn cần vào https://code.earthengine.google.com/tasks để bấm RUN nếu dùng Python API thuần.")
print("   Tuy nhiên, với geemap, ta có thể thử download trực tiếp nếu vùng nhỏ.")

# Thử download trực tiếp dùng geemap
region_geom = AOI.geometry()

print("\n📥 Đang thử download trực tiếp GPM (sẽ mất vài phút)...")
gpm_col = ee.ImageCollection(gpm_images).toBands()
geemap.download_ee_image(
    gpm_col,
    filename=os.path.join(OUT_DIR, "HaTinh_Rain_GPM_IMERG.tif"),
    scale=SCALE,
    region=region_geom,
    crs=CRS
)

print("\n📥 Đang thử download trực tiếp ERA5 (sẽ mất vài phút)...")
era5_col = ee.ImageCollection(era5_images).toBands()
geemap.download_ee_image(
    era5_col,
    filename=os.path.join(OUT_DIR, "HaTinh_Rain_ERA5_Land.tif"),
    scale=SCALE,
    region=region_geom,
    crs=CRS
)

print("\n✅ HOÀN THÀNH!")
