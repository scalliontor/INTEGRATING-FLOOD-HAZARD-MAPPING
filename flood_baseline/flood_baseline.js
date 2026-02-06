// ==========================================================
// 1. KHAI BÁO VÙNG & DỮ LIỆU TĨNH
// ==========================================================
var aoi = ee.FeatureCollection("FAO/GAUL/2015/level1")
    .filter(ee.Filter.eq('ADM1_NAME', 'Ha Tinh'));
Map.centerObject(aoi, 9);

var dem = ee.Image("USGS/SRTMGL1_003").clip(aoi);
var slope = ee.Terrain.slope(dem);
// Lấy mặt nước vĩnh cửu (Sông, Hồ) để loại bỏ, chỉ giữ lại LŨ
var permanent_water = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi).eq(80);

// ==========================================================
// 2. DANH SÁCH SỰ KIỆN (ĐÃ NỚI RỘNG NGÀY ĐỂ BẮT VỆ TINH)
// ==========================================================
var events = [
    { name: '01_2016_Lu_Ho_Ho',       start: '2016-10-10', end: '2016-10-25' },
    { name: '02_2016_Lu_T11_Dot2',    start: '2016-10-28', end: '2016-11-15' },
    { name: '03_2017_Bao_So_2',       start: '2017-07-14', end: '2017-07-30' },
    { name: '04_2017_ATND_Sau_Bao',   start: '2017-10-05', end: '2017-10-25' },
    { name: '05_2018_Mua_T7',         start: '2018-07-12', end: '2018-07-30' },
    { name: '06_2019_Lu_Dau_Mua',     start: '2019-08-30', end: '2019-09-15' },
    { name: '07_2019_Lu_T10',         start: '2019-10-10', end: '2019-10-25' },
    { name: '08_2020_Bao_So_5',       start: '2020-09-15', end: '2020-09-30' },
    { name: '09_2020_Lu_Dau_T10',     start: '2020-10-02', end: '2020-10-14' },
    { name: '10_2020_DAI_HONG_THUY',  start: '2020-10-15', end: '2020-11-05' },
    { name: '11_2021_Lu_T9',          start: '2021-09-19', end: '2021-10-05' },
    { name: '12_2021_Lu_T10_Dot1',    start: '2021-10-12', end: '2021-10-25' },
    { name: '13_2021_Lu_T10_Dot2',    start: '2021-10-24', end: '2021-11-05' },
    { name: '14_2022_Bao_Noru',       start: '2022-09-24', end: '2022-10-15' },
    { name: '15_2023_Lu_T9',          start: '2023-09-22', end: '2023-10-07' },
    { name: '16_2023_Lu_Vu_Quang',    start: '2023-10-25', end: '2023-11-15' },
    { name: '17_2024_Sau_Bao_Soulik', start: '2024-09-15', end: '2024-10-05' },
    { name: '18_2025_Lu_Bat_Thuong_T5', start: '2025-05-15', end: '2025-06-05' },
    { name: '19_2025_Lu_T11',         start: '2025-10-25', end: '2025-11-15' }
];

// ==========================================================
// 3. HÀM XỬ LÝ ẢNH (LOGIC CỦA BẠN + NÂNG CẤP)
// ==========================================================
var process_event = function(ev) {
  var collection = ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(aoi)
    .filterDate(ev.start, ev.end)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));

  // Kiểm tra xem có ảnh không (tránh lỗi)
  var count = collection.size();
  
  return ee.Image(ee.Algorithms.If(
    count.gt(0),
    (function(){
      // --- LOGIC XỬ LÝ GỐC CỦA BẠN ---
      
      // Hàm lọc nhiễu biên (Quan trọng 1)
      var maskBorder = function(image) {
        var angle = image.select('angle');
        return image.updateMask(angle.gt(31).and(angle.lt(45)));
      };

      // 1. Min Composite (Lấy pixel tối nhất trong chuỗi ngày nới rộng)
      var min_img = collection.map(maskBorder).select('VH').min().clip(aoi);

      // 2. Speckle Filter
      var smoothed = min_img.focal_median(50, 'circle', 'meters');

      // 3. Threshold -19dB
      var water = smoothed.lt(-19.0);

      // 4. Slope Filter (< 10 độ)
      var flood_mask = water.updateMask(slope.lt(10));
      
      // 4.1. (BỔ SUNG) Mask sông hồ vĩnh cửu (Chỉ lấy LŨ để train model)
      var flood_only = flood_mask.updateMask(permanent_water.not());

      // 5. Morphological Clean (Quan trọng 2: Xóa đốm < 20 pixels)
      var clean_flood = flood_only.selfMask().connectedPixelCount().gte(20);
      
      // Trả về ảnh sạch sẽ, nền 0
      return clean_flood.unmask(0).rename(ev.name).toByte();
    })(),
    // Nếu không có ảnh: Trả về ảnh rỗng
    ee.Image.constant(0).rename(ev.name).clip(aoi).toByte()
  ));
};

// Map hàm xử lý qua danh sách
var flood_images_list = events.map(process_event);
var flood_stack = ee.ImageCollection(flood_images_list).toBands();

// Sửa tên band
var new_names = flood_stack.bandNames().map(function(n){ return ee.String(n).replace("^\\d+_", "") });
var flood_stack_final = flood_stack.rename(new_names);


// ==========================================================
// 4. HIỂN THỊ VISUALIZATION
// ==========================================================
events.forEach(function(ev) {
  var band_name = ev.name;
  var layer = flood_stack_final.select(band_name);
  Map.addLayer(layer.selfMask(), {palette: ['FF0000']}, band_name, false);
});
Map.addLayer(permanent_water.selfMask(), {palette:['blue']}, 'Sông/Hồ Vĩnh Cửu', true);


// ==========================================================
// 5. EXPORT STACK (CHO ML PIPELINE)
// ==========================================================
Export.image.toDrive({
  image: flood_stack_final,
  description: 'HaTinh_Flood_Stack_19Events_FullLogic',
  scale: 30,
  region: aoi,
  crs: 'EPSG:32648',
  maxPixels: 1e13
});