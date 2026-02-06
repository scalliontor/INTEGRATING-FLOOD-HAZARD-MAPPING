// ==========================================================
// 1. CẤU HÌNH VÙNG & HỆ TỌA ĐỘ
// ==========================================================
var aoi = ee.FeatureCollection("FAO/GAUL/2015/level1")
    .filter(ee.Filter.eq('ADM1_NAME', 'Ha Tinh'));
Map.centerObject(aoi, 9);

// Hệ tọa độ UTM Zone 48N (Đơn vị: MÉT chuẩn)
var CRS_UTM = 'EPSG:32648';
var SCALE = 30;

// ==========================================================
// 2. NHÓM ĐỊA HÌNH (DEM & Derivatives)
// ==========================================================
var srtm = ee.Image("USGS/SRTMGL1_003").clip(aoi);

// A. Làm mượt DEM (Giảm nhiễu hạt tiêu cho Slope/Curvature)
var smooth_kernel = ee.Kernel.gaussian({ radius: 3, sigma: 1.5, units: 'pixels' });
var dem_smooth = srtm.convolve(smooth_kernel);

// B. Các biến phái sinh cơ bản
var slope = ee.Terrain.slope(dem_smooth).rename('slope');
var aspect = ee.Terrain.aspect(dem_smooth).rename('aspect');

// C. Curvature (Độ cong - giúp phát hiện gờ/rãnh)
var curv_lap = dem_smooth.convolve(ee.Kernel.laplacian8())
    .clamp(-0.5, 0.5)
    .rename('curv_lap');

// D. Local Relief (Độ sâu tương đối - Proxy cho HAND)
// Lấy độ cao gốc TRỪ đi độ cao thấp nhất trong bán kính 2km
// Giá trị càng nhỏ -> Càng gần đáy thung lũng
var valley_bottom = srtm.focal_min({ radius: 2000, units: 'meters' });
var relief_2km = srtm.subtract(valley_bottom).rename('relief_2km');

// ==========================================================
// 3. NHÓM THỦY VĂN (Hydrology Features)
// ==========================================================

// A. Permanent Water Mask (ESA WorldCover class 80)
// Đây là biến bạn yêu cầu thêm
var worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().clip(aoi);
var water_mask = worldcover.eq(80).rename('water_mask'); // 1=Nước, 0=Đất

// B. Distance to Water (Cost Distance)
// Dùng Water Mask làm nguồn, tính khoảng cách "chi phí"
var water_source = water_mask.selfMask(); // Chỉ lấy pixel nước
var water_utm = water_source.reproject({ crs: CRS_UTM, scale: SCALE });
var cost_surface = ee.Image(1).reproject({ crs: CRS_UTM, scale: SCALE });

var dist_water = cost_surface.cumulativeCost({
    source: water_utm,
    maxDistance: 50000 // Tìm tối đa 50km
}).rename('dist_water').clip(aoi);

// C. Flow Accumulation & TWI
var hydro = ee.Image("WWF/HydroSHEDS/15ACC");
var flow_accum_raw = hydro.select('b1').clip(aoi);
var flow_accum = flow_accum_raw.rename('flow_accum');

// Tính TWI (Topographic Wetness Index)
// Công thức: ln(a / tan(b))
var slope_rad = slope.multiply(Math.PI / 180);
var twi = flow_accum_raw.add(1)
    .divide(slope_rad.tan().add(0.001)) // +0.001 để tránh chia cho 0
    .log()
    .rename('twi');

// ==========================================================
// 4. NHÓM KHÍ HẬU NỀN & MẶT PHỦ (Context Features)
// ==========================================================

// A. Mưa khí hậu (WorldClim BIO16 - Quý mưa nhiều nhất)
// Biến này giúp model biết vùng nào "vốn dĩ mưa nhiều" theo địa lý
var worldclim = ee.Image("WORLDCLIM/V1/BIO");
var precip_clim = worldclim.select('bio16')
    .resample('bilinear')
    .clip(aoi)
    .rename('precip_clim');

// B. LULC (Giữ nguyên class gốc)
var lulc = worldcover.rename('lulc');

// ==========================================================
// 5. TỔNG HỢP (STACKING) & ĐỒNG BỘ LƯỚI
// ==========================================================
// Gom tất cả vào 1 ảnh nhiều bands
var feature_stack_raw = ee.Image.cat([
    srtm.rename('elevation'), // 1
    slope,                    // 2
    aspect,                   // 3
    curv_lap,                 // 4
    relief_2km,               // 5
    twi,                      // 6
    flow_accum,               // 7 (Mới thêm)
    dist_water,               // 8
    water_mask,               // 9 (Mới thêm)
    lulc,                     // 10
    precip_clim               // 11
]).float();

// Reproject về lưới chuẩn UTM để pixel thẳng hàng 100%
var feature_stack_final = feature_stack_raw.reproject({
    crs: CRS_UTM,
    scale: SCALE
});

// ==========================================================
// 6. KIỂM TRA & HIỂN THỊ
// ==========================================================
print('Thông tin Stack Tĩnh:', feature_stack_final);

// Hiển thị thử lớp TWI và Khoảng cách sông
Map.addLayer(twi, { min: 5, max: 20, palette: ['brown', 'yellow', 'blue'] }, 'TWI (Độ ẩm địa hình)');
Map.addLayer(dist_water, { min: 0, max: 5000, palette: ['blue', 'white'] }, 'Distance to Water');

// ==========================================================
// 7. EXPORT (XUẤT FILE)
// ==========================================================
Export.image.toDrive({
    image: feature_stack_final,
    description: 'HaTinh_Static_Full_Features_11Bands',
    folder: 'GEE_HaTinh_Flood',
    scale: SCALE,
    region: aoi,
    crs: CRS_UTM,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
});