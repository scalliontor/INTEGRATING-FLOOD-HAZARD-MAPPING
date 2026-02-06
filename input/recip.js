// ==========================================================
// 1. CẤU HÌNH VÙNG & DANH SÁCH SỰ KIỆN (GIỐNG HỆT CODE TRƯỚC)
// ==========================================================
var aoi = ee.FeatureCollection("FAO/GAUL/2015/level1")
    .filter(ee.Filter.eq('ADM1_NAME', 'Ha Tinh'));
Map.centerObject(aoi, 9);

// Danh sách sự kiện đã nới rộng ngày (Start -3, End +7)
// CHÚ Ý: Biến 'start' ở đây được dùng làm mốc T0 (Ngày bắt đầu mưa lũ)
var events = [
    { name: '01_2016_Lu_Ho_Ho', start: '2016-10-10' },
    { name: '02_2016_Lu_T11_Dot2', start: '2016-10-28' },
    { name: '03_2017_Bao_So_2', start: '2017-07-14' },
    { name: '04_2017_ATND_Sau_Bao', start: '2017-10-05' },
    { name: '05_2018_Mua_T7', start: '2018-07-12' },
    { name: '06_2019_Lu_Dau_Mua', start: '2019-08-30' },
    { name: '07_2019_Lu_T10', start: '2019-10-10' },
    { name: '08_2020_Bao_So_5', start: '2020-09-15' },
    { name: '09_2020_Lu_Dau_T10', start: '2020-10-02' },
    { name: '10_2020_DAI_HONG_THUY', start: '2020-10-15' },
    { name: '11_2021_Lu_T9', start: '2021-09-19' },
    { name: '12_2021_Lu_T10_Dot1', start: '2021-10-12' },
    { name: '13_2021_Lu_T10_Dot2', start: '2021-10-24' },
    { name: '14_2022_Bao_Noru', start: '2022-09-24' },
    { name: '15_2023_Lu_T9', start: '2023-09-22' },
    { name: '16_2023_Lu_Vu_Quang', start: '2023-10-25' },
    { name: '17_2024_Sau_Bao_Soulik', start: '2024-09-15' },
    { name: '18_2025_Lu_Bat_Thuong_T5', start: '2025-05-15' },
    { name: '19_2025_Lu_T11', start: '2025-10-25' }
];

// ==========================================================
// 2. HÀM TÍNH MƯA TỪ CHIRPS (DAILY)
// ==========================================================
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterBounds(aoi)
    .select('precipitation');

// Hàm tính Antecedent Moisture (Mưa 14 ngày trước có trọng số giảm dần)
// Công thức: Tổng (Mưa_ngày_i * w_i) với w giảm dần khi càng xa T0
var calculate_AM14 = function (t0_date) {
    var days = ee.List.sequence(1, 14); // 14 ngày trước

    var am_image = ee.ImageCollection(days.map(function (d) {
        d = ee.Number(d);
        var current_day = t0_date.advance(d.multiply(-1), 'day');
        // Trọng số w = 0.9^d (Ngày hôm qua w=0.9, 14 ngày trước w=0.2)
        var weight = ee.Number(0.9).pow(d);

        var rain_day = chirps.filterDate(current_day, current_day.advance(1, 'day'))
            .sum(); // Lấy mưa ngày đó

        return rain_day.multiply(weight); // Nhân trọng số
    })).sum().rename('Rain_AM14'); // Cộng dồn lại

    return am_image;
};

// Hàm tạo Stack mưa cho 1 sự kiện
var create_rain_features = function (ev) {
    var t0 = ee.Date(ev.start); // Ngày bắt đầu đợt lũ

    // 1. Rain 3 Day: Tổng mưa 3 ngày (T0-2 đến T0) -> Kích hoạt lũ nhanh
    var rain_3d = chirps.filterDate(t0.advance(-2, 'day'), t0.advance(1, 'day'))
        .sum().rename('Rain_3D');

    // 2. Rain 7 Day: Tổng mưa 7 ngày (T0-6 đến T0) -> Tích nước hồ đập/sông lớn
    var rain_7d = chirps.filterDate(t0.advance(-6, 'day'), t0.advance(1, 'day'))
        .sum().rename('Rain_7D');

    // 3. Rain Max 1D: Ngày mưa lớn nhất trong 7 ngày qua -> Cường độ cực đoan
    var rain_max = chirps.filterDate(t0.advance(-6, 'day'), t0.advance(1, 'day'))
        .max().rename('Rain_Max');

    // 4. AM 14D: Độ ẩm đất trước lũ
    var rain_am14 = calculate_AM14(t0);

    // Ghép lại thành ảnh 4 bands, đổi tên theo Event để dễ quản lý
    var combined = ee.Image.cat([rain_3d, rain_7d, rain_max, rain_am14])
        .toFloat() // Ép kiểu float cho nhẹ
        .clip(aoi);

    // Đổi tên band: TênEvent_TênBiến (VD: 01_2016_Lu_Ho_Ho_Rain_3D)
    var band_names = combined.bandNames().map(function (bname) {
        return ee.String(ev.name).cat('_').cat(bname);
    });

    return combined.rename(band_names);
};

// ==========================================================
// 3. CHẠY VÒNG LẶP & TẠO STACK CUỐI CÙNG
// ==========================================================
var rain_images_list = events.map(create_rain_features);
var rain_stack = ee.ImageCollection(rain_images_list).toBands();

// Sửa tên band: Bỏ prefix số thứ tự của GEE (VD: 0_01_2016... -> 01_2016...)
var new_names = rain_stack.bandNames().map(function (n) {
    return ee.String(n).replace("^\\d+_", "");
});
var rain_stack_final = rain_stack.rename(new_names);

// Kiểm tra thông tin Stack
print('Rain Stack Info:', rain_stack_final);

// ==========================================================
// 4. XUẤT DỮ LIỆU (EXPORT)
// ==========================================================
Export.image.toDrive({
    image: rain_stack_final,
    description: 'HaTinh_Rain_Stack_CHIRPS_19Events_4Vars', // 19 events * 4 vars = 76 bands
    folder: 'GEE_HaTinh_Flood',
    scale: 5566, // Độ phân giải gốc của CHIRPS (~5km). 
    // CHÚ Ý: Xuất 5km cho nhẹ, khi train ML sẽ resample sau.
    // Nếu bạn muốn khớp luôn pixel 30m thì sửa thành 30 (nhưng file sẽ rất nặng và thừa).
    region: aoi,
    crs: 'EPSG:32648',
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
});