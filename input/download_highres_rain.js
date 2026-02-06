// ==========================================================
// GEE SCRIPT: TẢI DỮ LIỆU MƯA CAO RESOLUTION HƠN
// GPM IMERG (0.1° ~ 11km) và ERA5-Land (0.1° resampled)
// ==========================================================

var aoi = ee.FeatureCollection("FAO/GAUL/2015/level1")
    .filter(ee.Filter.eq('ADM1_NAME', 'Ha Tinh'));
Map.centerObject(aoi, 9);

var CRS_UTM = 'EPSG:32648';
var SCALE = 30; // Output resolution

// Danh sách sự kiện (copy từ recip.js)
var events = [
    { name: '01_2016_Lu_Ho_Ho', start: '2016-10-10', end: '2016-10-24' },
    { name: '02_2016_Lu_T11_Dot2', start: '2016-11-12', end: '2016-11-26' },
    { name: '03_2017_Bao_So_2', start: '2017-07-13', end: '2017-07-27' },
    { name: '04_2017_ATND_Sau_Bao', start: '2017-10-06', end: '2017-10-20' },
    { name: '05_2018_Mua_T7', start: '2018-07-15', end: '2018-07-29' },
    { name: '06_2019_Lu_Dau_Mua', start: '2019-08-27', end: '2019-09-10' },
    { name: '07_2019_Lu_T10', start: '2019-10-20', end: '2019-11-03' },
    { name: '08_2020_Bao_So_5', start: '2020-10-15', end: '2020-10-29' },
    { name: '09_2020_Lu_Dau_T10', start: '2020-10-03', end: '2020-10-17' },
    { name: '10_2020_DAI_HONG_THUY', start: '2020-10-15', end: '2020-10-29' },
    { name: '11_2021_Lu_T9', start: '2021-09-25', end: '2021-10-09' },
    { name: '12_2021_Lu_T10_Dot1', start: '2021-10-06', end: '2021-10-20' },
    { name: '13_2021_Lu_T10_Dot2', start: '2021-10-24', end: '2021-11-07' },
    { name: '14_2022_Bao_Noru', start: '2022-09-25', end: '2022-10-09' },
    { name: '15_2023_Lu_T9', start: '2023-09-18', end: '2023-10-02' },
    { name: '16_2023_Lu_Vu_Quang', start: '2023-10-28', end: '2023-11-11' },
    { name: '17_2024_Sau_Bao_Soulik', start: '2024-09-16', end: '2024-09-30' },
    { name: '18_2025_Lu_Bat_Thuong_T5', start: '2025-05-10', end: '2025-05-24' },
    { name: '19_2025_Lu_T11', start: '2025-11-05', end: '2025-11-19' }
];

// ==========================================================
// 1. GPM IMERG V07 (0.1° ~ 11km, half-hourly → sum to daily)
// ==========================================================
print('📊 OPTION A: GPM IMERG V07');
print('   Resolution: 0.1° (~11km)');
print('   Temporal: 30-minute → aggregate to daily');

function getGPM_Rain(eventName, startDate, endDate) {
    var start = ee.Date(startDate);
    var end = ee.Date(endDate);

    // GPM IMERG Final Run (best quality)
    var gpm = ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
        .filterDate(start, end)
        .filterBounds(aoi)
        .select('precipitation'); // mm/hr for 30min slot

    // Calculate metrics
    var rain_3d = gpm.filterDate(start.advance(-3, 'day'), start)
        .sum().multiply(0.5).rename(eventName + '_GPM_Rain_3D'); // 0.5 to convert to mm

    var rain_7d = gpm.filterDate(start.advance(-7, 'day'), start)
        .sum().multiply(0.5).rename(eventName + '_GPM_Rain_7D');

    var rain_max = gpm.filterDate(start.advance(-7, 'day'), start)
        .max().multiply(0.5).rename(eventName + '_GPM_Rain_Max');

    return ee.Image.cat([rain_3d, rain_7d, rain_max]).clip(aoi);
}

// ==========================================================
// 2. ERA5-Land (0.1° reanalysis, hourly)
// ==========================================================
print('📊 OPTION B: ERA5-Land');
print('   Resolution: 0.1° (~11km)');
print('   Source: ECMWF Reanalysis');

function getERA5_Rain(eventName, startDate, endDate) {
    var start = ee.Date(startDate);
    var end = ee.Date(endDate);

    // ERA5-Land hourly
    var era5 = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY')
        .filterDate(start.advance(-7, 'day'), end)
        .filterBounds(aoi)
        .select('total_precipitation_hourly'); // meters

    // Convert to mm and sum
    var toMM = function (img) {
        return img.multiply(1000); // m to mm
    };

    var rain_3d = era5.filterDate(start.advance(-3, 'day'), start)
        .map(toMM).sum().rename(eventName + '_ERA5_Rain_3D');

    var rain_7d = era5.filterDate(start.advance(-7, 'day'), start)
        .map(toMM).sum().rename(eventName + '_ERA5_Rain_7D');

    var rain_max = era5.filterDate(start.advance(-7, 'day'), start)
        .map(toMM).max().multiply(24).rename(eventName + '_ERA5_Rain_Max'); // hourly max * 24

    return ee.Image.cat([rain_3d, rain_7d, rain_max]).clip(aoi);
}

// ==========================================================
// 3. TEST FIRST EVENT
// ==========================================================
var testEvent = events[0];
print('Testing with event:', testEvent.name);

// GPM
var gpmTest = getGPM_Rain(testEvent.name, testEvent.start, testEvent.end);
print('GPM bands:', gpmTest.bandNames());
Map.addLayer(gpmTest.select(0), { min: 0, max: 200, palette: ['blue', 'yellow', 'red'] }, 'GPM Rain 3D');

// ERA5
var era5Test = getERA5_Rain(testEvent.name, testEvent.start, testEvent.end);
print('ERA5 bands:', era5Test.bandNames());
Map.addLayer(era5Test.select(0), { min: 0, max: 200, palette: ['blue', 'green', 'red'] }, 'ERA5 Rain 3D');

// ==========================================================
// 4. EXPORT ALL EVENTS
// ==========================================================

// Build multi-band images
var allGPM = [];
var allERA5 = [];

events.forEach(function (event, idx) {
    // Chỉ process events có data (GPM available from 2000, ERA5 from 1950)
    var year = parseInt(event.start.substring(0, 4));

    if (year <= 2024) { // GPM và ERA5 đều có đến 2024
        allGPM.push(getGPM_Rain(event.name, event.start, event.end));
        allERA5.push(getERA5_Rain(event.name, event.start, event.end));
    }
});

// Stack all bands
var gpmStack = ee.ImageCollection(allGPM).toBands();
var era5Stack = ee.ImageCollection(allERA5).toBands();

print('GPM total bands:', gpmStack.bandNames().size());
print('ERA5 total bands:', era5Stack.bandNames().size());

// ==========================================================
// 5. EXPORT OPTIONS
// ==========================================================

// Export GPM
Export.image.toDrive({
    image: gpmStack,
    description: 'HaTinh_Rain_GPM_IMERG',
    folder: 'HaTinh_FloodRisk',
    fileNamePrefix: 'HaTinh_Rain_GPM_IMERG',
    region: aoi.geometry(),
    scale: SCALE,
    crs: CRS_UTM,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
});

// Export ERA5
Export.image.toDrive({
    image: era5Stack,
    description: 'HaTinh_Rain_ERA5_Land',
    folder: 'HaTinh_FloodRisk',
    fileNamePrefix: 'HaTinh_Rain_ERA5_Land',
    region: aoi.geometry(),
    scale: SCALE,
    crs: CRS_UTM,
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
});

print('🔧 HƯỚNG DẪN:');
print('1. Chạy script này trong GEE Code Editor');
print('2. Click "Run" để xem preview');
print('3. Vào tab "Tasks" → Run export tasks');
print('4. Download từ Google Drive');
print('5. Thay thế CHIRPS rain raster');
