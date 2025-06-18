import streamlit as st
import pandas as pd
import plotly.express as px
import ee
import geemap.foliumap as geemap
from datetime import datetime, timedelta

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Klasifikasi Lahan Sangir - Earth Engine",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inisialisasi Earth Engine
@st.cache_resource
def initialize_earth_engine(project='ee-mrgridhoarazzak'):
    try:
        ee.Initialize(project='ee-mrgridhoarazzak)
        return True, "Earth Engine berhasil diinisialisasi"
    except Exception:
        try:
            ee.Authenticate()
            ee.Initialize()
            return True, "Earth Engine berhasil diinisialisasi setelah authentication"
        except Exception as e2:
            return False, f"Gagal inisialisasi Earth Engine: {str(e2)}"

# Area Studi Sangir
def get_sangir_geometry():
    sangir_coords = [
        [100.4, -1.2], [100.7, -1.2], 
        [100.7, -0.9], [100.4, -0.9], 
        [100.4, -1.2]
    ]
    return ee.Geometry.Polygon([sangir_coords])

# Fungsi klasifikasi
@st.cache_data(ttl=3600)
def perform_land_classification(_geometry, start_date, end_date):
    try:
        sentinel2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterDate(start_date, end_date)
                    .filterBounds(_geometry)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                    .median()
                    .clip(_geometry))

        bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        image = sentinel2.select(bands)

        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')

        composite = image.addBands([ndvi, ndwi, ndbi])

        forest = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([100.45, -1.05]), {'landcover': 1}),
            ee.Feature(ee.Geometry.Point([100.55, -1.15]), {'landcover': 1}),
            ee.Feature(ee.Geometry.Point([100.65, -1.0]), {'landcover': 1})
        ])

        agriculture = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([100.5, -1.1]), {'landcover': 2}),
            ee.Feature(ee.Geometry.Point([100.6, -1.05]), {'landcover': 2}),
            ee.Feature(ee.Geometry.Point([100.48, -0.95]), {'landcover': 2})
        ])

        urban = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([100.52, -1.08]), {'landcover': 3}),
            ee.Feature(ee.Geometry.Point([100.58, -1.12]), {'landcover': 3})
        ])

        water = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([100.47, -1.02]), {'landcover': 4}),
            ee.Feature(ee.Geometry.Point([100.62, -1.18]), {'landcover': 4})
        ])

        training_points = forest.merge(agriculture).merge(urban).merge(water)

        training = composite.sampleRegions(
            collection=training_points,
            properties=['landcover'],
            scale=10
        )

        classifier = ee.Classifier.smileRandomForest(50).train(
            features=training,
            classProperty='landcover',
            inputProperties=composite.bandNames()
        )

        classified = composite.classify(classifier)

        pixel_area = ee.Image.pixelArea()
        area_image = pixel_area.addBands(classified)

        areas = area_image.reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=1, groupName='landcover'),
            geometry=_geometry,
            scale=10,
            maxPixels=1e9
        )

        return {
            'classified_image': classified,
            'composite_image': composite,
            'areas': areas,
            'success': True
        }

    except Exception as e:
        return {'error': str(e), 'success': False}

def create_earth_engine_map(classified_image, composite_image, geometry):
    try:
        Map = geemap.Map(center=[-1.05, 100.55], zoom=12)

        Map.addLayer(composite_image, {
            'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000, 'gamma': 1.4
        }, 'Sentinel-2 RGB')

        Map.addLayer(classified_image, {
            'min': 1, 'max': 4,
            'palette': ['#228B22', '#DAA520', '#8B4513', '#4169E1']
        }, 'Klasifikasi Lahan')

        Map.addLayer(geometry, {'color': 'red'}, 'Area Studi Sangir')

        Map.add_legend(legend_dict={
            'Hutan': '#228B22',
            'Pertanian': '#DAA520',
            'Permukiman': '#8B4513',
            'Air/Sungai': '#4169E1'
        }, title='Kelas Lahan')

        return Map
    except Exception as e:
        st.error(f"Gagal membuat peta: {str(e)}")
        return None

def process_area_data(areas_data):
    try:
        class_names = ['Hutan', 'Pertanian', 'Permukiman', 'Air/Sungai']
        if 'groups' in areas_data.getInfo():
            groups = areas_data.getInfo()['groups']
            data = []
            for group in groups:
                class_id = group['landcover']
                area_m2 = group['sum']
                area_ha = area_m2 / 10000
                if 1 <= class_id <= 4:
                    data.append({
                        'Kelas': class_names[class_id - 1],
                        'Luas (Ha)': round(area_ha, 2),
                        'Luas (m²)': int(area_m2),
                        'Kelas_ID': class_id
                    })
            return pd.DataFrame(data)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal memproses data area: {str(e)}")
        return pd.DataFrame()

def main():
    st.title("Dashboard Klasifikasi Lahan Sangir 🛰️")

    ok, msg = initialize_earth_engine()
    if not ok:
        st.error(msg)
        return
    st.success(msg)

    st.sidebar.header("Filter Tanggal")
    today = datetime.today()
    start_date = st.sidebar.date_input("Tanggal Mulai", today - timedelta(days=30))
    end_date = st.sidebar.date_input("Tanggal Akhir", today)

    if start_date > end_date:
        st.sidebar.error("Tanggal mulai harus sebelum tanggal akhir.")
        return

    geometry = get_sangir_geometry()

    with st.spinner("Melakukan klasifikasi lahan..."):
        result = perform_land_classification(
            geometry,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )

    if not result["success"]:
        st.error(f"Terjadi kesalahan saat klasifikasi: {result.get('error', 'Tidak diketahui')}")
        return

    Map = create_earth_engine_map(result['classified_image'], result['composite_image'], geometry)
    if Map:
        Map.to_streamlit(height=500)

    df_area = process_area_data(result["areas"])
    if not df_area.empty:
        st.subheader("Luas Kelas Lahan (Ha)")
        st.dataframe(df_area[['Kelas', 'Luas (Ha)']])
        fig = px.pie(df_area, names="Kelas", values="Luas (Ha)",
                     color_discrete_sequence=['#228B22', '#DAA520', '#8B4513', '#4169E1'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Tidak ada data luas area.")

if __name__ == "__main__":
    main()
