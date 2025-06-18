import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sidebar-header {
        color: #1f77b4;
        font-weight: bold;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
    }
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(90deg, #f8f9fa, #e9ecef);
        border-radius: 10px 10px 0 0;
        border: 1px solid #dee2e6;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1f77b4, #0d6efd);
        color: white;
    }
    .ee-info-box {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(45deg, #ffc107, #fd7e14);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Inisialisasi Earth Engine
@st.cache_resource
def initialize_earth_engine(project='ee-mrgridhoarazzak'):
    """Inisialisasi Google Earth Engine"""
    try:
        ee.Initialize()
        return True, "Earth Engine berhasil diinisialisasi"
    except Exception as e:
        try:
            ee.Authenticate()
            ee.Initialize()
            return True, "Earth Engine berhasil diinisialisasi setelah authentication"
        except Exception as e2:
            return False, f"Gagal inisialisasi Earth Engine: {str(e2)}"

# Mendefinisikan area studi Sangir
def get_sangir_geometry():
    sangir_coords = [
        [100.4, -1.2],
        [100.7, -1.2], 
        [100.7, -0.9],
        [100.4, -0.9],
        [100.4, -1.2]
    ]
    return ee.Geometry.Polygon([sangir_coords])

# Fungsi klasifikasi lahan menggunakan Earth Engine
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
            reducer=ee.Reducer.sum().group(
                groupField=1,
                groupName='landcover'
            ),
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
        return {
            'error': str(e),
            'success': False
        }

def create_earth_engine_map(classified_image, composite_image, geometry):
    try:
        Map = geemap.Map(center=[-1.05, 100.55], zoom=12)
        
        rgb_vis = {
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 3000,
            'gamma': 1.4
        }
        Map.addLayer(composite_image, rgb_vis, 'Sentinel-2 RGB')
        
        class_vis = {
            'min': 1,
            'max': 4,
            'palette': ['#228B22', '#DAA520', '#8B4513', '#4169E1']
        }
        Map.addLayer(classified_image, class_vis, 'Klasifikasi Lahan')
        
        Map.addLayer(geometry, {'color': 'red'}, 'Area Studi Sangir')
        
        legend_dict = {
            'Hutan': '#228B22',
            'Pertanian': '#DAA520', 
            'Permukiman': '#8B4513',
            'Air/Sungai': '#4169E1'
        }
        Map.add_legend(legend_dict=legend_dict, title='Kelas Lahan')
        
        return Map
        
    except Exception as e:
        st.error(f"Error membuat peta: {str(e)}")
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
        st.error(f"Error memproses data area: {str(e)}")
        return pd.DataFrame()

# Main app UI
def main():
    st.markdown('<h1 class="main-header">Dashboard Klasifikasi Lahan Sangir</h1>', unsafe_allow_html=True)
    
    # Inisialisasi EE
    ok, msg = initialize_earth_engine()
    if not ok:
        st.error(msg)
        return
    else:
        st.info(msg)
    
    # Sidebar untuk input tanggal
    st.sidebar.markdown('<div class="sidebar-header">Filter Tanggal</div>', unsafe_allow_html=True)
    today = datetime.today()
    start_date = st.sidebar.date_input('Tanggal mulai', today - timedelta(days=30))
    end_date = st.sidebar.date_input('Tanggal akhir', today)
    
    if start_date > end_date:
        st.sidebar.error("Tanggal mulai harus sebelum tanggal akhir.")
        return
    
    geometry = get_sangir_geometry()
    
    with st.spinner('Melakukan klasifikasi lahan...'):
        result = perform_land_classification(geometry, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if not result['success']:
        st.error(f"Error memproses klasifikasi: {result.get('error', 'Tidak diketahui')}")
        return
    
    classified_image = result['classified_image']
    composite_image = result['composite_image']
    areas = result['areas']
    
    # Tampilkan peta
    Map = create_earth_engine_map(classified_image, composite_image, geometry)
    if Map:
        Map.to_streamlit(height=500)
    
    # Proses data area dan tampilkan dalam tabel
    df_area = process_area_data(areas)
    if not df_area.empty:
        st.markdown('### Luas Kelas Lahan (hektar)')
        st.dataframe(df_area[['Kelas', 'Luas (Ha)']])
        
        # Grafik pie chart
        fig = px.pie(df_area, names='Kelas', values='Luas (Ha)', 
                     color_discrete_sequence=['#228B22', '#DAA520', '#8B4513', '#4169E1'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data luas area tidak tersedia.")

if __name__ == "__main__":
    main()
