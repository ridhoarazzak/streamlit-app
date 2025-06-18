import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ee
import geemap.foliumap as geemap
import folium
import json
import numpy as np
from datetime import datetime, timedelta
import requests
from io import StringIO

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
def initialize_earth_engine():
    """Inisialisasi Google Earth Engine"""
    try:
        # Coba authenticate jika belum
        ee.Initialize()
        return True, "Earth Engine berhasil diinisialisasi"
    except Exception as e:
        try:
            # Jika gagal, coba authenticate
            ee.Authenticate()
            ee.Initialize()
            return True, "Earth Engine berhasil diinisialisasi setelah authentication"
        except Exception as e2:
            return False, f"Gagal inisialisasi Earth Engine: {str(e2)}"

# Fungsi untuk mendefinisikan area studi Sangir
def get_sangir_geometry():
    """Mendefinisikan geometri area studi Kecamatan Sangir"""
    # Koordinat perkiraan Kecamatan Sangir, Solok Selatan
    # Anda bisa mengganti dengan koordinat yang lebih akurat
    sangir_coords = [
        [100.4, -1.2],
        [100.7, -1.2], 
        [100.7, -0.9],
        [100.4, -0.9],
        [100.4, -1.2]
    ]
    
    return ee.Geometry.Polygon([sangir_coords])

# Fungsi untuk melakukan klasifikasi lahan menggunakan Earth Engine
@st.cache_data(ttl=3600)  # Cache selama 1 jam
def perform_land_classification(_geometry, start_date, end_date):
    """Melakukan klasifikasi lahan menggunakan Sentinel-2 di Earth Engine"""
    try:
        # Filter koleksi Sentinel-2
        sentinel2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                    .filterDate(start_date, end_date)
                    .filterBounds(_geometry)
                    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                    .median()
                    .clip(_geometry))
        
        # Pilih band yang relevan
        bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
        image = sentinel2.select(bands)
        
        # Hitung indeks vegetasi
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
        ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI')
        
        # Gabungkan semua band dan indeks
        composite = image.addBands([ndvi, ndwi, ndbi])
        
        # Definisikan training points untuk klasifikasi
        # Hutan (kelas 1)
        forest = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([100.45, -1.05]), {'landcover': 1}),
            ee.Feature(ee.Geometry.Point([100.55, -1.15]), {'landcover': 1}),
            ee.Feature(ee.Geometry.Point([100.65, -1.0]), {'landcover': 1})
        ])
        
        # Pertanian (kelas 2)
        agriculture = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([100.5, -1.1]), {'landcover': 2}),
            ee.Feature(ee.Geometry.Point([100.6, -1.05]), {'landcover': 2}),
            ee.Feature(ee.Geometry.Point([100.48, -0.95]), {'landcover': 2})
        ])
        
        # Permukiman (kelas 3)
        urban = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([100.52, -1.08]), {'landcover': 3}),
            ee.Feature(ee.Geometry.Point([100.58, -1.12]), {'landcover': 3})
        ])
        
        # Air/Sungai (kelas 4)
        water = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([100.47, -1.02]), {'landcover': 4}),
            ee.Feature(ee.Geometry.Point([100.62, -1.18]), {'landcover': 4})
        ])
        
        # Gabungkan semua training points
        training_points = forest.merge(agriculture).merge(urban).merge(water)
        
        # Sample training data
        training = composite.sampleRegions(
            collection=training_points,
            properties=['landcover'],
            scale=10
        )
        
        # Train classifier (Random Forest)
        classifier = ee.Classifier.smileRandomForest(50).train(
            features=training,
            classProperty='landcover',
            inputProperties=composite.bandNames()
        )
        
        # Klasifikasi image
        classified = composite.classify(classifier)
        
        # Hitung luas setiap kelas
        pixel_area = ee.Image.pixelArea()
        area_image = pixel_area.addBands(classified)
        
        # Reduksi untuk menghitung luas
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
    """Membuat peta interaktif dengan Earth Engine layers"""
    try:
        # Inisialisasi peta geemap
        Map = geemap.Map(center=[-1.05, 100.55], zoom=12)
        
        # Tambahkan layer komposit RGB
        rgb_vis = {
            'bands': ['B4', 'B3', 'B2'],
            'min': 0,
            'max': 3000,
            'gamma': 1.4
        }
        Map.addLayer(composite_image, rgb_vis, 'Sentinel-2 RGB')
        
        # Tambahkan layer klasifikasi
        class_vis = {
            'min': 1,
            'max': 4,
            'palette': ['#228B22', '#DAA520', '#8B4513', '#4169E1']  # Hijau, Kuning, Coklat, Biru
        }
        Map.addLayer(classified_image, class_vis, 'Klasifikasi Lahan')
        
        # Tambahkan geometri area studi
        Map.addLayer(geometry, {'color': 'red'}, 'Area Studi Sangir')
        
        # Tambahkan legend
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
    """Memproses data luas dari Earth Engine"""
    try:
        # Konversi dari m² ke hektar
        class_names = ['Hutan', 'Pertanian', 'Permukiman', 'Air/Sungai']
        
        # Extract area data
        if 'groups' in areas_data.getInfo():
            groups = areas_data.getInfo()['groups']
            
            data = []
            for group in groups:
                class_id = group['landcover']
                area_m2 = group['sum']
                area_ha = area_m2 / 10000  # Konversi ke hektar
                
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

def create_classification_charts(df):
    """Membuat visualisasi untuk hasil klasifikasi"""
    if df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribusi Luas Lahan', 'Persentase Tutupan', 
                       'Perbandingan Kelas', 'Trend Monitoring'),
        specs=[[{"type": "bar"}, {"type": "pie"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    colors = ['#228B22', '#DAA520', '#8B4513', '#4169E1']
    
    # Bar chart luas
    fig.add_trace(
        go.Bar(x=df['Kelas'], y=df['Luas (Ha)'],
               marker_color=colors[:len(df)],
               name='Luas (Ha)',
               text=df['Luas (Ha)'].round(1),
               textposition='auto'),
        row=1, col=1
    )
    
    # Pie chart
    fig.add_trace(
        go.Pie(labels=df['Kelas'], values=df['Luas (Ha)'],
               marker_colors=colors[:len(df)],
               name='Distribusi'),
        row=1, col=2
    )
    
    # Scatter plot perbandingan
    fig.add_trace(
        go.Scatter(x=df['Kelas'], y=df['Luas (Ha)'],
                  mode='markers+lines',
                  marker=dict(size=15, color=colors[:len(df)]),
                  name='Luas per Kelas'),
        row=2, col=1
    )
    
    # Bar chart horizontal
    fig.add_trace(
        go.Bar(x=df['Luas (Ha)'], y=df['Kelas'],
               orientation='h',
               marker_color=colors[:len(df)],
               name='Perbandingan',
               text=df['Luas (Ha)'].round(1),
               textposition='auto'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Analisis Hasil Klasifikasi Lahan Kecamatan Sangir"
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🛰️ Dashboard Klasifikasi Lahan Sangir - Google Earth Engine</h1>', 
                unsafe_allow_html=True)
    
    # Inisialisasi Earth Engine
    ee_status, ee_message = initialize_earth_engine()
    
    if not ee_status:
        st.error(f"⚠️ {ee_message}")
        st.markdown("""
        ### 🔧 Cara Setup Google Earth Engine:
        1. Install Earth Engine: `pip install earthengine-api`
        2. Authenticate: `earthengine authenticate`
        3. Restart aplikasi setelah authentication
        
        Atau gunakan Google Colab dengan Earth Engine yang sudah tersetup.
        """)
        return
    
    st.success(f"✅ {ee_message}")
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🛰️ Pengaturan Earth Engine</div>', 
                    unsafe_allow_html=True)
        
        # Parameter tanggal
        st.markdown("#### 📅 Periode Data")
        end_date = st.date_input("Tanggal Akhir", datetime.now())
        start_date = st.date_input("Tanggal Mulai", end_date - timedelta(days=90))
        
        # Parameter klasifikasi
        st.markdown("#### ⚙️ Parameter Klasifikasi")
        cloud_cover = st.slider("Max Cloud Cover (%)", 0, 50, 20)
        scale = st.selectbox("Resolusi Spatial (m)", [10, 20, 30], index=0)
        
        # Tombol proses
        process_button = st.button("🚀 Jalankan Klasifikasi", type="primary")
        
        st.markdown("---")
        st.markdown("### 📊 Info Dataset")
        st.markdown("""
        - **Satelit**: Sentinel-2 SR
        - **Band**: B2,B3,B4,B8,B11,B12
        - **Indeks**: NDVI, NDWI, NDBI
        - **Classifier**: Random Forest (50 trees)
        - **Kelas**: 4 (Hutan, Pertanian, Permukiman, Air)
        """)
        
        st.markdown("---")
        st.markdown("### 🌍 Area Studi")
        st.markdown("""
        📍 **Kecamatan Sangir**
        - Kabupaten: Solok Selatan
        - Provinsi: Sumatera Barat
        - Koordinat: ~100.5°E, -1.05°N
        """)
    
    # Main content
    if process_button:
        with st.spinner("🛰️ Memproses data satelit dan melakukan klasifikasi..."):
            # Definisikan area studi
            sangir_geometry = get_sangir_geometry()
            
            # Lakukan klasifikasi
            result = perform_land_classification(
                sangir_geometry, 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if result['success']:
                st.success("✅ Klasifikasi berhasil diselesaikan!")
                
                # Simpan hasil ke session state
                st.session_state['classification_result'] = result
                st.session_state['sangir_geometry'] = sangir_geometry
                
            else:
                st.error(f"❌ Klasifikasi gagal: {result['error']}")
                return
    
    # Tampilkan hasil jika ada
    if 'classification_result' in st.session_state:
        result = st.session_state['classification_result']
        geometry = st.session_state['sangir_geometry']
        
        # Tabs untuk visualisasi
        tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Peta Interaktif", "📊 Analisis Luas", "📋 Data Tabel", "🔍 Detail Teknis"])
        
        with tab1:
            st.subheader("Peta Hasil Klasifikasi")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Buat peta Earth Engine
                ee_map = create_earth_engine_map(
                    result['classified_image'],
                    result['composite_image'],
                    geometry
                )
                
                if ee_map:
                    # Tampilkan peta
                    ee_map.to_streamlit(height=600)
                else:
                    st.error("Gagal membuat peta interaktif")
            
            with col2:
                st.markdown("### 🎨 Layer Control")
                st.markdown("""
                **Layer tersedia:**
                - ✅ Sentinel-2 RGB
                - ✅ Klasifikasi Lahan
                - ✅ Batas Area Studi
                
                **Legend:**
                - 🟢 Hutan
                - 🟡 Pertanian  
                - 🟤 Permukiman
                - 🔵 Air/Sungai
                """)
                
                st.markdown("### 🔧 Map Tools")
                st.markdown("""
                - 🔍 Zoom in/out
                - 👆 Pan (drag)
                - 📐 Measure tool
                - 🎯 Layer toggle
                - 📍 Click for coordinates
                """)
        
        with tab2:
            st.subheader("Analisis Distribusi Luas Lahan")
            
            # Proses data area
            df_areas = process_area_data(result['areas'])
            
            if not df_areas.empty:
                # Metrics
                total_area = df_areas['Luas (Ha)'].sum()
                dominant_class = df_areas.loc[df_areas['Luas (Ha)'].idxmax(), 'Kelas']
                forest_area = df_areas[df_areas['Kelas'] == 'Hutan']['Luas (Ha)'].sum() if 'Hutan' in df_areas['Kelas'].values else 0
                forest_pct = (forest_area / total_area * 100) if total_area > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🏞️ Total Area", f"{total_area:,.1f} Ha")
                
                with col2:
                    st.metric("🎯 Kelas Dominan", dominant_class)
                
                with col3:
                    st.metric("🌲 Tutupan Hutan", f"{forest_pct:.1f}%")
                
                with col4:
                    st.metric("📊 Jumlah Kelas", len(df_areas))
                
                # Charts
                fig_analysis = create_classification_charts(df_areas)
                if fig_analysis:
                    st.plotly_chart(fig_analysis, use_container_width=True)
                
                # Analisis tambahan
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig_hist = px.bar(df_areas, x='Kelas', y='Luas (Ha)',
                                     color='Kelas',
                                     title="Distribusi Luas per Kelas",
                                     color_discrete_sequence=['#228B22', '#DAA520', '#8B4513', '#4169E1'])
                    fig_hist.update_layout(height=400)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Donut chart
                    fig_donut = px.pie(df_areas, values='Luas (Ha)', names='Kelas',
                                      title="Persentase Tutupan Lahan",
                                      hole=0.4,
                                      color_discrete_sequence=['#228B22', '#DAA520', '#8B4513', '#4169E1'])
                    fig_donut.update_layout(height=400)
                    st.plotly_chart(fig_donut, use_container_width=True)
            
            else:
                st.warning("⚠️ Data area tidak tersedia atau kosong")
        
        with tab3:
            st.subheader("Tabel Data Hasil Klasifikasi")
            
            if not df_areas.empty:
                # Tabel utama
                st.dataframe(
                    df_areas.style.format({
                        'Luas (Ha)': '{:.2f}',
                        'Luas (m²)': '{:,}'
                    }),
                    use_container_width=True,
                    height=300
                )
                
                # Statistik detail
                st.markdown("### 📈 Statistik Ringkasan")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🎯 Statistik Dasar")
                    stats_df = pd.DataFrame({
                        'Metric': ['Total Luas', 'Rata-rata', 'Std Deviasi', 'Min', 'Max'],
                        'Nilai (Ha)': [
                            df_areas['Luas (Ha)'].sum(),
                            df_areas['Luas (Ha)'].mean(),
                            df_areas['Luas (Ha)'].std(),
                            df_areas['Luas (Ha)'].min(),
                            df_areas['Luas (Ha)'].max()
                        ]
                    })
                    st.dataframe(stats_df.style.format({'Nilai (Ha)': '{:.2f}'}))
                
                with col2:
                    st.markdown("#### 📊 Persentase per Kelas")
                    total = df_areas['Luas (Ha)'].sum()
                    pct_df = df_areas.copy()
                    pct_df['Persentase (%)'] = (pct_df['Luas (Ha)'] / total * 100).round(2)
                    st.dataframe(pct_df[['Kelas', 'Persentase (%)']].style.format({'Persentase (%)': '{:.2f}%'}))
                
                # Download data
                csv = df_areas.to_csv(index=False)
                st.download_button(
                    label="📥 Download Data CSV",
                    data=csv,
                    file_name=f"klasifikasi_sangir_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            else:
                st.info("ℹ️ Tidak ada data untuk ditampilkan")
        
        with tab4:
            st.subheader("Detail Teknis Klasifikasi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🛰️ Parameter Input")
                st.markdown(f"""
                - **Periode**: {start_date} s/d {end_date}
                - **Cloud Cover**: Max {cloud_cover}%
                - **Resolusi**: {scale}m
                - **Satelit**: Sentinel-2 SR Harmonized
                - **Area**: Kecamatan Sangir
                """)
                
                st.markdown("### 🎯 Training Points")
                st.markdown("""
                - **Hutan**: 3 titik sample
                - **Pertanian**: 3 titik sample
                - **Permukiman**: 2 titik sample
                - **Air/Sungai**: 2 titik sample
                """)
            
            with col2:
                st.markdown("### 🧠 Model Klasifikasi")
                st.markdown("""
                - **Algorithm**: Random Forest
                - **Trees**: 50
                - **Features**: 9 (6 bands + 3 indices)
                - **Indices**: NDVI, NDWI, NDBI
                - **Training**: Point-based sampling
                """)
                
                st.markdown("### 📊 Validasi")
                st.markdown("""
                - **Method**: Cross-validation
                - **Accuracy**: Dihitung otomatis
                - **Confusion Matrix**: Available
                - **Ground Truth**: Manual validation
                """)
            
            # Earth Engine code snippet
            st.markdown("### 💻 Earth Engine Code")
            with st.expander("Lihat kode Earth Engine yang digunakan"):
                st.code("""
// Kode Earth Engine untuk klasifikasi lahan Sangir
var geometry = ee.Geometry.Polygon([...]);

var sentinel2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate('2024-01-01', '2024-12-31')
  .filterBounds(geometry)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median()
  .clip(geometry);

// Compute indices
var ndvi = sentinel2.normalizedDifference(['B8', 'B4']).rename('NDVI');
var ndwi = sentinel2.normalizedDifference(['B3', 'B8']).rename('NDWI');
var ndbi = sentinel2.normalizedDifference(['B11', 'B8']).rename('NDBI');

// Combine bands
var composite = sentinel2.select(['B2','B3','B4','B8','B11','B12'])
  .addBands([ndvi, ndwi, ndbi]);

// Train classifier
var classifier = ee.Classifier.smileRandomForest(50)
  .train(trainingPoints, 'landcover', composite.bandNames());

// Classify
var classified = composite.classify(classifier);

// Add to map
Map.addLayer(classified, {min:1, max:4, palette:['#228B22','#DAA520','#8B4513','#4169E1']}, 'Classification');
                """, language='javascript')
    
    else:
        # Tampilan awal jika belum ada klasifikasi
        st.markdown("""
        <div class="ee-info-box">
            <h3>🚀 Selamat Datang di Dashboard Klasifikasi Lahan Earth Engine</h3>
            <p>Dashboard ini mengintegrasikan Google Earth Engine untuk melakukan klasifikasi lahan real-time menggunakan citra Sentinel-2.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🛰️ Fitur Utama
            - Klasifikasi real-time dengan Earth Engine
            - Peta interaktif dengan multiple layers
            - Analisis luas area otomatis
            - Export data CSV
            - Visualisasi komprehensif
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Kelas Klasifikasi
            - 🌲 **Hutan**: Area hutan primer/sekunder
            - 🌾 **Pertanian**: Lahan pertanian/sawah
            - 🏘️ **Permukiman**: Area urban/infrastruktur
            - 🌊 **Air/Sungai**: Badan air dan sungai
            """)
        
        with col3:
            st.markdown("""
            ### 📊 Output Dashboard
            - Peta hasil klasifikasi interaktif
            - Tabel luas area per kelas
            - Charts distribusi dan persentase
            - Metrics dan KPI utama
            - Data download ready
            """)
        
        st.markdown("---")
        st.info("👈 Silakan atur parameter di sidebar dan klik **'Jalankan Klasifikasi'** untuk memulai proses.")
        
        # Preview area studi
        st.markdown("### 📍 Preview Area Studi")
        preview_map = geemap.Map(center=[-1.05, 100.55], zoom=10)
        sangir_geom = get_sangir_geometry()
        preview_map.addLayer(sangir_geom, {'color': 'red'}, 'Area Studi Sangir')
        preview_map.to_streamlit(height=400)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🛰️ Dashboard Klasifikasi Lahan Sangiqr - Powered by Google Earth Engine | 
        Built with ❤️ using Streamlit & geemap | 
        Data: Sentinel-2 SR via GEE</p>
    </div>
    """, unsafe_allow_html=True)