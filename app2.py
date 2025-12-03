import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy import stats

# --- 1. Simulasi Data dan Pra-pemrosesan (Sesuai Notebook) ---

# Simulasi data mentah (df) untuk mendapatkan statistik scaling
# Dalam deployment nyata, Anda harus menyimpan dan memuat scaler dan model K-Means yang telah dilatih
@st.cache_data
def load_and_prepare_data():
    # Memuat data asli (simulasi, asumsikan laptop_price.csv ada)
    # Ganti 'laptop_price.csv' dengan path file Anda jika perlu
    try:
        df = pd.read_csv('laptop_price.csv', encoding='latin-1')
    except FileNotFoundError:
        # Jika file tidak ada, buat DataFrame dummy atau gunakan path yang benar
        st.error("File 'laptop_price.csv' tidak ditemukan. Menggunakan data dummy.")
        data = {
            'laptop_ID': range(1303), 'Company': ['HP'] * 1303, 'Product': ['Notebook'] * 1303,
            'TypeName': ['Notebook'] * 1303, 'Inches': [15.6] * 1303, 'ScreenResolution': ['Full HD'] * 1303,
            'Cpu': ['Intel Core i5 2.5GHz'] * 1303, 'Ram': ['8GB'] * 1303, 'Memory': ['256GB SSD'] * 1303,
            'Gpu': ['Intel HD Graphics'] * 1303, 'OpSys': ['Windows 10'] * 1303, 'Weight': ['2.0kg'] * 1303,
            'Price_euros': [1000.0] * 1303
        }
        df = pd.DataFrame(data)


    # Langkah Pra-pemrosesan (Sesuai notebook)
    # Convert Ram to numeric
    df['Ram_GB'] = df['Ram'].str.replace('GB', '').astype(int)
    # Convert Weight to numeric
    df['Weight_kg'] = df['Weight'].str.replace('kg', '').astype(float)
    # Extract screen resolution information
    df['Resolution_Width'] = df['ScreenResolution'].str.extract(r'(\d+)x\d+').astype(float)
    df['Resolution_Height'] = df['ScreenResolution'].str.extract(r'\d+x(\d+)').astype(float)
    df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False)
    df['IPS'] = df['ScreenResolution'].str.contains('IPS', case=False)
    # Extract CPU information
    df['CPU_Brand'] = df['Cpu'].str.split().str[0]
    df['CPU_Speed_GHz'] = df['Cpu'].str.extract(r'(\d+\.\d+)GHz').astype(float)
    # Extract Memory information (menggunakan fungsi yang sama)
    def parse_memory(memory):
        total_gb = 0
        parts = str(memory).split('+')
        for part in parts:
            part = part.strip()
            if 'TB' in part:
                tb_value = part.split('TB')[0].strip()
                numbers = ''.join(filter(str.isdigit, tb_value))
                if numbers:
                    total_gb += float(numbers) * 1024
            elif 'GB' in part:
                gb_value = part.split('GB')[0].strip()
                numbers = ''.join(filter(str.isdigit, gb_value))
                if numbers:
                    total_gb += float(numbers)
        return total_gb
    def parse_ssd(memory):
        ssd_gb = 0
        parts = str(memory).split('+')
        for part in parts:
            part = part.strip()
            if 'SSD' in part:
                if 'TB' in part:
                    tb_value = part.split('TB')[0].strip()
                    numbers = ''.join(filter(str.isdigit, tb_value))
                    if numbers:
                        ssd_gb += float(numbers) * 1024
                elif 'GB' in part:
                    gb_value = part.split('GB')[0].strip()
                    numbers = ''.join(filter(str.isdigit, gb_value))
                    if numbers:
                        ssd_gb += float(numbers)
        return ssd_gb
    def parse_hdd(memory):
        hdd_gb = 0
        parts = str(memory).split('+')
        for part in parts:
            part = part.strip()
            if 'HDD' in part:
                if 'TB' in part:
                    tb_value = part.split('TB')[0].strip()
                    numbers = ''.join(filter(str.isdigit, tb_value))
                    if numbers:
                        hdd_gb += float(numbers) * 1024
                elif 'GB' in part:
                    gb_value = part.split('GB')[0].strip()
                    numbers = ''.join(filter(str.isdigit, gb_value))
                    if numbers:
                        hdd_gb += float(numbers)
        return hdd_gb
    def parse_flash(memory):
        flash_gb = 0
        parts = str(memory).split('+')
        for part in parts:
            part = part.strip()
            if 'Flash' in part:
                if 'TB' in part:
                    tb_value = part.split('TB')[0].strip()
                    numbers = ''.join(filter(str.isdigit, tb_value))
                    if numbers:
                        flash_gb += float(numbers) * 1024
                elif 'GB' in part:
                    gb_value = part.split('GB')[0].strip()
                    numbers = ''.join(filter(str.isdigit, gb_value))
                    if numbers:
                        flash_gb += float(numbers)
        return flash_gb
    
    df['Total_Storage_GB'] = df['Memory'].apply(parse_memory)
    df['SSD_GB'] = df['Memory'].apply(parse_ssd)
    df['HDD_GB'] = df['Memory'].apply(parse_hdd)
    df['Flash_Storage_GB'] = df['Memory'].apply(parse_flash)

    return df

df = load_and_prepare_data()

# Definisikan fitur-fitur clustering
clustering_features = [
    'Inches', 'Ram_GB', 'Weight_kg', 'Resolution_Width', 'CPU_Speed_GHz',
    'Total_Storage_GB', 'SSD_GB', 'HDD_GB', 'Flash_Storage_GB',
    'Price_euros',
    'Touchscreen', 'IPS'
]
numerical_features = [
    'Inches', 'Ram_GB', 'Weight_kg', 'Resolution_Width', 
    'CPU_Speed_GHz', 'Total_Storage_GB', 'SSD_GB', 'HDD_GB', 
    'Flash_Storage_GB', 'Price_euros'
]
boolean_features = ['Touchscreen', 'IPS']

# Data untuk training scaler dan model (Handle missing values by imputing median)
X_train_cluster = df[clustering_features].copy()
X_train_cluster['CPU_Speed_GHz'].fillna(X_train_cluster['CPU_Speed_GHz'].median(), inplace=True)
X_train_cluster['Resolution_Width'].fillna(X_train_cluster['Resolution_Width'].median(), inplace=True) # Walaupun tidak ada missing, langkah ini aman
X_train_cluster['Resolution_Height'].fillna(X_train_cluster['Resolution_Height'].median(), inplace=True) # Walaupun tidak ada missing, langkah ini aman

# Scaling data training
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train_cluster[numerical_features])
X_boolean_train = X_train_cluster[boolean_features].astype(int).values
X_processed_train = np.column_stack([X_scaled_train, X_boolean_train])

# K-Means Final Model (K=3)
optimal_k = 3 
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(X_processed_train)
df['Cluster'] = cluster_labels

# Ringkasan klaster untuk ditampilkan
cluster_profile = df.groupby('Cluster')[['Price_euros', 'Ram_GB', 'SSD_GB', 'CPU_Speed_GHz', 'Resolution_Width']].mean().reset_index()
cluster_count = df['Cluster'].value_counts().reset_index()
cluster_count.columns = ['Cluster', 'Count']
cluster_profile = cluster_profile.merge(cluster_count, on='Cluster')
cluster_profile = cluster_profile.sort_values(by='Price_euros')

# Definisikan profil klaster (sesuai interpretasi notebook)
cluster_map = {
    cluster_profile.iloc[0]['Cluster']: "1. Entry-Level/Budget (Rata-rata: €728)",
    cluster_profile.iloc[1]['Cluster']: "0. Mid-Range (Rata-rata: €1206)",
    cluster_profile.iloc[2]['Cluster']: "2. High-End/Premium (Rata-rata: €2150)"
}


# --- 2. Fungsi Prediksi ---

def predict_cluster(input_df, scaler, kmeans_model, numerical_features, boolean_features):
    # 1. Pilih fitur numerik
    X_input_numeric = input_df[numerical_features].copy()

    # 2. Scaling data input
    X_scaled_input = scaler.transform(X_input_numeric)

    # 3. Pilih fitur boolean (dikonversi ke int)
    X_boolean_input = input_df[boolean_features].astype(int).values

    # 4. Gabungkan (stack) fitur
    X_processed_input = np.column_stack([X_scaled_input, X_boolean_input])

    # 5. Prediksi klaster
    cluster = kmeans_model.predict(X_processed_input)[0]
    return cluster

# --- 3. Aplikasi Streamlit ---

st.set_page_config(
    page_title="Laptop Market Segmentation",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💻 Deployment Model Segmentasi Pasar Laptop")
st.markdown("---")

st.sidebar.header("Input Spesifikasi Laptop")

# Input pengguna (nilai default diambil dari statistik data)
input_data = {}

# Grup Input Numerik
st.sidebar.subheader("Spesifikasi Kinerja & Harga")
input_data['Price_euros'] = st.sidebar.number_input(
    'Harga (Euro)',
    min_value=float(df['Price_euros'].min()),
    max_value=float(df['Price_euros'].max()),
    value=float(df['Price_euros'].median()),
    step=10.0
)
input_data['Ram_GB'] = st.sidebar.number_input(
    'RAM (GB)',
    min_value=int(df['Ram_GB'].min()),
    max_value=int(df['Ram_GB'].max()),
    value=int(df['Ram_GB'].median()),
    step=2
)
input_data['CPU_Speed_GHz'] = st.sidebar.number_input(
    'Kecepatan CPU (GHz)',
    min_value=float(df['CPU_Speed_GHz'].min()),
    max_value=float(df['CPU_Speed_GHz'].max()),
    value=float(df['CPU_Speed_GHz'].median()),
    step=0.1,
    format="%.1f"
)

# Grup Input Storage
st.sidebar.subheader("Spesifikasi Storage (GB)")
input_data['Total_Storage_GB'] = st.sidebar.number_input(
    'Total Storage (GB) [Contoh: 256, 500, 1024]',
    min_value=float(df['Total_Storage_GB'].min()),
    max_value=float(df['Total_Storage_GB'].max()),
    value=float(df['Total_Storage_GB'].median()),
    step=1.0
)
input_data['SSD_GB'] = st.sidebar.number_input(
    'SSD Storage (GB) [Contoh: 0, 128, 256]',
    min_value=float(df['SSD_GB'].min()),
    max_value=float(df['SSD_GB'].max()),
    value=float(df['SSD_GB'].median()),
    step=1.0
)
input_data['HDD_GB'] = st.sidebar.number_input(
    'HDD Storage (GB) [Contoh: 0, 500, 1024]',
    min_value=float(df['HDD_GB'].min()),
    max_value=float(df['HDD_GB'].max()),
    value=float(df['HDD_GB'].median()),
    step=1.0
)
input_data['Flash_Storage_GB'] = st.sidebar.number_input(
    'Flash Storage (GB) [Contoh: 0, 128, 256]',
    min_value=float(df['Flash_Storage_GB'].min()),
    max_value=float(df['Flash_Storage_GB'].max()),
    value=0.0,
    step=1.0
)

# Grup Input Fisik & Layar
st.sidebar.subheader("Spesifikasi Fisik & Layar")
input_data['Inches'] = st.sidebar.number_input(
    'Ukuran Layar (Inches)',
    min_value=float(df['Inches'].min()),
    max_value=float(df['Inches'].max()),
    value=float(df['Inches'].median()),
    step=0.1,
    format="%.1f"
)
input_data['Weight_kg'] = st.sidebar.number_input(
    'Berat (kg)',
    min_value=float(df['Weight_kg'].min()),
    max_value=float(df['Weight_kg'].max()),
    value=float(df['Weight_kg'].median()),
    step=0.1,
    format="%.2f"
)
input_data['Resolution_Width'] = st.sidebar.number_input(
    'Resolusi Layar (Lebar Pixel)',
    min_value=float(df['Resolution_Width'].min()),
    max_value=float(df['Resolution_Width'].max()),
    value=float(df['Resolution_Width'].median()),
    step=1.0
)
input_data['Touchscreen'] = st.sidebar.checkbox('Touchscreen', value=False)
input_data['IPS'] = st.sidebar.checkbox('IPS Panel', value=False)

# Convert input data to DataFrame for prediction
input_df = pd.DataFrame([input_data])


# --- 4. Tampilkan Hasil Prediksi ---
st.header("🎯 Hasil Segmentasi (Clustering)")

if st.sidebar.button('Segmentasi Laptop', type="primary"):
    # Pastikan imputasi median untuk CPU_Speed_GHz diterapkan pada input jika ada NaN (meskipun input field memastikan tidak ada NaN)
    if input_df['CPU_Speed_GHz'].isna().any():
        input_df['CPU_Speed_GHz'].fillna(df['CPU_Speed_GHz'].median(), inplace=True)

    # Prediksi klaster
    predicted_cluster = predict_cluster(input_df, scaler, kmeans_final, numerical_features, boolean_features)
    cluster_name = cluster_map.get(predicted_cluster, "Klaster Tidak Dikenal")

    # Ambil data rata-rata klaster
    avg_price = cluster_profile[cluster_profile['Cluster'] == predicted_cluster]['Price_euros'].iloc[0]
    avg_ram = cluster_profile[cluster_profile['Cluster'] == predicted_cluster]['Ram_GB'].iloc[0]
    avg_ssd = cluster_profile[cluster_profile['Cluster'] == predicted_cluster]['SSD_GB'].iloc[0]

    st.success(f"Laptop ini termasuk dalam **Segmen Pasar: {cluster_name}**")
    
    st.markdown("### Ringkasan Klaster")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Rata-rata Harga Klaster (Acuan)", f"€{avg_price:,.2f}")
    with col2:
        st.metric("Rata-rata RAM Klaster", f"{avg_ram:.2f} GB")
    with col3:
        st.metric("Rata-rata SSD Klaster", f"{avg_ssd:.2f} GB")

    st.markdown("---")
    
    st.markdown("### Profil Lengkap Segmen Pasar")
    st.dataframe(
        cluster_profile.rename(columns={'Cluster': 'Segmen ID', 
                                        'Price_euros': 'Rata-rata Harga (€)',
                                        'Ram_GB': 'Rata-rata RAM (GB)',
                                        'SSD_GB': 'Rata-rata SSD (GB)',
                                        'CPU_Speed_GHz': 'Rata-rata CPU Speed (GHz)',
                                        'Resolution_Width': 'Rata-rata Resolusi Lebar (px)',
                                        'Count': 'Jumlah Unit dalam Klaster'}),
        hide_index=True
    )

    st.caption("Gunakan data rata-rata klaster ini untuk perbandingan strategis.")

else:
    st.info("Atur spesifikasi laptop di sidebar dan klik 'Segmentasi Laptop' untuk melihat hasilnya.")