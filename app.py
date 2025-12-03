import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- 1. Load Data & Preprocessing (Robust Version) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('laptop_price.csv', encoding='latin-1')
    except FileNotFoundError:
        return None

    # 1. Cleaning RAM & Weight
    df['Ram_GB'] = df['Ram'].str.replace('GB', '').astype(int)
    df['Weight_kg'] = df['Weight'].str.replace('kg', '').astype(float)
    
    # 2. Cleaning Screen Resolution
    res = df['ScreenResolution'].str.extract(r'(\d+)x\d+')
    df['Resolution_Width'] = res[0].astype(float)
    
    # 3. Fitur Tambahan
    df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False).astype(int)
    df['IPS'] = df['ScreenResolution'].str.contains('IPS', case=False).astype(int)
    df['CPU_Speed_GHz'] = df['Cpu'].str.extract(r'(\d+\.\d+)GHz').astype(float)
    
    # 4. Robust Storage Parsing (PENTING untuk Mid-Range)
    df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)
    df["Memory"] = df["Memory"].str.replace('GB', '')
    df["Memory"] = df["Memory"].str.replace('TB', '000')
    new = df["Memory"].str.split("+", n = 1, expand = True)

    # Layer 1
    df["first"]= new[0].str.strip()
    df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
    df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
    df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)
    df['first'] = df['first'].str.replace(r'\D', '', regex=True).astype(int)

    # Layer 2
    df["second"]= new[1].fillna("0").str.strip()
    df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
    df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
    df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)
    df['second'] = df['second'].str.replace(r'\D', '', regex=True).astype(int)

    # Total per Tipe
    df["HDD_GB"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
    df["SSD_GB"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
    df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
    df["Flash_Storage_GB"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])
    df['Total_Storage_GB'] = df['SSD_GB'] + df['HDD_GB'] + df['Hybrid'] + df['Flash_Storage_GB']
    
    # Fill NA
    df = df.fillna(df.median(numeric_only=True))
    return df

# --- 2. Training Model (Cached) ---
@st.cache_resource
def train_model(df):
    features = ['Ram_GB', 'SSD_GB', 'Resolution_Width', 'CPU_Speed_GHz', 'Weight_kg', 'Price_euros', 'Touchscreen', 'IPS']
    X = df[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters
    
    # Mapping Label (Budget -> Mid -> Premium)
    cluster_order_idx = df.groupby('Cluster')['Price_euros'].mean().sort_values().index
    cluster_mapping = {
        cluster_order_idx[0]: '1. Budget/Entry', 
        cluster_order_idx[1]: '2. Mid-Range', 
        cluster_order_idx[2]: '3. High-End/Premium'
    }
    df['Segment'] = df['Cluster'].map(cluster_mapping)
    
    return scaler, kmeans, cluster_mapping, df, features

# --- Load & Train ---
df_raw = load_data()

if df_raw is not None:
    scaler, kmeans, cluster_mapping, df, features = train_model(df_raw)
    segment_order = ['1. Budget/Entry', '2. Mid-Range', '3. High-End/Premium']

    # --- UI Streamlit ---
    st.set_page_config(page_title="Laptop Market Intelligence", layout="wide")

    # === SIDEBAR INPUT ===
    st.sidebar.title("Panel Kontrol")
    st.sidebar.info("Masukkan spesifikasi laptop di sini.")
    
    st.sidebar.subheader("Spesifikasi Teknis")
    # Default value diset ke Mid-Range specs agar mudah dicoba
    ram = st.sidebar.slider("RAM (GB)", 2, 64, 8)
    ssd = st.sidebar.selectbox("SSD Storage (GB)", [0, 128, 256, 512, 1024], index=2)
    hdd = st.sidebar.selectbox("HDD Storage (GB)", [0, 500, 1024, 2048], index=0)
    cpu = st.sidebar.slider("CPU Speed (GHz)", 0.9, 3.6, 2.5)
    res_width = st.sidebar.select_slider("Resolusi Layar (Width)", options=[1366, 1600, 1920, 2560, 3840], value=1920)
    weight = st.sidebar.number_input("Berat (kg)", 0.5, 5.0, 2.0)
    price = st.sidebar.number_input("Harga Estimasi (€)", 100, 6000, 1100)
    
    is_touch = st.sidebar.checkbox("Touchscreen")
    is_ips = st.sidebar.checkbox("IPS Panel")
    
    predict_btn = st.sidebar.button("Cek Segmen Pasar", type="primary")

    # === BAGIAN UTAMA ===
    st.title("Laptop Market Intelligence System")
    
    tab1, tab2 = st.tabs(["Prediksi Segmen", "Analisis Pasar"])

    # === TAB 1: PREDIKSI ===
    with tab1:
        st.header("Hasil Segmentasi Produk")
        
        if predict_btn:
            # Prepare Input
            input_data = np.array([[ram, ssd, res_width, cpu, weight, price, int(is_touch), int(is_ips)]])
            input_scaled = scaler.transform(input_data)
            
            # Predict
            pred_cluster_id = kmeans.predict(input_scaled)[0]
            pred_segment_name = cluster_mapping[pred_cluster_id]

            # Warna teks hasil
            color = "orange" if "Mid" in pred_segment_name else ("red" if "Premium" in pred_segment_name else "green")
            st.markdown(f"### Laptop ini masuk ke segmen: :{color}[**{pred_segment_name}**]")
            
            # Stats Segmen Terpilih
            cluster_stats = df[df['Cluster'] == pred_cluster_id][features].mean()
            
            # 1. ANALISIS HARGA
            st.markdown("#### 1. Analisis Harga")
            price_diff = price - cluster_stats['Price_euros']
            price_status = "Premium Strategy" if price_diff > 100 else ("Value Strategy" if price_diff < -100 else "Market Average")
            
            c1, c2 = st.columns(2)
            c1.metric("Rata-rata Harga Segmen", f"€{cluster_stats['Price_euros']:.0f}")
            c2.metric("Posisi Harga Anda", f"€{price}", delta=f"{price_diff:.0f} €", delta_color="inverse")
            st.caption(f"Status Harga: **{price_status}**")

            st.divider()

            # 2. ANALISIS FITUR (VISUALISASI BARU)
            st.markdown("#### 2. Analisis Fitur vs Standar Segmen")
            st.write("Perbandingan performa laptop Anda terhadap rata-rata kompetitor di segmen yang sama.")

            # A. Metric Cards dengan Delta
            m1, m2, m3 = st.columns(3)
            m1.metric("RAM", f"{ram} GB", f"{ram - cluster_stats['Ram_GB']:.1f} GB")
            m2.metric("SSD", f"{ssd} GB", f"{ssd - cluster_stats['SSD_GB']:.0f} GB")
            m3.metric("CPU Speed", f"{cpu} GHz", f"{cpu - cluster_stats['CPU_Speed_GHz']:.1f} GHz")

            st.write("") # Spacer

            # B. Visualisasi Bar Chart Horizontal (Persentase)
            avg_ssd = cluster_stats['SSD_GB'] if cluster_stats['SSD_GB'] > 0 else 1 # Hindari div by zero
            
            pct_data = pd.DataFrame({
                'Spesifikasi': ['RAM Capacity', 'SSD Capacity', 'CPU Speed'],
                'Persentase': [
                    (ram / cluster_stats['Ram_GB']) * 100,
                    (ssd / avg_ssd) * 100,
                    (cpu / cluster_stats['CPU_Speed_GHz']) * 100
                ]
            })

            # Plotting
            fig_comp, ax_comp = plt.subplots(figsize=(8, 2))
            # Garis 100%
            ax_comp.axvline(100, color='gray', linestyle='--', alpha=0.6)
            ax_comp.text(102, -0.6, 'Rata-rata Segmen (100%)', color='gray', fontsize=9)

            # Bar Chart
            colors = ['#2ecc71' if x >= 100 else '#e74c3c' for x in pct_data['Persentase']]
            sns.barplot(data=pct_data, y='Spesifikasi', x='Persentase', palette=colors, ax=ax_comp)
            
            # Annotasi Angka
            for i, v in enumerate(pct_data['Persentase']):
                ax_comp.text(v + 2, i, f"{v:.0f}%", va='center', fontweight='bold')

            ax_comp.set_xlim(0, max(150, pct_data['Persentase'].max() + 40))
            ax_comp.set_xlabel('Persentase terhadap Standar Segmen (%)')
            ax_comp.set_ylabel('')
            st.pyplot(fig_comp)

            # Logic Saran Bisnis
            if ram < cluster_stats['Ram_GB']:
                st.warning(f"**Perhatian Stok:** RAM laptop ini ({ram}GB) di bawah rata-rata segmen ({cluster_stats['Ram_GB']:.1f}GB).")
            elif ram > cluster_stats['Ram_GB'] * 1.5:
                st.info("**Selling Point:** Kapasitas RAM sangat unggul. Gunakan sebagai materi promosi utama.")

        else:
            st.info("Silakan atur spesifikasi di Sidebar kiri, lalu tekan tombol 'Cek Segmen Pasar'.")
            
        # Debugging Centroid (Opsional)
        with st.expander("Lihat Data Acuan"):
            centroids = df.groupby('Segment')[['Price_euros', 'Ram_GB', 'SSD_GB', 'CPU_Speed_GHz']].mean().reindex(segment_order)
            st.dataframe(centroids.style.format("{:.1f}"))

    # === TAB 2: ANALISIS ===
    with tab2:
        st.header("Analisis Profil Segmen Pasar")
        
        st.subheader("1. Profil Spesifikasi per Segmen")
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Distribusi Harga", "Distribusi RAM", "Distribusi SSD"])
        
        with chart_tab1:
            fig1, ax1 = plt.subplots(figsize=(8, 3))
            sns.boxplot(data=df, x='Segment', y='Price_euros', order=segment_order, palette='viridis', ax=ax1)
            st.pyplot(fig1)

        with chart_tab2:
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            sns.boxplot(data=df, x='Segment', y='Ram_GB', order=segment_order, palette='magma', ax=ax2)
            st.pyplot(fig2)

        with chart_tab3:
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            sns.boxplot(data=df, x='Segment', y='SSD_GB', order=segment_order, palette='rocket', ax=ax3)
            st.pyplot(fig3)

        st.divider()
        st.subheader("2. Peta Posisi Pasar (Positioning)")
        
        c_scat1, c_scat2 = st.columns(2)
        with c_scat1:
            fig4, ax4 = plt.subplots(figsize=(6, 5))
            sns.scatterplot(data=df, x='Ram_GB', y='Price_euros', hue='Segment', hue_order=segment_order, palette='viridis', ax=ax4, alpha=0.6)
            ax4.set_title('RAM vs Harga')
            st.pyplot(fig4)

        with c_scat2:
            fig5, ax5 = plt.subplots(figsize=(6, 5))
            sns.scatterplot(data=df, x='CPU_Speed_GHz', y='Price_euros', hue='Segment', hue_order=segment_order, palette='viridis', ax=ax5, alpha=0.6)
            ax5.set_title('CPU Speed vs Harga')
            st.pyplot(fig5)

        st.divider()
        st.subheader("3. Faktor Paling Berpengaruh")
        corr_matrix = df[features].corr()
        price_corr = corr_matrix['Price_euros'].sort_values(ascending=False).drop('Price_euros')
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 3))
        sns.barplot(x=price_corr.values, y=price_corr.index, palette='coolwarm', ax=ax_corr)
        ax_corr.set_title('Korelasi Spesifikasi terhadap Harga')
        st.pyplot(fig_corr)

else:
    st.error("File 'laptop_price.csv' tidak ditemukan. Harap upload file tersebut.")