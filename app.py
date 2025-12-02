import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# --- 1. Load Data & Preprocessing (PERBAIKAN UTAMA) ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('laptop_price.csv', encoding='latin-1')
    except FileNotFoundError:
        return None

    # 1. Cleaning RAM & Weight (Standar)
    df['Ram_GB'] = df['Ram'].str.replace('GB', '').astype(int)
    df['Weight_kg'] = df['Weight'].str.replace('kg', '').astype(float)
    
    # 2. Cleaning Screen Resolution (Lebih Aman)
    # Mengambil angka resolusi pertama (Width)
    df['Resolution_Width'] = df['ScreenResolution'].str.extract(r'(\d+)x\d+').astype(float)
    
    # 3. Fitur Tambahan
    df['Touchscreen'] = df['ScreenResolution'].str.contains('Touchscreen', case=False).astype(int)
    df['IPS'] = df['ScreenResolution'].str.contains('IPS', case=False).astype(int)
    df['CPU_Speed_GHz'] = df['Cpu'].str.extract(r'(\d+\.\d+)GHz').astype(float)
    
    # 4. COMPLEX STORAGE PARSING (LOGIKA DARI NOTEBOOK)
    # Ini adalah kunci agar Mid-Range terdeteksi dengan benar
    
    # Hapus spasi berlebih di kolom Memory
    df['Memory'] = df['Memory'].astype(str).replace(r'\.0', '', regex=True)
    df["Memory"] = df["Memory"].str.replace('GB', '')
    df["Memory"] = df["Memory"].str.replace('TB', '000') # Konversi TB ke MB (simpelnya 1000, nanti dibagi)
    
    # Pisahkan jika ada dua drive (misal: 128 SSD + 1TB HDD)
    new = df["Memory"].str.split("+", n = 1, expand = True)

    # Layer 1 (Drive Pertama)
    df["first"]= new[0]
    df["first"]=df["first"].str.strip()
    
    df["Layer1HDD"] = df["first"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer1SSD"] = df["first"].apply(lambda x: 1 if "SSD" in x else 0)
    df["Layer1Hybrid"] = df["first"].apply(lambda x: 1 if "Hybrid" in x else 0)
    df["Layer1Flash_Storage"] = df["first"].apply(lambda x: 1 if "Flash Storage" in x else 0)
    
    df['first'] = df['first'].str.replace(r'\D', '', regex=True) # Hapus karakter non-angka

    # Layer 2 (Drive Kedua, jika ada)
    df["second"]= new[1]
    df["second"]=df["second"].fillna("0")
    df["second"]=df["second"].str.strip()

    df["Layer2HDD"] = df["second"].apply(lambda x: 1 if "HDD" in x else 0)
    df["Layer2SSD"] = df["second"].apply(lambda x: 1 if "SSD" in x else 0)
    df["Layer2Hybrid"] = df["second"].apply(lambda x: 1 if "Hybrid" in x else 0)
    df["Layer2Flash_Storage"] = df["second"].apply(lambda x: 1 if "Flash Storage" in x else 0)
    
    df['second'] = df['second'].str.replace(r'\D', '', regex=True) # Hapus karakter non-angka

    # Konversi ke integer
    df["first"] = df["first"].astype(int)
    df["second"] = df["second"].astype(int)

    # Hitung Total per Tipe
    df["HDD_GB"]=(df["first"]*df["Layer1HDD"]+df["second"]*df["Layer2HDD"])
    df["SSD_GB"]=(df["first"]*df["Layer1SSD"]+df["second"]*df["Layer2SSD"])
    df["Hybrid"]=(df["first"]*df["Layer1Hybrid"]+df["second"]*df["Layer2Hybrid"])
    df["Flash_Storage_GB"]=(df["first"]*df["Layer1Flash_Storage"]+df["second"]*df["Layer2Flash_Storage"])
    
    # Total Storage
    df['Total_Storage_GB'] = df['SSD_GB'] + df['HDD_GB'] + df['Hybrid'] + df['Flash_Storage_GB']
    
    # Fill NA finally
    df = df.fillna(df.median(numeric_only=True))
    return df

df = load_data()

if df is not None:
    # Setup Model Clustering
    features = ['Ram_GB', 'SSD_GB', 'Resolution_Width', 'CPU_Speed_GHz', 'Weight_kg', 'Price_euros', 'Touchscreen', 'IPS']
    X = df[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Labeling Klaster & Urutan (Budget -> Mid -> Premium)
    cluster_order_idx = df.groupby('Cluster')['Price_euros'].mean().sort_values().index
    cluster_mapping = {cluster_order_idx[0]: 'Budget/Entry', cluster_order_idx[1]: 'Mid-Range', cluster_order_idx[2]: 'High-End/Premium'}
    df['Segment'] = df['Cluster'].map(cluster_mapping)
    
    segment_order = ['Budget/Entry', 'Mid-Range', 'High-End/Premium']

    # --- Tampilan Aplikasi ---
    st.set_page_config(page_title="Laptop Market Intelligence", layout="wide")
    
    # === SIDEBAR INPUT ===
    st.sidebar.title("Panel Input")
    st.sidebar.info("Masukkan spesifikasi laptop untuk melakukan prediksi segmen.")
    
    st.sidebar.subheader("Spesifikasi Teknis")
    ram = st.sidebar.slider("RAM (GB)", 2, 64, 8)
    ssd = st.sidebar.selectbox("SSD Storage (GB)", [0, 128, 256, 512, 1024], index=2)
    hdd = st.sidebar.selectbox("HDD Storage (GB)", [0, 500, 1024, 2048], index=0)
    cpu = st.sidebar.slider("CPU Speed (GHz)", 0.9, 3.6, 2.5)
    res_width = st.sidebar.select_slider("Resolusi Layar (Width)", options=[1366, 1600, 1920, 2560, 3840], value=1920)
    weight = st.sidebar.number_input("Berat (kg)", 0.5, 5.0, 2.0)
    price = st.sidebar.number_input("Harga Estimasi (€)", 100, 5000, 1000)
    is_touch = st.sidebar.checkbox("Touchscreen")
    is_ips = st.sidebar.checkbox("IPS Panel")
    
    predict_btn = st.sidebar.button("Cek Segmen Pasar", type="primary")

    # === BAGIAN UTAMA (MAIN PAGE) ===
    st.title("Laptop Market Intelligence System")
    
    tab1, tab2 = st.tabs(["Prediksi Segmen", "Analisis Pasar"])

    # === TAB 1: PREDIKSI ===
    with tab1:
        st.header("Hasil Segmentasi Produk")
        
        if predict_btn:
            # Prediksi
            input_data = np.array([[ram, ssd, res_width, cpu, weight, price, int(is_touch), int(is_ips)]])
            input_scaled = scaler.transform(input_data)
            pred_cluster = kmeans.predict(input_scaled)[0]
            pred_segment = cluster_mapping[pred_cluster]

            st.success(f"### Laptop ini masuk ke segmen: **{pred_segment}**")
            
            # Mengambil rata-rata klaster terpilih
            cluster_stats = df[df['Cluster'] == pred_cluster][features].mean()
            
            # Benchmark Harga
            price_diff = price - cluster_stats['Price_euros']
            if price_diff > 100:
                price_status = "Di atas rata-rata (Premium)"
                price_color = "inverse"
            elif price_diff < -100:
                price_status = "Di bawah rata-rata (Value)"
                price_color = "normal"
            else:
                price_status = "Sesuai rata-rata pasar"
                price_color = "off"

            # 1. Section Harga
            st.markdown("#### 1. Analisis Harga")
            c1, c2 = st.columns(2)
            c1.metric("Rata-rata Harga Segmen", f"€{cluster_stats['Price_euros']:.0f}")
            c2.metric("Posisi Harga Anda", f"€{price}", delta=f"{price_diff:.0f} €", delta_color=price_color)
            st.caption(f"Status: {price_status}")
            st.markdown("---")

            # 2. Section Fitur (PERUBAHAN DISINI)
            st.markdown("#### 2. Analisis Fitur vs Standar Segmen")
            st.write("Perbandingan spesifikasi laptop Anda dengan rata-rata laptop lain di segmen yang sama.")

            # A. Tampilan Metric Cards (Angka Langsung)
            m1, m2, m3 = st.columns(3)
            
            # RAM Metric
            delta_ram = ram - cluster_stats['Ram_GB']
            m1.metric("RAM", f"{ram} GB", f"{delta_ram:.1f} GB")
            
            # SSD Metric
            delta_ssd = ssd - cluster_stats['SSD_GB']
            m2.metric("SSD", f"{ssd} GB", f"{delta_ssd:.0f} GB")
            
            # CPU Metric
            delta_cpu = cpu - cluster_stats['CPU_Speed_GHz']
            m3.metric("CPU Speed", f"{cpu} GHz", f"{delta_cpu:.1f} GHz")

            # B. Tampilan Visual (Chart)
            st.write("") # Spacer
            
            # Hitung Persentase performa vs rata-rata (100% = sama dengan rata-rata)
            # Hindari pembagian nol untuk SSD
            avg_ssd = cluster_stats['SSD_GB'] if cluster_stats['SSD_GB'] > 0 else 1
            
            pct_data = pd.DataFrame({
                'Spesifikasi': ['RAM', 'SSD Capacity', 'CPU Speed'],
                'Persentase': [
                    (ram / cluster_stats['Ram_GB']) * 100,
                    (ssd / avg_ssd) * 100,
                    (cpu / cluster_stats['CPU_Speed_GHz']) * 100
                ]
            })

            # Plot Chart Horizontal
            fig_comp, ax_comp = plt.subplots(figsize=(7, 1))
            # Garis referensi 100% (Rata-rata)
            ax_comp.axvline(100, color='grey', linestyle='--', alpha=0.5)
            ax_comp.text(102, -0.6, 'Rata-rata Segmen (100%)', color='grey', fontsize=8)

            # Barplot
            colors = ['#2ecc71' if x >= 100 else '#e74c3c' for x in pct_data['Persentase']]
            sns.barplot(data=pct_data, y='Spesifikasi', x='Persentase', palette=colors, ax=ax_comp)
            
            # Label Angka di Bar
            for i, v in enumerate(pct_data['Persentase']):
                ax_comp.text(v + 2, i, f"{v:.0f}%", va='center', fontweight='bold')

            ax_comp.set_xlim(0, max(150, pct_data['Persentase'].max() + 30))
            ax_comp.set_xlabel('Persentase terhadap Standar (%)')
            ax_comp.set_ylabel('')
            st.pyplot(fig_comp)

            # Logic Saran Bisnis
            if ram < cluster_stats['Ram_GB']:
                st.warning(f"**Perhatian Stok:** RAM laptop ini ({ram}GB) berada di bawah rata-rata segmen ini ({cluster_stats['Ram_GB']:.1f}GB). Pertimbangkan upgrade agar kompetitif.")
            elif ram > cluster_stats['Ram_GB'] * 1.5:
                st.info("**Selling Point:** Kapasitas RAM sangat unggul dibanding kompetitor di kelas ini. Gunakan sebagai materi promosi utama.")
        else:
            st.info("Silakan atur spesifikasi di Sidebar kiri, lalu tekan tombol 'Cek Segmen Pasar'.")
            
        # --- DEBUGGING SECTION (PENTING) ---
        # Ini untuk membuktikan bahwa 3 segmen itu ADA
        with st.expander("ℹ️ Lihat Definisi Acuan Klaster"):
            st.write("Tabel di bawah menunjukkan nilai rata-rata dari setiap segmen:")
            # Group by Segment Name agar lebih jelas
            centroids = df.groupby('Segment')[['Price_euros', 'Ram_GB', 'SSD_GB', 'CPU_Speed_GHz']].mean().reindex(segment_order)
            st.dataframe(centroids.style.format("{:.1f}"))
            st.caption("*Jika input Anda mendekati angka di baris 'Mid-Range', maka hasil prediksi akan masuk ke Mid-Range.*")

    # === TAB 2: ANALISIS ===
    with tab2:
        st.header("Analisis Profil Segmen Pasar")
        st.markdown("Wawasan mendalam untuk **Tim Pemasaran** dan **Manajemen Stok**.")

        st.subheader("1. Profil Spesifikasi per Segmen")
        chart_tab1, chart_tab2, chart_tab3 = st.tabs(["Distribusi Harga", "Distribusi RAM", "Distribusi SSD"])
        
        with chart_tab1:
            fig1, ax1 = plt.subplots(figsize=(8, 3))
            sns.boxplot(data=df, x='Segment', y='Price_euros', order=segment_order, palette='viridis', ax=ax1)
            ax1.set_title('Rentang Harga per Segmen')
            ax1.set_ylabel('Harga (€)')
            st.pyplot(fig1)

        with chart_tab2:
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            sns.boxplot(data=df, x='Segment', y='Ram_GB', order=segment_order, palette='magma', ax=ax2)
            ax2.set_title('Standar Kapasitas RAM per Segmen')
            ax2.set_ylabel('RAM (GB)')
            st.pyplot(fig2)

        with chart_tab3:
            fig3, ax3 = plt.subplots(figsize=(8, 3))
            sns.boxplot(data=df, x='Segment', y='SSD_GB', order=segment_order, palette='rocket', ax=ax3)
            ax3.set_title('Standar Kapasitas SSD per Segmen')
            ax3.set_ylabel('SSD (GB)')
            st.pyplot(fig3)

        st.markdown("---")
        st.subheader("2. Peta Posisi Pasar")
        
        col_scatter1, col_scatter2 = st.columns(2)
        with col_scatter1:
            fig4, ax4 = plt.subplots(figsize=(6, 5))
            sns.scatterplot(data=df, x='Ram_GB', y='Price_euros', hue='Segment', hue_order=segment_order, palette='viridis', ax=ax4, alpha=0.6)
            ax4.set_title('Posisi: RAM vs Harga')
            st.pyplot(fig4)

        with col_scatter2:
            fig5, ax5 = plt.subplots(figsize=(6, 5))
            sns.scatterplot(data=df, x='CPU_Speed_GHz', y='Price_euros', hue='Segment', hue_order=segment_order, palette='viridis', ax=ax5, alpha=0.6)
            ax5.set_title('Posisi: Kecepatan CPU vs Harga')
            st.pyplot(fig5)

        st.markdown("---")
        st.subheader("3. Faktor Teknis Paling Berpengaruh")
        corr_matrix = df[features].corr()
        price_corr = corr_matrix['Price_euros'].sort_values(ascending=False).drop('Price_euros')
        
        fig_corr, ax_corr = plt.subplots(figsize=(10, 3))
        sns.barplot(x=price_corr.values, y=price_corr.index, palette='coolwarm', ax=ax_corr)
        ax_corr.set_title('Korelasi Spesifikasi terhadap Harga')
        st.pyplot(fig_corr)