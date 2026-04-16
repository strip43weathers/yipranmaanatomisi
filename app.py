import streamlit as st
import pandas as pd
import plotly.express as px

# Modülleri içe aktarıyoruz
from src.data_loader import load_and_preprocess_data
from src.synthetic_data import generate_synthetic_data
from src.model_pipeline import train_baldness_model

# Sayfa Sekmesi Ayarları
st.set_page_config(page_title="Saç Dökülme Analizi", page_icon="🧬", layout="wide")


@st.cache_resource
def prepare_ai():
    # KENDI YENI DOSYANIN ADI BURADA (Lütfen klasöründeki isminle birebir aynı olduğundan emin ol)
    df = load_and_preprocess_data("data/Tuğçe proje ödevi.xlsx")

    # Meslek isimlerini (temiz hallerini) alalım
    unique_professions = sorted(df['meslek'].unique().tolist())

    # 1000 kişilik sentetik veri
    df_large = generate_synthetic_data(df, target_size=1000)

    # Model eğitimi
    model, acc, importances, all_features, meslek_cols = train_baldness_model(df_large)

    return model, acc, importances, all_features, meslek_cols, unique_professions, df_large


# Arka planda verileri ve modeli yükle
model, accuracy, importances, all_features, meslek_cols, professions, df_large = prepare_ai()

# ----- WEB SİTESİ ARAYÜZÜ (UI) -----
st.title("🧬 Modern Çalışma Hayatı: Saç Dökülme Tahminleyicisi")
st.markdown(f"**Yapay Zeka Doğruluk Oranı:** %{accuracy * 100:.1f} *(1000 Kişilik Sentetik Veri ile Eğitildi)*")

st.sidebar.header("Kişisel Bilgilerinizi Girin")

yas = st.sidebar.slider("Yaşınız", 18, 80, 30)
secilen_meslek = st.sidebar.selectbox("Mesleğiniz", professions)  # YENİ MESLEK SEÇİCİ
calisma_saati = st.sidebar.slider("Günlük Çalışma Saatiniz", 1.0, 24.0, 8.0, step=0.5)
stres_seviyesi = st.sidebar.slider("İşteki Stres Seviyeniz (1-5)", 1, 5, 3)

genetik = st.sidebar.radio("Ailenizde kellik var mı?", ["Hayır", "Evet"])
sigara = st.sidebar.radio("Sigara kullanıyor musunuz?", ["Hayır", "Evet"])
alkol = st.sidebar.radio("Alkol kullanıyor musunuz?", ["Hayır", "Evet"])
cinsiyet = st.sidebar.radio("Cinsiyetiniz", ["Kadın", "Erkek"])

# Hesapla Butonu
if st.button("Saç Dökülme İhtimalimi Hesapla 🔍", use_container_width=True):

    # Kullanıcı verilerini modele uygun hale getirme
    user_input = {
        'yas': [yas],
        'calisma_saati': [calisma_saati],
        'stres_seviyesi': [stres_seviyesi],
        'genetik_kellik': [1 if genetik == "Evet" else 0],
        'sigara': [1 if sigara == "Evet" else 0],
        'alkol': [1 if alkol == "Evet" else 0],
        'cinsiyet_encoded': [1 if cinsiyet == "Erkek" else 0]
    }

    # Meslek One-Hot Sütunlarını Doldurma
    for col in meslek_cols:
        user_input[col] = [1 if col == f"meslek_{secilen_meslek}" else 0]

    # DataFrame'i modelin istediği sıraya sok
    user_df = pd.DataFrame(user_input)[all_features]

    # Tahmin
    prediction_prob = model.predict_proba(user_df)[0][1]

    st.subheader("Yapay Zeka Analiz Sonucu:")
    if prediction_prob > 0.50:
        st.error(f"⚠️ **Dikkat!** Kellik / saç dökülmesi riskiniz: **%{prediction_prob * 100:.1f}**")
        st.write("Yapay zekaya göre saç folikülleriniz baskı altında.")
    else:
        st.success(f"✅ **Güvendesiniz!** Kellik / saç dökülmesi riskiniz: **%{prediction_prob * 100:.1f}**")
        st.write("Genetiğiniz ve çalışma hayatınızdaki dengeniz şu an için saçlarınızı koruyor.")

# ==========================================
# 5 PROFESYONEL VERİ BİLİMİ GRAFİĞİ VE OKUMA REHBERLERİ
# ==========================================
st.divider()

# Grafik 1 ve 2'yi yan yana şık göstermek için sütunlara böldük
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 1. Karar Faktörleri (Feature Importance)")
    st.markdown("*Yapay zeka 'kellik' kararını verirken neye bakıyor?*")

    top_features = importances[~importances['Özellik'].str.startswith('meslek_')].head(5)
    st.bar_chart(top_features.set_index('Özellik'))

    # OKUMA REHBERİ
    st.info(
        "**Nasıl Okunur?** Çubuk ne kadar uzunsa, o özellik saç dökülmesinde o kadar belirleyicidir. Örneğin; Genetik ve Yaş genellikle en üstte yer alarak asıl başrol oyuncuları olduklarını kanıtlarken, Stres ve Çalışma Saati onları takip eden çevresel faktörlerdir.")

with col2:
    st.subheader("🕸️ 2. Yaşam Tarzı DNA'sı (Radar Analizi)")
    st.markdown("*Kel olanlar ile sağlıklı saçlara sahip olanların profil farkı nedir?*")

    radar_df = df_large.groupby('hedef_kellik')[['yas', 'calisma_saati', 'stres_seviyesi']].mean().reset_index()
    radar_df['yas'] = (radar_df['yas'] / 80) * 100
    radar_df['calisma_saati'] = (radar_df['calisma_saati'] / 24) * 100
    radar_df['stres_seviyesi'] = (radar_df['stres_seviyesi'] / 5) * 100
    radar_df['hedef_kellik'] = radar_df['hedef_kellik'].map({0: 'Kel Değil', 1: 'Kel/Dökülüyor'})
    radar_melted = radar_df.melt(id_vars=['hedef_kellik'], var_name='Özellik', value_name='Skor')
    fig_radar = px.line_polar(radar_melted, r='Skor', theta='Özellik', color='hedef_kellik', line_close=True,
                              template="plotly_dark", color_discrete_sequence=["#2ecc71", "#e74c3c"])
    fig_radar.update_traces(fill='toself')
    st.plotly_chart(fig_radar, use_container_width=True)

    # OKUMA REHBERİ
    st.info(
        "**Nasıl Okunur?** Kırmızı alan (Riskli Grup) hangi yöne doğru daha fazla dışarı taşmışsa, o özellik o grupta daha yüksektir. Kırmızı alanın yaş ve stres yönüne doğru genişlemesi, yıpranmanın haritasını çizer.")

st.divider()

st.subheader("🔥 3. Değişkenlerin Etkileşimi (Korelasyon Haritası)")
st.markdown("*Hangi durumlar birbiriyle matematiksel olarak bağlantılı?*")

corr_df = df_large[['yas', 'calisma_saati', 'stres_seviyesi', 'genetik_kellik', 'hedef_kellik']].corr()
corr_df.columns = ['Yaş', 'Saat', 'Stres', 'Genetik', 'Kellik Durumu']
corr_df.index = ['Yaş', 'Saat', 'Stres', 'Genetik', 'Kellik Durumu']
fig_corr = px.imshow(corr_df, text_auto=".2f", color_continuous_scale='RdBu_r', aspect="auto")
st.plotly_chart(fig_corr, use_container_width=True)

# OKUMA REHBERİ
st.info(
    "**Nasıl Okunur?** Kırmızı renge (1.0'a) yaklaşan kutular, iki durumun birbirini tetiklediğini (doğru orantı) gösterir. Örneğin; Yaş ile Kellik Durumu arasındaki kutu kırmızıysa, yaş arttıkça riskin de arttığını anlarız. Mavi kutular ise aralarında zıt bir ilişki olduğunu belirtir.")

st.divider()

st.subheader("⚠️ 4. Risk Kümelenmesi: Yaş ve Stres Kesitleri")
st.markdown("*Tehlike haritanın neresinde yoğunlaşıyor?*")

scatter_df = df_large.copy()
scatter_df['Saç Durumu'] = scatter_df['hedef_kellik'].map({0: 'Sağlıklı', 1: 'Kel/Dökülüyor'})
fig_scatter = px.scatter(scatter_df, x="yas", y="stres_seviyesi", color="Saç Durumu", size="calisma_saati",
                         hover_data=["meslek"], color_discrete_sequence=["#2ecc71", "#e74c3c"])
fig_scatter.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
st.plotly_chart(fig_scatter, use_container_width=True)

# OKUMA REHBERİ
st.info(
    "**Nasıl Okunur?** Noktaların Yeri: Kişinin yaşı ve stres seviyesini gösterir. Noktanın Rengi: Kırmızı olanlar saç dökülmesi yaşayanlardır. Noktanın Büyüklüğü: O kişinin günde kaç saat çalıştığıdır. Kırmızı noktaların grafiğin neresinde (hangi yaş ve streste) yığıldığına bakarak riskli bölgeyi kendi gözlerinizle keşfedebilirsiniz.")

st.divider()

st.subheader("☀️ 5. Kaderin Akışı: Genetikten Kelliğe Yolculuk")
st.markdown("*Genetik ve stres birleşince sonuç ne oluyor?*")

sunburst_df = df_large.copy()
sunburst_df['Genetik'] = sunburst_df['genetik_kellik'].map({1: 'Genetik Var', 0: 'Genetik Yok'})
sunburst_df['Stres Durumu'] = sunburst_df['stres_seviyesi'].apply(
    lambda x: 'Yüksek Stres (4-5)' if x >= 4 else 'Düşük Stres (1-3)')
sunburst_df['Kellik'] = sunburst_df['hedef_kellik'].map({1: 'Kel', 0: 'Saçlı'})
fig_sun = px.sunburst(sunburst_df, path=['Genetik', 'Stres Durumu', 'Kellik'], color='Kellik',
                      color_discrete_map={'Kel': '#e74c3c', 'Saçlı': '#2ecc71', '(?!)': '#34495e'})
st.plotly_chart(fig_sun, use_container_width=True)

# OKUMA REHBERİ
st.info(
    "**Nasıl Okunur?** Grafiği tam merkezden dışarıya doğru adım adım okumalısınız. Örneğin önce içteki 'Genetik Var' dilimine tıklayın/bakın. Ardından onun dışındaki 'Yüksek Stres' dilimine geçin ve en dış katmanda bu yolculuğun yüzde kaç oranında 'Kel' olarak sonuçlandığını görün. Dilimlerin genişliği, o gruptaki insan sayısını yansıtır.")
