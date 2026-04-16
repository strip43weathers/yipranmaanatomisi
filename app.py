import streamlit as st
import pandas as pd
import plotly.express as px  # YENİ EKLENDİ

# Kendi yazdığımız modülleri içe aktarıyoruz
from src.data_loader import load_and_preprocess_data
from src.synthetic_data import generate_synthetic_data
from src.model_pipeline import train_baldness_model

# Sayfa Sekmesi Ayarları
st.set_page_config(page_title="Saç Dökülme Analizi", page_icon="🧬", layout="centered")


@st.cache_resource
def prepare_ai():
    DATA_PATH = "data/Tuğçe proje ödevi.xlsx"
    df = load_and_preprocess_data(DATA_PATH)
    df_large = generate_synthetic_data(df, target_size=1000)
    # model_pipeline'dan 3 değer dönüyor: model, accuracy ve importances
    model, acc, importances = train_baldness_model(df_large)
    return model, acc, importances, df_large

# Değişkenleri alırken importances'ı da ekle
model, accuracy, importances, df_large = prepare_ai()

# ----- WEB SİTESİ ARAYÜZÜ (UI) -----

st.title("🧬 Modern Çalışma Hayatı: Saç Dökülme Tahminleyicisi")
st.markdown(f"**Yapay Zeka Doğruluk Oranı:** %{accuracy * 100:.1f} *(1000 Kişilik Sentetik Veri ile Eğitildi)*")
st.write("Aşağıdaki değerleri değiştirerek yapay zekanın sizin için ne tahmin edeceğini anlık olarak görün.")

st.sidebar.header("Kişisel Bilgilerinizi Girin")

# Kullanıcıdan Veri Alacağımız Kaydırıcılar (Sliders) ve Butonlar
yas = st.sidebar.slider("Yaşınız", 18, 80, 30)
calisma_saati = st.sidebar.slider("Günlük Çalışma Saatiniz", 1.0, 24.0, 8.0, step=0.5)
stres_seviyesi = st.sidebar.slider("İşteki Stres Seviyeniz (1 Düşük - 5 Yüksek)", 1, 5, 3)

genetik = st.sidebar.radio("Ailenizde (özellikle anne/baba tarafı) kellik var mı?", ["Hayır", "Evet"])
sigara = st.sidebar.radio("Sigara kullanıyor musunuz?", ["Hayır", "Evet"])
alkol = st.sidebar.radio("Alkol kullanıyor musunuz?", ["Hayır", "Evet"])
cinsiyet = st.sidebar.radio("Cinsiyetiniz", ["Kadın", "Erkek"])

# Kullanıcının Girdiği Verileri Modelin Anlayacağı Matrise (DataFrame) Çeviriyoruz
user_data = pd.DataFrame({
    'yas': [yas],
    'calisma_saati': [calisma_saati],
    'stres_seviyesi': [stres_seviyesi],
    'genetik_kellik': [1 if genetik == "Evet" else 0],
    'sigara': [1 if sigara == "Evet" else 0],
    'alkol': [1 if alkol == "Evet" else 0],
    'cinsiyet_encoded': [1 if cinsiyet == "Erkek" else 0]
})

st.divider()

# Hesapla Butonu
if st.button("Saç Dökülme İhtimalimi Hesapla 🔍", use_container_width=True):

    # Modelden Olasılık (Probability) Tahmini Alıyoruz
    prediction_prob = model.predict_proba(user_data)[0][1]

    st.subheader("Yapay Zeka Analiz Sonucu:")

    # Sonuca göre renkli ve dinamik mesajlar
    if prediction_prob > 0.50:
        st.error(
            f"⚠️ **Dikkat!** Mevcut yaşam tarzınız ve genetiğinize göre kellik / saç dökülmesi riskiniz: **%{prediction_prob * 100:.1f}**")
        st.write(
            "Yapay zekaya göre saç folikülleriniz baskı altında. Stresinizi azaltmayı veya çalışma saatlerinizi esnetmeyi deneyerek soldaki çubuklardan oranı nasıl düşürebileceğinizi test edebilirsiniz.")
    else:
        st.success(
            f"✅ **Güvendesiniz!** Kellik / saç dökülmesi riskiniz oldukça düşük: **%{prediction_prob * 100:.1f}**")
        st.write(
            "Genetiğiniz ve çalışma hayatınızdaki dengeniz şu an için saçlarınızı koruyor gibi görünüyor. Böyle devam edin!")

# GRAFİK 1: BAR

st.divider()
st.subheader("📊 1. Modelin Karar Mekanizması (Özellik Önemleri)")
st.write("Yapay zeka tahmin yaparken hangi özelliğe yüzde kaç ağırlık veriyor?")


chart_data = importances.set_index('Özellik')


st.bar_chart(chart_data)

st.info("Bu grafik, modelin 'Öğrenme' aşamasında genetik, yaş ve stres gibi faktörler arasında kurduğu matematiksel hiyerarşiyi gösterir.")

# GRAFİK 2: RADAR

st.divider()
st.subheader("🕸️ 2. Yaşam Tarzı DNA'sı: Kel vs. Sırma Saçlı Profil Dağılımı")
st.write("Veri setimizdeki insanların özelliklerine göre ortaya çıkan iki farklı yaşam tarzının radar haritası:")

# İki grubun (Kel Olanlar vs Kel Olmayanlar) yaş, saat ve stres ortalamalarını alıyoruz
radar_df = df_large.groupby('hedef_kellik')[['yas', 'calisma_saati', 'stres_seviyesi']].mean().reset_index()

# Verilerin hepsi radar ağında rahat görünsün diye 100 üzerinden normalize ediyoruz (Yüzdelik Skala)
# Çünkü Yaş (örn: 45) ile Stres (örn: 3) aynı grafikte yan yana olursa stres çizgisi görünmez!
radar_df['yas'] = (radar_df['yas'] / 80) * 100
radar_df['calisma_saati'] = (radar_df['calisma_saati'] / 24) * 100
radar_df['stres_seviyesi'] = (radar_df['stres_seviyesi'] / 5) * 100

radar_df['hedef_kellik'] = radar_df['hedef_kellik'].map({0: 'Kel Değil', 1: 'Kel/Dökülüyor'})

# Veriyi Plotly'nin sevdiği "uzun" (melt) formata çevirme
radar_melted = radar_df.melt(id_vars=['hedef_kellik'], var_name='Özellik', value_name='Skor (100 üzerinden)')

# Türkçeleştirme
radar_melted['Özellik'] = radar_melted['Özellik'].map({
    'yas': 'Yaş İlerlemesi',
    'calisma_saati': 'Çalışma Saati Yoğunluğu',
    'stres_seviyesi': 'Stres Yükü'
})

# Grafiği Çiz (Dark Tema çok şık durur)
fig = px.line_polar(
    radar_melted,
    r='Skor (100 üzerinden)',
    theta='Özellik',
    color='hedef_kellik',
    line_close=True,
    template="plotly_dark",
    color_discrete_sequence=["#2ecc71", "#e74c3c"]  # Kel değil yeşil, Kel olan kırmızı
)

# Grafikteki iç rengi hafif transparan dolduruyoruz
fig.update_traces(fill='toself')

st.plotly_chart(fig, use_container_width=True)

st.info(
    "Not: Yukarıdaki radar grafiğindeki eksenler, özellikler arası daha iyi bir görsel kıyaslama yapılabilmesi için maksimum değerlerine (Yaş: 80, Saat: 24, Stres: 5) oranlanarak 100 üzerinden puanlanmıştır.")

# ==========================================
# GRAFİK 3: KORELASYON ISI HARİTASI
# ==========================================
st.divider()
st.subheader("🔥 3. Değişkenlerin Etkileşimi (Korelasyon Haritası)")
st.write("Hangi özelliklerin birbiriyle matematiksel olarak ne kadar bağlantılı olduğunu (Pearson Korelasyonu) gösterir. 1'e veya -1'e yaklaşan değerler güçlü ilişkiyi simgeler.")

# Sadece sayısal sütunları seçip korelasyon matrisini çıkarıyoruz
corr_df = df_large[['yas', 'calisma_saati', 'stres_seviyesi', 'genetik_kellik', 'hedef_kellik']].corr()

# İsimleri Türkçeleştirelim ki grafikte şık dursun
corr_df.columns = ['Yaş', 'Çalışma Saati', 'Stres', 'Genetik', 'Kellik Durumu']
corr_df.index = ['Yaş', 'Çalışma Saati', 'Stres', 'Genetik', 'Kellik Durumu']

fig_corr = px.imshow(corr_df, text_auto=".2f", aspect="auto",
                     color_continuous_scale='RdBu_r',
                     title="Veri Setindeki Gizli Matematiksel İlişkiler")
st.plotly_chart(fig_corr, use_container_width=True)

# ==========================================
# GRAFİK 4: ÇOK BOYUTLU BALONCUK GRAFİĞİ (SCATTER BUBBLE)
# ==========================================
st.divider()
st.subheader("⚠️ 4. Risk Kümelenmesi: Yaş ve Stres Kesitleri")
st.write("Noktaların **rengi** saç durumunu, **büyüklüğü** ise günlük çalışma saatini temsil etmektedir.")

# Radar grafiğinde çevirdiğimiz "Kel/Değil" verisini burada tekrar formatlayalım
scatter_df = df_large.copy()
scatter_df['Saç Durumu'] = scatter_df['hedef_kellik'].map({0: 'Sağlıklı', 1: 'Kel/Dökülüyor'})

fig_scatter = px.scatter(scatter_df, x="yas", y="stres_seviyesi", color="Saç Durumu",
                         size="calisma_saati", hover_data=["meslek"],
                         color_discrete_sequence=["#2ecc71", "#e74c3c"],
                         labels={"yas": "Kişinin Yaşı", "stres_seviyesi": "İşteki Stres Seviyesi"},
                         title="Hangi Yaş ve Stres Kombinasyonu Daha Tehlikeli?")

# Kenarlara yoğunluk kutuları ekleyerek bilimsel bir hava katalım
fig_scatter.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# GRAFİK 5: GÜNEŞ IŞINI (SUNBURST) HİYERARŞİSİ
# ==========================================
st.divider()
st.subheader("☀️ 5. Kaderin Akışı: Genetikten Kelliğe Yolculuk")
st.write("İçeriden dışarıya doğru okuyun: Genetik yatkınlık ve Stres birleştiğinde kellikle sonuçlanma oranları.")

sunburst_df = df_large.copy()
# İsimleri grafiğin içine yazılacak şekilde formatlıyoruz
sunburst_df['Genetik'] = sunburst_df['genetik_kellik'].map({1: 'Genetik Var', 0: 'Genetik Yok'})
sunburst_df['Stres Durumu'] = sunburst_df['stres_seviyesi'].apply(lambda x: 'Yüksek Stres (4-5)' if x >=4 else 'Düşük Stres (1-3)')
sunburst_df['Kellik'] = sunburst_df['hedef_kellik'].map({1: 'Kel', 0: 'Saçlı'})

fig_sun = px.sunburst(sunburst_df, path=['Genetik', 'Stres Durumu', 'Kellik'],
                      color='Kellik',
                      color_discrete_map={'Kel':'#e74c3c', 'Saçlı':'#2ecc71', '(?!)':'#34495e'},
                      title="Neden-Sonuç İlişkisi Dağılımı")

st.plotly_chart(fig_sun, use_container_width=True)
