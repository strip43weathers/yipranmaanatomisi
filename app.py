import streamlit as st
import pandas as pd

# Kendi yazdığımız modülleri içe aktarıyoruz
from src.data_loader import load_and_preprocess_data
from src.synthetic_data import generate_synthetic_data
from src.model_pipeline import train_baldness_model

# Sayfa Sekmesi Ayarları
st.set_page_config(page_title="Saç Dökülme Analizi", page_icon="🧬", layout="centered")


# Yapay Zekayı Sadece 1 Kere Eğitmek İçin Önbelleğe (Cache) Alıyoruz
@st.cache_resource
def prepare_ai():
    DATA_PATH = "data/Tuğçe proje ödevi.xlsx"
    df = load_and_preprocess_data(DATA_PATH)
    df_large = generate_synthetic_data(df, target_size=1000)
    # Az önce değiştirdiğimiz yerden modeli alıyoruz
    model, acc, importances = train_baldness_model(df_large)
    return model, acc


# Modeli arka planda yükle
model, accuracy = prepare_ai()

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
    # predict_proba bize iki değer verir: [Kel Olmama Olasılığı, Kel Olma Olasılığı]
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

    st.info(
        "Not: Bu analiz medikal bir gerçeklik taşımaz. Sadece 104 kişilik gerçek hayat verisinden öğrenilen istatistiksel ve matematiksel örüntüleri yansıtır.")
