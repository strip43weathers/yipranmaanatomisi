import pandas as pd
import numpy as np


def load_and_preprocess_data(filepath):
    print("Veri yükleniyor ve kodlama (encoding) aranıyor...")

    # Olası tüm kodlamaları sırayla deneyecek liste
    encodings_to_try = ['utf-8-sig', 'mac_turkish', 'iso-8859-9', 'cp1254', 'utf-8', 'latin1']
    df = None

    for enc in encodings_to_try:
        try:
            df = pd.read_csv(filepath, index_col=0, encoding=enc)
            print(f"--> Başarılı! Dosya '{enc}' formatında çözüldü ve okundu.\n")
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(
            "Dosya bilinen hiçbir formatla okunamadı. Lütfen Excel'den tekrar CSV (UTF-8) olarak kaydedin.")

    # 1. Sütun isimlerini kodlaması kolay ve standart hale getirme
    df.columns = [
        'cinsiyet', 'kellik_durumu', 'yas', 'meslek', 'tecrube_yili',
        'calisma_saati', 'stres_seviyesi', 'isini_seviyor_mu',
        'genetik_kellik', 'sigara', 'alkol'
    ]

    # 2. Kellik Durumunu Sınıflandırma (Binary Encoding)
    df['kellik_durumu'] = df['kellik_durumu'].astype(str).str.strip()
    df['hedef_kellik'] = df['kellik_durumu'].apply(lambda x: 0 if x == 'Değil' else 1)

    # 3. Çalışma Saati Sütununu Temizleme (Özel Fonksiyon)
    def saat_duzelt(val):
        val = str(val).strip().lower()
        if val in ['nan', '', 'süre yok']:
            return np.nan
        if 'nöbet' in val:
            return 12.0
        if '-' in val:
            parts = val.split('-')
            return (float(parts[0]) + float(parts[1])) / 2

        val = val.replace(',', '.')
        try:
            return float(val)
        except:
            return np.nan

    df['calisma_saati'] = df['calisma_saati'].apply(saat_duzelt)
    df['calisma_saati'] = df['calisma_saati'].fillna(df['calisma_saati'].median())

    # 4. Tecrübe yılı ve Stres seviyesindeki boşlukları düzeltme
    df['tecrube_yili'] = pd.to_numeric(df['tecrube_yili'], errors='coerce').fillna(0)
    df['stres_seviyesi'] = pd.to_numeric(df['stres_seviyesi'], errors='coerce').fillna(df['stres_seviyesi'].median())

    # 5. Evet/Hayır ve Var/Yok şeklindeki kategorik verileri 1 ve 0'a çevirme
    binary_map = {'Evet': 1, 'Hayır': 0, 'Var': 1, 'Yok': 0}
    for col in ['isini_seviyor_mu', 'genetik_kellik', 'sigara', 'alkol']:
        df[col] = df[col].map(binary_map).fillna(0).astype(int)

    # Cinsiyeti sayısal yapma (Erkek: 1, Kadın: 0)
    df['cinsiyet_encoded'] = df['cinsiyet'].apply(lambda x: 1 if x == 'Erkek' else 0)

    print("Veri başarıyla temizlendi ve sayısallaştırıldı!\n")
    return df