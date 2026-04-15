from src.data_loader import load_and_preprocess_data

# Dosya adını senin gönderdiğin CSV adıyla eşleştirdim.
# Eğer data klasörü içindeki adını değiştirdiysen burayı güncelle.
DATA_PATH = "data/Tuğçe proje ödevi.xlsx"


def main():
    # 1. Aşama: Veriyi Çek ve Temizle
    df = load_and_preprocess_data(DATA_PATH)

    # Temizlenmiş veriye genel bir bakış atalım
    print("--- TEMİZLENMİŞ VERİ SETİ (İlk 5 Satır) ---")
    print(df[['yas', 'meslek', 'calisma_saati', 'stres_seviyesi', 'hedef_kellik']].head())

    print("\n--- VERİ BİLGİSİ ---")
    print(df.info())


if __name__ == "__main__":
    main()
