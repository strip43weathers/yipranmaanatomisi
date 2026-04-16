from src.data_loader import load_and_preprocess_data
from src.statistical_tests import run_all_tests
from src.synthetic_data import generate_synthetic_data
from src.model_pipeline import train_baldness_model

DATA_PATH = "data/Tuğçe proje ödevi.xlsx"


def main():
    # 1. Aşama: Orijinal Veriyi Çek ve Temizle (104 Kişi)
    df_original = load_and_preprocess_data(DATA_PATH)

    # 2. Aşama: Orijinal Veri Üzerinden Bilimsel Testler
    run_all_tests(df_original)

    # 3. Aşama: SENTETİK VERİ ÜRETİMİ (1000 Kişiye Çıkarma)
    df_large = generate_synthetic_data(df_original, target_size=1000)

    # 4. Aşama: Makine Öğrenmesi
    accuracy, importances = train_baldness_model(df_large)

    print("\n✅ Pipeline başarıyla çalıştı! Modeller ve testler tamamlandı.")


if __name__ == "__main__":
    main()
