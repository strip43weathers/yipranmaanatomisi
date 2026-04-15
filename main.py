from src.data_loader import load_and_preprocess_data
from src.statistical_tests import run_all_tests
from src.synthetic_data import generate_synthetic_data  # YENİ EKLENDİ
from src.model_pipeline import train_baldness_model
from src.report_generator import create_pdf_report

DATA_PATH = "data/Tuğçe proje ödevi.xlsx"


def main():
    # 1. Aşama: Orijinal Veriyi Çek ve Temizle (104 Kişi)
    df_original = load_and_preprocess_data(DATA_PATH)

    # 2. Aşama: Orijinal Veri Üzerinden Bilimsel Testler
    # (İstatistiksel gerçekliği bozmamak için testleri orijinal 104 kişiye yapıyoruz)
    run_all_tests(df_original)

    # 3. Aşama: SENTETİK VERİ ÜRETİMİ (1000 Kişiye Çıkarma)
    df_large = generate_synthetic_data(df_original, target_size=1000)

    # 4. Aşama: Makine Öğrenmesi (Modeli artık 1000 kişiyle eğitiyoruz!)
    accuracy, importances = train_baldness_model(df_large)

    # 5. Aşama: Kurumsal Rapor Üretimi
    create_pdf_report(accuracy, importances)


if __name__ == "__main__":
    main()
