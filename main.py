from src.data_loader import load_and_preprocess_data
from src.statistical_tests import run_all_tests
from src.model_pipeline import train_baldness_model
from src.report_generator import create_pdf_report

DATA_PATH = "data/Tuğçe proje ödevi.xlsx"


def main():
    # 1. Aşama: Veriyi Çek ve Temizle
    df = load_and_preprocess_data(DATA_PATH)

    # 2. Aşama: İstatistiksel Hipotez Testleri
    run_all_tests(df)

    # 3. Aşama: Makine Öğrenmesi
    accuracy, importances = train_baldness_model(df)

    # 4. Aşama: Kurumsal Rapor Üretimi
    create_pdf_report(accuracy, importances)


if __name__ == "__main__":
    main()
