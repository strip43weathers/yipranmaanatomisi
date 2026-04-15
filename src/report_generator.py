import os
from xhtml2pdf import pisa
import matplotlib.pyplot as plt
import pandas as pd


def generate_feature_plot(feature_importance_df, output_path="data/grafik.png"):
    # Özellik önemlerini bar grafiğine dönüştür
    plt.figure(figsize=(8, 5))
    plt.barh(feature_importance_df['Özellik'], feature_importance_df['Önem (%)'], color='#2c3e50')
    plt.gca().invert_yaxis()
    plt.title('Kellik Tahmininde Etkili Olan Faktörler', fontsize=14)
    plt.xlabel('Etki Yüzdesi (%)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def create_pdf_report(accuracy, feature_importances, output_pdf="data/Yipranma_Anatomisi_Raporu.pdf"):
    print("\nKurumsal PDF Raporu Hazırlanıyor...")

    # 1. Grafiği Oluştur
    grafik_yolu = generate_feature_plot(feature_importances)
    grafik_mutlak_yol = os.path.abspath(grafik_yolu)  # xhtml2pdf tam yol ister

    # 2. Dinamik HTML Şablonu (Türkçe karakter destekli yapı)
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {{
                size: A4;
                margin: 2cm;
            }}
            body {{
                font-family: Helvetica, Arial, sans-serif; /* Türkçe font sorunu için standart sans-serif */
                color: #333;
                line-height: 1.6;
            }}
            .header {{
                text-align: center;
                border-bottom: 2px solid #2c3e50;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            h1 {{ color: #2c3e50; font-size: 24px; }}
            h2 {{ color: #e74c3c; font-size: 18px; border-bottom: 1px solid #ccc; padding-bottom: 5px;}}
            .box {{
                background-color: #f9f9f9;
                border-left: 4px solid #2980b9;
                padding: 15px;
                margin: 20px 0;
            }}
            .img-container {{ text-align: center; margin-top: 20px; }}
            img {{ width: 80%; border: 1px solid #ddd; padding: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>MODERN ÇALIŞMA HAYATININ YIPRANMA ANATOMİSİ</h1>
            <p><strong>Veri Analizi ve Makine Öğrenmesi Referans Kataloğu</strong></p>
        </div>

        <h2>1. Yönetici Özeti</h2>
        <p>104 kisilik veri seti uzerinde gerceklestirilen analizler ve egitilen Yapay Zeka modeli sonucunda, stres ve calisma saatlerinin fiziksel yipranma (kellik) uzerindeki etkileri bilimsel olarak olculmustur.</p>

        <div class="box">
            <strong>Yapay Zeka (Random Forest) Dogruluk Orani:</strong> % {accuracy * 100:.2f}
        </div>

        <h2>2. Etki Analizi (Feature Importances)</h2>
        <p>Modelin kellik tahminlemesi yaparken dikkate aldigi matematiksel agirliklar asagidaki grafikte sunulmustur:</p>

        <div class="img-container">
            <img src="{grafik_mutlak_yol}" />
        </div>

        <h2>3. Istatistiksel Cikarimlar</h2>
        <ul>
            <li><strong>Genetik ve Yas:</strong> Sac dokulmesinde en buyuk etken (%56 agirlik) biyolojik faktorler olmustur.</li>
            <li><strong>Stres ve Is Hayati:</strong> Calisma saati ve is stresi, sanilanin aksine dogrudan kellik garantisi vermemektedir, ancak %30 luk bir paya sahiptirler.</li>
            <li><strong>Kotu Aliskanliklar:</strong> Sigara ve alkol kullaniminin model uzerindeki etkisi matematiksel olarak ihmal edilebilir duzeyde cikmistir.</li>
        </ul>

        <p style="text-align:center; margin-top:50px; font-size:12px; color:#777;">
            Bu rapor Python ve Makine Ogrenmesi algoritmalari tarafindan otomatik uretilmistir.
        </p>
    </body>
    </html>
    """

    # 3. HTML'i PDF'e Çevirme
    with open(output_pdf, "w+b") as result_file:
        pisa_status = pisa.CreatePDF(
            src=html_content,
            dest=result_file,
            encoding='UTF-8'
        )

    if pisa_status.err:
        print(f"PDF Olusturulurken Hata: {pisa_status.err}")
    else:
        print(f"-> MUKEMMEL! Rapor basariyla uretildi: {output_pdf}")
