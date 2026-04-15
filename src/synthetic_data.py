import pandas as pd
import numpy as np


def generate_synthetic_data(df, target_size=1000):
    print("\n" + "=" * 50)
    print("AKILLI SENTETİK VERİ ÜRETİMİ (BOOTSTRAP & NOISE)")
    print("=" * 50)

    current_size = len(df)
    needed_size = target_size - current_size

    if needed_size <= 0:
        return df

    # 1. KLONLAMA (Bootstrap Resampling)
    # Gerçek 104 kişinin içinden rastgele (kopya çekerek) 896 kere seçim yapıyoruz.
    # Böylece Genetik-Kellik, İş stresi-Çalışma saati gibi ilişkiler %100 korunuyor.
    synthetic_df = df.sample(n=needed_size, replace=True).reset_index(drop=True)

    # 2. MUTASYON (Gaussian Noise)
    # Klonlanan bu sanal insanlar orijinaliyle birebir aynı olmasın (model ezberlemesin)
    # diye sadece sayısal verilerine çok hafif mantıklı değişimler ekliyoruz.

    # Yaşlarına -2 ile +2 arasında rastgele bir yaş ekleyelim/çıkaralım
    synthetic_df['yas'] += np.random.randint(-2, 3, size=needed_size)
    synthetic_df['yas'] = np.clip(synthetic_df['yas'], 18, 80)  # Yaş sınırları

    # Çalışma saatlerine -1 ile +1 saat arasında sapma ekleyelim
    synthetic_df['calisma_saati'] += np.random.normal(0, 1, size=needed_size)
    synthetic_df['calisma_saati'] = np.clip(np.round(synthetic_df['calisma_saati'], 1), 1, 24)

    # DİKKAT: Cinsiyet, Genetik ve Kellik durumu gibi "kategorik" verilere DOKUNMUYORUZ.
    # Çünkü dokunursak aralarındaki bilimsel matematiği bozarız.

    # İşimize yaramayan metin (string) sütunlarını modelde kullanmayacağımız için sentetik veride boş bırakabiliriz
    for col in df.columns:
        if col not in synthetic_df.columns:
            synthetic_df[col] = 'Sanal Klon'

    # Orijinal 104 kişi ile Sanal 896 kişiyi alt alta birleştir
    df_combined = pd.concat([df, synthetic_df], ignore_index=True)

    print(f"-> {current_size} gerçek veriden klonlanarak {needed_size} adet AKILLI SANAL İNSAN eklendi!")
    print(f"-> Yeni Veri Seti Boyutu: {len(df_combined)} satır.")
    print("=" * 50 + "\n")

    return df_combined
