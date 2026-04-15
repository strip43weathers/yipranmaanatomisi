import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_baldness_model(df):
    print("\n" + "=" * 50)
    print("MAKİNE ÖĞRENMESİ: KELLİK TAHMİN MODELİ (RANDOM FOREST)")
    print("=" * 50)

    # 1. Özellikler (Features - X) ve Hedef (Target - y) Seçimi
    # Modelin öğrenmesi için kullanacağı sütunları seçiyoruz.
    features = ['yas', 'calisma_saati', 'stres_seviyesi', 'genetik_kellik',
                'sigara', 'alkol', 'cinsiyet_encoded']

    X = df[features]
    y = df['hedef_kellik']

    # 2. Veriyi Eğitim (%80) ve Test (%20) Olarak Ayırma
    # random_state=42 ile her çalıştırmada aynı rastgeleliği elde ederiz.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Modeli Tanımlama ve Eğitme
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    # 4. Test Verisi Üzerinde Tahmin Yapma
    predictions = model.predict(X_test)

    # 5. Modelin Başarısını Ölçme
    acc = accuracy_score(y_test, predictions)
    print(f"\nModel Doğruluk Oranı (Accuracy): % {acc * 100:.2f}")

    print("\nDetaylı Sınıflandırma Raporu:")
    print(classification_report(y_test, predictions, target_names=['Kel Değil (0)', 'Kel/Dökülüyor (1)']))

    # 6. Özelliklerin Önem Derecesini (Feature Importance) Çıkarma
    print("\nModel Karar Verirken En Çok Neye Baktı? (Özellik Önemleri):")
    importances = model.feature_importances_

    # Özellikleri önem sırasına göre dizelim
    feature_importance_df = pd.DataFrame({
        'Özellik': features,
        'Önem (%)': importances * 100
    }).sort_values(by='Önem (%)', ascending=False)

    print(feature_importance_df.to_string(index=False))

    print("\n" + "=" * 50 + "\n")
    return acc, feature_importance_df
