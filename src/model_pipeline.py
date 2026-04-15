import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def plot_learning_curve(model, X, y):
    print("Öğrenme Eğrisi (Doyum Noktası) Hesaplanıyor ve Çiziliyor...")

    # Modeli %10'dan %100'e kadar farklı veri büyüklüklerinde test et
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )

    # 5 katlı (cv=5) çapraz doğrulamanın ortalamalarını al
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Grafiği çiz
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'o-', color="#2980b9", label="Eğitim Başarısı")
    plt.plot(train_sizes, test_mean, 'o-', color="#27ae60", label="Test (Doğrulama) Başarısı")

    plt.title("Modelin Öğrenme Eğrisi (Kapasite Analizi)", fontsize=14)
    plt.xlabel("Kullanılan Veri Miktarı (Satır Sayısı)", fontsize=11)
    plt.ylabel("Doğruluk Oranı (Accuracy)", fontsize=11)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)

    # Kaydet ve kapat
    plt.tight_layout()
    plt.savefig("data/ogrenme_egrisi.png")
    plt.close()
    print("-> Harika! Grafik 'data/ogrenme_egrisi.png' olarak kaydedildi.\n")


def train_baldness_model(df):
    print("\n" + "=" * 50)
    print("MAKİNE ÖĞRENMESİ: KELLİK TAHMİN MODELİ (RANDOM FOREST)")
    print("=" * 50)

    # 1. Özellikler ve Hedef
    features = ['yas', 'calisma_saati', 'stres_seviyesi', 'genetik_kellik',
                'sigara', 'alkol', 'cinsiyet_encoded']
    X = df[features]
    y = df['hedef_kellik']

    # 2. Veriyi Eğitim ve Test Olarak Ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Modeli Tanımlama ve Eğitme
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    # 4. ÖĞRENME EĞRİSİNİ ÇİZ (Yeni Eklenen Adım)
    plot_learning_curve(model, X, y)

    # 5. Test Verisi Üzerinde Tahmin Yapma ve Başarı Ölçme
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    print(f"Model Doğruluk Oranı (Accuracy): % {acc * 100:.2f}")

    # 6. Özellik Önemleri
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Özellik': features,
        'Önem (%)': importances * 100
    }).sort_values(by='Önem (%)', ascending=False)

    print("\n" + "=" * 50 + "\n")
    return model, acc, feature_importance_df
