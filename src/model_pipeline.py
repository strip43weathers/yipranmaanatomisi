import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def plot_learning_curve(model, X, y):
    print("Öğrenme Eğrisi (Doyum Noktası) Hesaplanıyor ve Çiziliyor...")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'o-', color="#2980b9", label="Eğitim Başarısı")
    plt.plot(train_sizes, test_mean, 'o-', color="#27ae60", label="Test (Doğrulama) Başarısı")
    plt.title("Modelin Öğrenme Eğrisi", fontsize=14)
    plt.xlabel("Kullanılan Veri Miktarı", fontsize=11)
    plt.ylabel("Doğruluk Oranı (Accuracy)", fontsize=11)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("data/ogrenme_egrisi.png")
    plt.close()
    print("-> Grafik 'data/ogrenme_egrisi.png' olarak kaydedildi.\n")


def train_baldness_model(df):
    print("\n" + "=" * 50)
    print("MAKİNE ÖĞRENMESİ: KELLİK TAHMİN MODELİ (MESLEKLİ)")
    print("=" * 50)

    # 1. Meslekleri Sayısallaştırma (One-Hot Encoding)
    df_encoded = pd.get_dummies(df, columns=['meslek'], prefix='meslek')

    # 2. Özellikleri Belirleme
    meslek_cols = [col for col in df_encoded.columns if col.startswith('meslek_')]
    base_features = ['yas', 'calisma_saati', 'stres_seviyesi', 'genetik_kellik', 'sigara', 'alkol', 'cinsiyet_encoded']
    all_features = base_features + meslek_cols

    X = df_encoded[all_features]
    y = df_encoded['hedef_kellik']

    # 3. Model Eğitimi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    # 4. Grafiği çizdir (İsteğe bağlı çalışır)
    plot_learning_curve(model, X, y)

    # 5. Doğruluk (Accuracy) ve Önemler
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Model Doğruluk Oranı: % {acc * 100:.2f}")

    importances = pd.DataFrame({
        'Özellik': all_features,
        'Önem (%)': model.feature_importances_ * 100
    }).sort_values(by='Önem (%)', ascending=False)

    return model, acc, importances, all_features, meslek_cols