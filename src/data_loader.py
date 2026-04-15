import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess_data(filepath):
    # Veriyi oku
    df = pd.read_excel(filepath)

    # Sütun isimlerini kodlaması kolay olsun diye İngilizce ve küçük harfe çevirebilirsin
    # Örn: df.columns = ['gender', 'baldness', 'age', 'profession', ...]

    # 1. Eksik Veri Kontrolü
    if df.isnull().sum().any():
        print("Uyarı: Eksik veriler var, dolduruluyor...")
        df.fillna(df.median(numeric_only=True), inplace=True)

    # 2. Kategorik Değişkenleri Sayısala Çevirme (Örn: Cinsiyet, Meslek)
    le = LabelEncoder()
    df['cinsiyet_encoded'] = le.fit_transform(df['cinsiyet'])
    # Meslekler için One-Hot Encoding daha iyi olabilir: pd.get_dummies(df, columns=['meslek'])

    # 3. Ölçeklendirme (Standartlaştırma)
    # Yaş ile kellik oranının matematiksel ağırlıkları farklıdır, bunları aynı düzleme çekmeliyiz.
    scaler = StandardScaler()
    df[['yas', 'calisma_saati', 'stres_seviyesi']] = scaler.fit_transform(
        df[['yas', 'calisma_saati', 'stres_seviyesi']])

    return df


if __name__ == "__main__":
    df = load_and_preprocess_data("../data/veriseti.xlsx")
    print(df.head())
