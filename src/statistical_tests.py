import pandas as pd
from scipy import stats


def run_all_tests(df):
    print("\n" + "=" * 50)
    print("İSTATİSTİKSEL HİPOTEZ TESTLERİ BAŞLIYOR")
    print("=" * 50)

    # 1. İşini Sevmek ile Stres Arasındaki İlişki (Mann-Whitney U Testi)
    # Stres seviyesi normal dağılmadığı (1-5 arası ordinal) için non-parametrik test kullanıyoruz.
    print("\n1. HİPOTEZ: İşini sevenlerin stres seviyesi daha mı düşüktür?")
    group_loves = df[df['isini_seviyor_mu'] == 1]['stres_seviyesi']
    group_hates = df[df['isini_seviyor_mu'] == 0]['stres_seviyesi']

    stat, p_value = stats.mannwhitneyu(group_loves, group_hates, alternative='two-sided')
    print(f"P-Değeri (p-value): {p_value:.4f}")
    if p_value < 0.05:
        print(
            "-> Sonuç: İstatistiksel olarak ANLAMLI! İşini sevme durumu ile stres seviyesi arasında kanıtlanmış bir bağ var.")
    else:
        print("-> Sonuç: Anlamlı bir fark bulunamadı. (Belki de herkes streslidir!)")

    # 2. Genetik Kellik ile Gerçek Kellik Arasındaki Bağıntı (Ki-Kare Testi)
    # İki kategorik (Evet/Hayır) değişken arasındaki ilişkiyi ölçer.
    print("\n2. HİPOTEZ: Genetik yatkınlık, gerçek kelliği doğrudan etkiler mi?")
    contingency_table = pd.crosstab(df['genetik_kellik'], df['hedef_kellik'])
    chi2, p_val_chi, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"P-Değeri (p-value): {p_val_chi:.4f}")
    if p_val_chi < 0.05:
        print("-> Sonuç: İstatistiksel olarak ANLAMLI! Genetik ile kellik arasında kesin bir ilişki var.")
    else:
        print("-> Sonuç: Anlamlı bir bağ kurulamadı (Stres ve çalışma hayatı genetiği yenmiş olabilir mi?)")

    # 3. Çalışma Saati ile Stres Arasında Korelasyon (Spearman Korelasyonu)
    print("\n3. HİPOTEZ: Çalışma saati arttıkça stres seviyesi artar mı?")
    corr, p_val_corr = stats.spearmanr(df['calisma_saati'], df['stres_seviyesi'])
    print(f"Korelasyon Katsayısı (r): {corr:.4f} | P-Değeri: {p_val_corr:.4f}")
    if p_val_corr < 0.05:
        yon = "pozitif (saat arttıkça stres artar)" if corr > 0 else "negatif (saat arttıkça stres düşer)"
        print(f"-> Sonuç: İstatistiksel olarak ANLAMLI! Aralarında {yon} bir ilişki var.")
    else:
        print("-> Sonuç: Anlamlı bir korelasyon bulunamadı.")

    print("\n" + "=" * 50 + "\n")
