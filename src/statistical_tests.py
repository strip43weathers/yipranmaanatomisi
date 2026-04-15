import pandas as pd
from scipy import stats


def check_stress_vs_job_satisfaction(df):
    # İşini sevenler ve sevmeyenleri iki ayrı gruba ayır
    group_loves_job = df[df['isini_seviyor_mu'] == 1]['stres_seviyesi']
    group_hates_job = df[df['isini_seviyor_mu'] == 0]['stres_seviyesi']

    # Mann-Whitney U Testi (Veri setimiz küçük olduğu ve normal dağılım garantisi olmadığı için)
    stat, p_value = stats.mannwhitneyu(group_loves_job, group_hates_job)

    print(f"P-Value: {p_value:.4f}")
    if p_value < 0.05:
        print("Sonuç: İşini sevmek ile stres seviyesi arasında istatistiksel olarak ANLAMLI bir fark var!")
    else:
        print("Sonuç: Bu iki grup arasında istatistiksel olarak anlamlı bir fark kanıtlanamadı.")
