import pyodbc
from scipy.stats import chisquare, kstest
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# MSSQL veritabanına bağlan
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=DESKTOP-EFBJ0MU;'  # Sunucu adını yaz
    'DATABASE=kadoDB;'  # Veritabanı adını yaz
    'UID=sa;'  # Kullanıcı adını yaz
    'PWD=Kadir81onur;'  # Şifreni yaz
)

# Veriyi çek
cursor = conn.cursor()
query = "SELECT Value FROM RedNumbers"  # Sütun ve tablo adını kontrol et
cursor.execute(query)
veriler = [row[0] for row in cursor.fetchall()]
conn.close()

# Verilerin özetini göster
print(f"Çekilen veri sayısı: {len(veriler)}")

# Chi-square testi
gruplar = np.histogram(veriler, bins=50)[0]  # 50 gruba ayır
istatistik, p_degeri = chisquare(gruplar)
print(f"Chi-square İstatistik: {istatistik}, P-Değeri: {p_degeri}")

if p_degeri < 0.05:
    print("Sonuç: Veriler rastgele olmayabilir!")
else:
    print("Sonuç: Veriler rastgele görünüyor.")

# Kolmogorov-Smirnov testi
kstest_sonuc = kstest(veriler, 'uniform')  # Uniform dağılıma uygunluğu kontrol et
print(f"K-S Test İstatistik: {kstest_sonuc.statistic}, P-Değeri: {kstest_sonuc.pvalue}")

# İlk 20 veriyi göster
print("İlk 20 sayı:", veriler[:20])

# Masaüstündeki "Grafikler" klasörüne kaydetme
grafik_kaydetme_yolu = os.path.join(os.path.expanduser("~"), "Desktop", "Grafikler")
os.makedirs(grafik_kaydetme_yolu, exist_ok=True)  # Klasör yoksa oluştur

# Histogram grafiği
plt.hist(veriler, bins=200, color='blue', alpha=0.7)
plt.title("Veri Dağılımı (Histogram)")
plt.xlabel("Değerler")
plt.ylabel("Frekans")
png_histogram = os.path.join(grafik_kaydetme_yolu, "veri_dagilimi_histogram.png")
plt.savefig(png_histogram)
plt.clf()  # Çizimi temizle
print(f"Histogram grafik kaydedildi: {png_histogram}")

# Çizgi grafiği
plt.plot(veriler[:1000], color='orange')  # İlk 1000 veriyi göster
plt.title("Veri Trend Çizgisi")
plt.xlabel("Index")
plt.ylabel("Değer")
png_trend = os.path.join(grafik_kaydetme_yolu, "veri_trend_cizgisi.png")
plt.savefig(png_trend)
plt.clf()
print(f"Çizgi grafik kaydedildi: {png_trend}")

# Yoğunluk grafiği (KDE)
sns.kdeplot(veriler, shade=True, color="blue")
plt.title("Veri Yoğunluğu (KDE)")
plt.xlabel("Değerler")
png_kde = os.path.join(grafik_kaydetme_yolu, "veri_yogunlugu_kde.png")
plt.savefig(png_kde)
plt.clf()
print(f"Yoğunluk grafiği kaydedildi: {png_kde}")