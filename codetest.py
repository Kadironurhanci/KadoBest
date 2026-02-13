import os
import pyodbc
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, Dense
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer
from datetime import datetime
import torch

# TensorFlow optimizasyon uyarısını kapat
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# MSSQL veritabanına bağlan ve veriyi çek
conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=SunucuAdresi;'  # Sunucu adını buraya yaz
    'DATABASE=VeritabaniAdi;'  # Veritabanı adını buraya yaz
    'UID=KullaniciAdi;'  # Kullanıcı adını buraya yaz
    'PWD=Sifre;'  # Şifreni buraya yaz
)
cursor = conn.cursor()
query = "SELECT Timestamp, random_sayi FROM prng_verisi"
cursor.execute(query)
data = cursor.fetchall()
conn.close()

# Veriyi kontrol et
if not data:
    raise ValueError("Tablo boş veya sorgu sonucu boş. SQL sorgusunu ve tabloyu kontrol et!")

# Veriyi işle
timestamplar = [row[0] for row in data]
veriler = [row[1] for row in data]

# Veriyi normalleştir
min_deger, max_deger = min(veriler), max(veriler)
egitim_verisi = (np.array(veriler[:2000]).reshape(-1, 1) - min_deger) / (max_deger - min_deger)
hedef_veriler = (np.array(veriler[1:2001]) - min_deger) / (max_deger - min_deger)

# GPU kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

# 1. Random Forest Regressor (RFR)
rfr_model = RandomForestRegressor(n_estimators=100, random_state=42)
rfr_model.fit(egitim_verisi, hedef_veriler)

# 2. RNN Modeli
rnn_model = Sequential([
    Input(shape=(1, 1)),  # input_shape yerine Input kullanımı
    SimpleRNN(50, activation='tanh'),
    Dense(1)
])
rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.fit(egitim_verisi, hedef_veriler, epochs=10, batch_size=32)

# 3. LSTM Modeli
lstm_model = Sequential([
    Input(shape=(1, 1)),  # input_shape yerine Input kullanımı
    LSTM(50, activation='tanh'),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(egitim_verisi, hedef_veriler, epochs=10, batch_size=32)


# PyQt5 Arayüzü
class TahminUygulamasi(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Yapay Zeka Tahmin Süreci")
        self.setGeometry(100, 100, 400, 300)

        # Arayüz elemanları
        self.layout = QVBoxLayout()

        self.label_tahmin_rfr = QLabel("Tahmin (RFR): ...")
        self.label_tahmin_rnn = QLabel("Tahmin (RNN): ...")
        self.label_tahmin_lstm = QLabel("Tahmin (LSTM): ...")
        self.label_nihai_tahmin = QLabel("Nihai Tahmin: ...")
        self.label_gercek = QLabel("Gerçek Sayı: ...")
        self.label_hata = QLabel("Hata Oranı: ...")

        self.layout.addWidget(self.label_tahmin_rfr)
        self.layout.addWidget(self.label_tahmin_rnn)
        self.layout.addWidget(self.label_tahmin_lstm)
        self.layout.addWidget(self.label_nihai_tahmin)
        self.layout.addWidget(self.label_gercek)
        self.layout.addWidget(self.label_hata)
        self.setLayout(self.layout)

        # Zamanlayıcı
        self.timer = QTimer()
        self.timer.timeout.connect(self.tahmin_yap)  # Her 3 saniyede bir tahmin yap
        self.timer.start(3000)

        # Tahmin başlangıç indeksi
        self.index = 2001

    def tahmin_yap(self):
        if self.index >= len(veriler):
            self.timer.stop()  # Tüm veriler işlendiğinde durdur
            return

        # Tahmin yap
        girdi = (np.array([[veriler[self.index - 1]]]) - min_deger) / (max_deger - min_deger)
        tahmin_rfr = rfr_model.predict(girdi)
        tahmin_rnn = rnn_model.predict(girdi.reshape(1, 1, 1))
        tahmin_lstm = lstm_model.predict(girdi.reshape(1, 1, 1))

        # Nihai Tahmini Hesapla
        nihai_tahmin = (tahmin_rfr[0] * 0.3) + (tahmin_rnn[0][0] * 0.3) + (tahmin_lstm[0][0] * 0.4)
        nihai_tahmin = nihai_tahmin * (max_deger - min_deger) + min_deger  # Gerçek aralığa dönüştür

        # Gerçek değeri al
        gercek = veriler[self.index]
        hata_orani = abs(nihai_tahmin - gercek) / gercek

        # Arayüzü güncelle
        self.label_tahmin_rfr.setText(f"Tahmin (RFR): {tahmin_rfr[0] * (max_deger - min_deger) + min_deger:.2f}")
        self.label_tahmin_rnn.setText(f"Tahmin (RNN): {tahmin_rnn[0][0] * (max_deger - min_deger) + min_deger:.2f}")
        self.label_tahmin_lstm.setText(f"Tahmin (LSTM): {tahmin_lstm[0][0] * (max_deger - min_deger) + min_deger:.2f}")
        self.label_nihai_tahmin.setText(f"Nihai Tahmin: {nihai_tahmin:.2f}")
        self.label_gercek.setText(f"Gerçek Sayı: {gercek:.2f}")
        self.label_hata.setText(f"Hata Oranı: {hata_orani:.2%}")

        # Modeli güncelle
        rfr_model.fit(girdi.reshape(-1, 1), [gercek])
        self.index += 1


# Uygulama başlat
app = QApplication([])
pencere = TahminUygulamasi()
pencere.show()
app.exec_()