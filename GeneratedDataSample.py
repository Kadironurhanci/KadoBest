from multiprocessing import connection

import pyodbc
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Veritabanı bağlantısı
conn = pyodbc.connect(
      'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=DESKTOP-EFBJ0MU;'  # Sunucu adını yaz
    'DATABASE=kadoDB;'  # Veritabanı adını yaz
    'UID=sa;'  # Kullanıcı adını yaz
    'PWD=Kadir81onur;') # Şifreni yaz)
cursor = conn.cursor()


# Gerçek verileri veritabanından alma fonksiyonu
def get_real_data():
    cursor.execute("SELECT Value, TimeStamp FROM RedNumbers")
    data = cursor.fetchall()
    real_values = [row[0] for row in data]
    return np.array(real_values)


# LSTM modelini oluşturma fonksiyonu
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# Gerçek verilerle LSTM modeli eğitme ve tahmin yapma fonksiyonu
def train_and_predict_lstm(real_data):
    # Veriyi MinMaxScaler ile normalize etme
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(real_data.reshape(-1, 1))

    # Eğitim verisini hazırlama
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # LSTM modelini oluşturma ve eğitme
    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

    # Son tahmini yapmak için en son veriyi alıp tahmin etme
    last_60_days = scaled_data[-60:]
    X_test = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
    predicted_value = model.predict(X_test)

    # Tahmin edilen değeri orijinal ölçeğe geri dönüştürme
    predicted_value = scaler.inverse_transform(predicted_value)

    return predicted_value[0][0]


# Veritabanına tahmin edilen veriyi ekleme fonksiyonu
def insert_predicted_data(predicted_value=7.8, predicted_time=datetime.datetime.now(), model_version="LSTM v1", loss=0.0023, epoch=5):
    # SQL sorgusunu tanımlayın
    insert_query = """
    INSERT INTO GeneratedData (GeneratedVal, GeneratedAt, ModelVersion, Loss, Epoch)
    VALUES (?, ?, ?, ?, ?)
    """

    # Bağlantıyı ve cursor'ı kullanarak veriyi ekleyin
    cursor.execute(insert_query, (float(predicted_value), predicted_time, model_version, loss, epoch))
    connection.commit()  # Değişiklikleri veritabanına kaydedin


# Gerçek veriyi alıp, LSTM ile tahmin yapalım
real_data = get_real_data()  # Veritabanından gerçek veriyi alıyoruz
predicted_value = train_and_predict_lstm(real_data)

# Tahmin edilen veriyi veritabanına ekleyelim
insert_predicted_data(predicted_value)
print("Tahmin edilen veri başarıyla veritabanına eklendi!")
