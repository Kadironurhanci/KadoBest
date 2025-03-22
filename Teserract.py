import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from torch.utils.tensorboard import SummaryWriter

# SQLAlchemy ile veritabanı bağlantısı
SERVER = 'DESKTOP-EFBJ0MU'
DATABASE = 'kadoDB'
USERNAME = 'sa'
PASSWORD = 'Kadir81onur'

connection_string = f"mssql+pyodbc://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server"
engine = create_engine(connection_string)

# Veritabanı Yöneticisi
class DatabaseManager:
    def __init__(self):
        self.engine = engine

    def fetch_data(self, query):
        try:
            df = pd.read_sql(query, self.engine)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Veri çekme hatası: {e}")
            return None

# LSTM Modeli
class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)  # Modeli direkt olarak cihaza taşı

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Veri İşleme ve Özellik Mühendisliği
class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def create_sequences(self, data, seq_length=60):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def detect_missing(self, df):
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
        df['missing'] = (df['time_diff'] > 180).astype(int)
        return df[['value', 'missing']].values

# Gerçek Zamanlı Tahmin Thread'i
class PredictionThread(QThread):
    update_signal = pyqtSignal(float, float, str)
    error_signal = pyqtSignal(str)

    def __init__(self, db_manager):
        super().__init__()
        self.db = db_manager
        self.model = None
        self.scaler = StandardScaler()
        self.seq_length = 60
        self.running = True
        self.writer = SummaryWriter('runs/time_series_experiment')

    def run(self):
        while self.running:
            try:
                # Verileri çek ve işle
                data = self.db.fetch_data("SELECT TOP 2000 timestamp, value FROM RedNumbers ORDER BY timestamp")
                if data is None:
                    time.sleep(5)
                    continue

                preprocessor = DataPreprocessor()
                processed_data = preprocessor.detect_missing(data)

                # Verileri ölçeklendir
                scaled_data = self.scaler.fit_transform(processed_data)

                # Modeli oluştur ve GPU'ya taşı
                if self.model is None:
                    self.model = TimeSeriesLSTM(input_size=2)  # input_size=2 (value + missing)
                    self.model.to(self.model.device)  # Modeli GPU'ya taşı
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

                X, y = preprocessor.create_sequences(scaled_data, self.seq_length)
                X_tensor = torch.FloatTensor(X).to(self.model.device)
                y_tensor = torch.FloatTensor(y[:, 0]).unsqueeze(1).to(self.model.device)  # Sadece 'value' sütununu al

                # Eğitim döngüsü
                self.model.train()
                for epoch in range(10):
                    optimizer.zero_grad()
                    outputs = self.model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()
                    self.writer.add_scalar('Loss/train', loss.item(), epoch)
                    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

                # Tahmin yap
                with torch.no_grad():
                    self.model.eval()
                    last_sequence = scaled_data[-self.seq_length:]
                    last_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.model.device)
                    prediction = self.model(last_tensor).cpu().numpy()
                    prediction = self.scaler.inverse_transform(
                        np.concatenate((prediction, [[0]]), axis=1))[:, 0]

                    # Gerçek zamanlı güncelleme
                    current_value = data['value'].iloc[-1]
                    self.update_signal.emit(current_value, prediction[0], time.strftime('%H:%M:%S'))

                time.sleep(1)  # 1 saniyede bir güncelleme

            except Exception as e:
                self.error_signal.emit(f"Hata: {str(e)}")
                time.sleep(1)

# PyQt Arayüzü
class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.db = DatabaseManager()
        self.initUI()
        self.start_prediction_thread()

    def initUI(self):
        self.setWindowTitle('Gerçek Zamanlı Tahmin Sistemi')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.value_label = QLabel('Anlık Değer: -')
        self.pred_label = QLabel('Tahmin: -')
        self.time_label = QLabel('Son Güncelleme: -')

        layout.addWidget(self.value_label)
        layout.addWidget(self.pred_label)
        layout.addWidget(self.time_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def start_prediction_thread(self):
        self.thread = PredictionThread(self.db)
        self.thread.update_signal.connect(self.update_ui)
        self.thread.error_signal.connect(self.show_error)
        self.thread.start()

    def update_ui(self, current, prediction, timestamp):
        self.value_label.setText(f"Anlık Değer: {current:.2f}")
        self.pred_label.setText(f"Tahmin: {prediction:.2f}")
        self.time_label.setText(f"Son Güncelleme: {timestamp}")

    def show_error(self, message):
        self.time_label.setText(message)

    def closeEvent(self, event):
        self.thread.running = False
        self.thread.quit()
        event.accept()

if __name__ == "__main__":
    app = QApplication([])
    window = MainApp()
    window.show()
    app.exec_()