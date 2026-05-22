# ==========================================
# HAFTA 4-6: VERİ ÖN İŞLEME, BASELINE VE LSTM MODELİ
# ==========================================

# 1. KÜTÜPHANELERİN YÜKLENMESİ
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Sadece SPY (S&P 500) dosyasını çekiyoruz (Veri sızıntısını önlemek için en güvenli yol)
TEST_DOSYASI = '/content/drive/MyDrive/Data/ETFs/spy.us.txt'
df = pd.read_csv(TEST_DOSYASI)

# Sadece 'Close' (Kapanış) fiyatlarını alıp Numpy matrisine çeviriyoruz
data = df.filter(['Close']).values

# ==========================================
# 2. VERİ ÖN İŞLEME (SCALING)
# ==========================================
# Yapay sinir ağları 0 ile 1 arasındaki sayılarla daha iyi öğrenir
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Veriyi Eğitim (%80) ve Test (%20) olarak bölüyoruz
training_data_len = int(np.ceil(len(data) * .8))
train_data = scaled_data[0:int(training_data_len), :]

# Pencereleme (Windowing): Son 60 günü (X) al, 61. günü (y) tahmin et!
x_train = []
y_train = []
PENCERE_BOYUTU = 60

for i in range(PENCERE_BOYUTU, len(train_data)):
    x_train.append(train_data[i-PENCERE_BOYUTU:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# LSTM Katmanı için veriyi 3 boyutlu hale (Reshape) getirmeliyiz: [Örnek Sayısı, Zaman Adımı, Özellik Sayısı]
x_train_lstm = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# ==========================================
# 3. BASELINE MODEL: LINEAR REGRESSION (KIYASLAMA İÇİN)
# ==========================================
# Hocaların istediği "Basit Model"
print("--- Baseline (Linear Regression) Eğitiliyor ---")
lr_model = LinearRegression()
# Linear Regression 2 boyutlu veri ister, o yüzden x_train kullanıyoruz
lr_model.fit(x_train, y_train) 
print("Baseline model başarıyla eğitildi!")

# ==========================================
# 4. DEEP LEARNING: LSTM MİMARİSİ
# ==========================================
print("\n--- LSTM Modeli Kuruluyor ---")
lstm_model = Sequential()

# 1. LSTM Katmanı (Kılavuz zorunluluğu)
lstm_model.add(LSTM(units=50, return_sequences=False, input_shape=(x_train_lstm.shape[1], 1)))

# 2. Dropout Katmanı (Kılavuz zorunluluğu - Aşırı öğrenmeyi önler)
lstm_model.add(Dropout(0.2))

# 3. Dense Çıktı Katmanı (Kılavuz zorunluluğu - Tek bir fiyat tahmini yapar)
lstm_model.add(Dense(units=1))

# Modeli Derleme (Derlerken 'adam' optimizasyonunu ve 'mean_squared_error' kullanıyoruz)
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# LSTM Modelini Eğitme
print("LSTM Eğitimi Başlıyor (Birkaç dakika sürebilir)...")
lstm_model.fit(x_train_lstm, y_train, batch_size=32, epochs=5)
print("LSTM modeli başarıyla eğitildi!")

# NOT: Test verisi, tahmin grafiği ve metrikler (RMSE, MAE) bir sonraki hücrede yazılacak.
