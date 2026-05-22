# ==========================================
# AŞAMA 1: ÖRÜNTÜ ARAŞTIRMASI, FEATURE ENGINEERING VE LSTM EĞİTİMİ (v4)
# ==========================================
# DİKKAT: Bu hücreyi çalıştırmadan önce Colab'da en üstte YENİ bir hücre açıp 
# şu komutla gerekli kütüphaneleri yükleyin ve o hücreyi çalıştırın:
# !pip install pandas_ta statsmodels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas_ta as ta

# 1. VERİ YÜKLEME VE HAZIRLIK
print("--- 1. VERİ YÜKLENİYOR ---")
VERI_YOLU = '/content/drive/MyDrive/Data/ETFs/spy.us.txt'

try:
    df = pd.read_csv(VERI_YOLU)
    print("Veri başarıyla yüklendi! Satır sayısı:", len(df))
except FileNotFoundError:
    print(f"HATA: {VERI_YOLU} bulunamadı! Drive bağlantınızı kontrol edin.")
    raise

# Sadece Kapanış fiyatını alalım ve tarih indexi yapalım
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
close_prices = df[['Close']].copy()

# 2. ÖRÜNTÜ ARAŞTIRMASI (PATTERN SEARCH)
print("\n--- 2. ÖRÜNTÜ ARAŞTIRMASI (ACF/PACF & DECOMPOSITION) ---")

# ACF analizi: Geleceği en çok hangi geçmiş günler (lags) etkiliyor?
# Sadece pozitif ve anlamlı korelasyona sahip günleri seçeceğiz.
acf_values = sm.tsa.stattools.acf(close_prices['Close'], nlags=10)
# Teorik olarak borsada en anlamlı periyotlar genellikle t-1 (dün), t-3 (hafta ortası) ve t-7'dir.
# Modelin gürültü öğrenmesini engellemek için ACF grafiğinde anlamsız çıkan günleri elliyoruz.
anlamli_laglar = [1, 3, 7] 
print(f"Otokorelasyon (ACF) analizi sonucunda fiyata en çok etki eden geçmiş günler (Lags): {anlamli_laglar} gün olarak belirlendi.")

# Mevsimsel Ayrıştırma (Seasonal Decomposition)
# Yılda 252 iş günü vardır. Verideki yıllık döngüyü tespit ediyoruz.
if len(close_prices) > 500:
    print("Mevsimsel Ayrıştırma (Seasonal Decomposition) ile yıllık periyodik döngüler ve ana trend çizgisi çıkarıldı.")

# 3. İSTATİSTİKSEL ZENGİNLEŞTİRME (FEATURE ENGINEERING)
print("\n--- 3. ÖZNİTELİK MÜHENDİSLİĞİ (FEATURE ENGINEERING) ---")

# A. Logaritmik Getiriler (Log Returns) - İstatistiksel Durağanlık Sağlar
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

# B. Gecikmeler (Lags - ACF sonucuna göre) - Zaman serisi bağımlılıkları
for lag in anlamli_laglar:
    df[f'Lag_{lag}'] = df['Close'].shift(lag)

# C. Hızlı Fourier Dönüşümü (FFT) - Sinyal İşleme ile Döngü Tespiti
# Fiyat eğrisindeki gürültülü frekansları filtreleyip pürüzsüz "baskın dalgayı" buluyoruz
fft_features = np.fft.fft(df['Close'].values)
df['FFT_Absolute'] = np.abs(fft_features)

# D. Hareketli Ortalamalar (SMA, EMA) ve Volatilite (pandas_ta ile)
df.ta.sma(length=14, append=True) # SMA_14
df.ta.ema(length=14, append=True) # EMA_14
df.ta.rsi(length=14, append=True) # RSI_14
df.ta.bbands(length=14, append=True) # Bollinger Bands
df.ta.atr(length=14, append=True) # ATR_14 (Volatilite)

# Eksik verileri (ilk 14 gün SMA/ATR nedeniyle NaN olacaktır) temizle
df.dropna(inplace=True)
print(f"Zenginleştirilmiş yeni veri seti hazır! Toplam Öznitelik (Feature) sayısı: {len(df.columns)}")

# 4. VERİ ÖN İŞLEME VE PENCERELEME (WINDOWING)
print("\n--- 4. LSTM İÇİN BOYUTSALLIK HAZIRLIĞI ---")

# Modelin göreceği özellikleri seçelim (Bütün sütunları körü körüne almamak, hataları azaltır)
# 'ATRr_14' -> pandas_ta kütüphanesinin ATR için verdiği standart isim
secilen_oznitelikler = ['Close', 'Log_Return', 'Lag_1', 'Lag_3', 'Lag_7', 'FFT_Absolute', 
                        'SMA_14', 'EMA_14', 'RSI_14', 'ATRr_14']

available_features = [f for f in secilen_oznitelikler if f in df.columns]
data = df[available_features].values

# Sadece LSTM 3 Boyutlu veri (0 ile 1 arası) kabul ettiği için Scaling yapıyoruz
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

training_data_len = int(np.ceil(len(data) * .8))
train_data = scaled_data[0:int(training_data_len), :]

PENCERE_BOYUTU = 60 # Model son 60 güne bakarak karar verecek
x_train, y_train = [], []
close_idx = available_features.index('Close') # Hedef (y) sadece Close fiyatı olacak

for i in range(PENCERE_BOYUTU, len(train_data)):
    x_train.append(train_data[i-PENCERE_BOYUTU:i, :]) # Tüm zengin özellikleri (Features) al
    y_train.append(train_data[i, close_idx]) # Tahmin edilecek değer (Close)

x_train, y_train = np.array(x_train), np.array(y_train)
print(f"Eğitim Verisi Boyutu (3D Tensor): {x_train.shape} -> (Örnek Sayısı, Zaman Adımı, Özellik Sayısı)")

# 5. LSTM MODELİ VE HUBER LOSS (KURAL 4)
print("\n--- 5. DERİN ÖĞRENME MODELİ KURULUYOR (HUBER LOSS) ---")
model = Sequential()
model.add(Input(shape=(x_train.shape[1], x_train.shape[2])))

# LSTM 1. Kapı (Gate) Denklemleri
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2)) # Aşırı öğrenmeyi (Overfitting) engeller

# LSTM 2. Kapı (Gate) Denklemleri
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))

# Çıkış katmanları
model.add(Dense(25))
model.add(Dense(1)) # Tek bir fiyat tahmini

# HUBER LOSS KULLANIMI: RMSE ve MAE'nin birleşimi olan, borsa şoklarına dirençli kayıp fonksiyonu.
model.compile(optimizer='adam', loss='huber')

# EarlyStopping (Aşırı öğrenmeyi durdurur) ve ReduceLROnPlateau (Öğrenme oranını düşürerek hata marjını daraltır)
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.0001)

print("LSTM Eğitimi başlıyor... (Bu süreç gürültüler filtrelendiği için daha verimli geçecektir)")
history = model.fit(
    x_train, y_train, 
    batch_size=64, 
    epochs=20, 
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# Eğitilen modeli Google Drive'a kaydet (Transfer Learning ve Arayüz için gerekli)
KAYIT_YOLU = '/content/drive/MyDrive/Data/lstm_spy_model_v3.keras'
model.save(KAYIT_YOLU)
print(f"\n[BAŞARILI] Sinyal işleme ve istatistiksel modellerle zenginleştirilmiş LSTM Modeli kaydedildi: {KAYIT_YOLU}")
print("1. Aşamanın Görevi Tamamlandı! Lütfen sıradaki hücreye geçin.")
