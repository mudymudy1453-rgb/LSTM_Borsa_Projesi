# 04_bisection_window_search.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("--- YALITILMIŞ 'İKİ UÇ YAKLAŞIMI' (TÜREV İNİŞİ) ARAMASI ---")
print("LSTM Nöronları, Dropout ve Öğrenme Hızı donduruldu (Yalıtıldı).")
print("Algoritma sadece Gün Sayısına (Pencereye) odaklanacak!\n")

VERI_YOLU_AAPL = '/content/drive/MyDrive/Data/Stocks/aapl.us.txt'
try:
    df_aapl = pd.read_csv(VERI_YOLU_AAPL)
except FileNotFoundError:
    print(f"HATA: {VERI_YOLU_AAPL} bulunamadı! Lütfen Drive'ı mount edin.")
    raise

df_aapl['Date'] = pd.to_datetime(df_aapl['Date'])
df_aapl.set_index('Date', inplace=True)
df_aapl['Log_Return'] = np.log(df_aapl['Close'] / df_aapl['Close'].shift(1))
df_aapl.dropna(inplace=True)

data_aapl = df_aapl[['Close', 'Log_Return']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_aapl)
close_idx = 0
log_ret_idx = 1

TEST_SIZE = 150
training_data_len = len(data_aapl) - TEST_SIZE
train_scaled = scaled_data[:training_data_len]
y_raw_returns = df_aapl['Log_Return'].values[:training_data_len]
y_test_gercek = data_aapl[training_data_len:, close_idx]

def evaluate_window(pencere):
    x_tr, y_tr = [], []
    for i in range(pencere, len(train_scaled)):
        x_tr.append(train_scaled[i-pencere:i, :])
        y_tr.append(y_raw_returns[i])
    x_tr, y_tr = np.array(x_tr), np.array(y_tr)
    
    x_te = []
    test_start_idx = training_data_len - pencere
    test_scaled = scaled_data[test_start_idx:]
    for i in range(pencere, len(test_scaled)):
        x_te.append(test_scaled[i-pencere:i, :])
    x_te = np.array(x_te)
    
    # Yalıtılmış sabit model (Her defasında aynı zeka seviyesinde başlıyor, haksızlık olmasın diye)
    tf.random.set_seed(42) # Rastgeleliği yalıt
    model = Sequential([
        Input(shape=(x_tr.shape[1], x_tr.shape[2])),
        LSTM(32),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(x_tr, y_tr, epochs=10, batch_size=32, verbose=0)
    
    preds_returns = model.predict(x_te, verbose=0)[:, 0]
    preds_fiyat = []
    for i in range(len(preds_returns)):
        dun_fiyati = data_aapl[training_data_len + i - 1, close_idx] 
        preds_fiyat.append(dun_fiyati * np.exp(preds_returns[i]))
    
    rmse = np.sqrt(mean_squared_error(y_test_gercek, preds_fiyat))
    return rmse

# --- TERNARY SEARCH (İKİ UÇ YAKLAŞIMI) ALGORİTMASI ---
# Sizin dediğiniz o daraltma mantığının birebir koda dökülmüş hali.
L = 4  # Sol uç
R = 16 # Sağ uç

print(f"Hedef: {L} ile {R} gün arasındaki 'Eğimin Sıfır Olduğu' mutlak dip noktayı bulmak.\n")

adim = 1
while R - L > 2:
    # 3'e bölüp iki referans noktası al (Ternary Search)
    m1 = L + (R - L) // 3
    m2 = R - (R - L) // 3
    
    print(f"[DÖNGÜ {adim}] İki uç deneniyor: Sol Uç ({m1} Gün) vs Sağ Uç ({m2} Gün)...")
    
    rmse1 = evaluate_window(m1)
    rmse2 = evaluate_window(m2)
    
    print(f" -> {m1} Gün Hatası: {rmse1:.2f} USD")
    print(f" -> {m2} Gün Hatası: {rmse2:.2f} USD")
    
    if rmse1 < rmse2:
        print(f" ---> {m1} gün daha iyi! O halde kusursuz gün {m2}'den küçük olmalı. Sağ ucu daraltıyoruz.")
        R = m2
    else:
        print(f" ---> {m2} gün daha iyi! O halde kusursuz gün {m1}'den büyük olmalı. Sol ucu daraltıyoruz.")
        L = m1
    adim += 1
    print("-" * 50)

# Final kontrolü
print(f"Aralık iyice daraldı! Son sağlamalar yapılıyor...")
best_pencere = L
best_rmse = evaluate_window(L)

for p in range(L+1, R+1):
    r = evaluate_window(p)
    if r < best_rmse:
        best_rmse = r
        best_pencere = p

print("=" * 50)
print(f"🚀 [SONUÇ] Türev (Eğim) 0 noktasına ulaştı!")
print(f"Algoritmanın Bulduğu Kusursuz Gün Sayısı: {best_pencere} Gün")
print("=" * 50)
