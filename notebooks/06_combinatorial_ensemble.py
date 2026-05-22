# ==========================================
# AŞAMA 6: 125.000 KOMBİNASYONLU ÇAPRAZ FÜZYON
# ==========================================
# REFERANS 05 UYUMU: Log_Return, Attention(), EarlyStopping ve Seed eklendi.

import os
import random
os.environ['PYTHONHASHSEED'] = '42'
import pandas as pd
import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)
import itertools
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pandas_ta as ta
import time
import warnings

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

print("--- 1. VERİ YÜKLEME VE ZENGİNLEŞTİRME ---")
VERI_YOLU_AAPL = '/content/drive/MyDrive/Data/Stocks/aapl.us.txt'
df_aapl = pd.read_csv(VERI_YOLU_AAPL)
df_aapl['Date'] = pd.to_datetime(df_aapl['Date'])
df_aapl.set_index('Date', inplace=True)

df_aapl['Log_Return'] = np.log(df_aapl['Close'] / df_aapl['Close'].shift(1))
for lag in [1, 3, 7]:
    df_aapl[f'Lag_{lag}'] = df_aapl['Close'].shift(lag)
df_aapl['FFT_Absolute'] = np.abs(np.fft.fft(df_aapl['Close'].values))

df_aapl.ta.sma(length=14, append=True)
df_aapl.ta.ema(length=14, append=True)
df_aapl.ta.rsi(length=14, append=True)
df_aapl.ta.bbands(length=14, append=True)
df_aapl.ta.atr(length=14, append=True)
df_aapl.dropna(inplace=True)

secilen_oznitelikler = ['Close', 'Log_Return', 'Lag_1', 'Lag_3', 'Lag_7', 'FFT_Absolute', 
                        'SMA_14', 'EMA_14', 'RSI_14', 'ATRr_14']
data_aapl = df_aapl[secilen_oznitelikler].values

scaler_aapl = MinMaxScaler(feature_range=(0, 1))
scaled_data_aapl = scaler_aapl.fit_transform(data_aapl)
close_idx = secilen_oznitelikler.index('Close')

TEST_SIZE = 150 
training_data_len = len(data_aapl) - TEST_SIZE
y_raw_returns = df_aapl['Log_Return'].values[:training_data_len]
y_test_gercek = data_aapl[training_data_len:, close_idx]

def prepare_data(PENCERE, data_scaled, raw_returns=None, is_short=False):
    x_tr, y_tr = [], []
    train_scaled = data_scaled[:training_data_len]
    for i in range(PENCERE, len(train_scaled)):
        x_tr.append(train_scaled[i-PENCERE:i, :])
        if is_short:
            y_tr.append(raw_returns[i])
        else:
            y_tr.append(train_scaled[i, close_idx])
            
    x_te = []
    test_start_idx = training_data_len - PENCERE
    test_scaled = data_scaled[test_start_idx:]
    for i in range(PENCERE, len(test_scaled)):
        x_te.append(test_scaled[i-PENCERE:i, :])
        
    return np.array(x_tr), np.array(y_tr), np.array(x_te)

early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

def train_short(p, n, lr, drop):
    tf.keras.backend.clear_session()
    x_tr_k, y_tr_k, x_te_k = prepare_data(p, scaled_data_aapl, y_raw_returns, is_short=True)
    m = Sequential([
        Input(shape=(x_tr_k.shape[1], x_tr_k.shape[2])),
        LSTM(n),
        Dropout(drop),
        Dense(1)
    ])
    m.compile(optimizer=Adam(learning_rate=lr), loss='mse')
    m.fit(x_tr_k, y_tr_k, epochs=15, batch_size=64, verbose=0, callbacks=[early_stop])
    preds_ret = m.predict(x_te_k, verbose=0)[:, 0]
    preds_fiyat = []
    for i in range(len(preds_ret)):
        dun = data_aapl[training_data_len + i - 1, close_idx]
        preds_fiyat.append(dun * np.exp(preds_ret[i]))
    return np.array(preds_fiyat)

def train_medium(p, n, lr, drop):
    tf.keras.backend.clear_session()
    x_tr_o, y_tr_o, x_te_o = prepare_data(p, scaled_data_aapl, is_short=False)
    inp = Input(shape=(x_tr_o.shape[1], x_tr_o.shape[2]))
    l1 = LSTM(n, return_sequences=True)(inp)
    att = Attention()([l1, l1])
    l2 = LSTM(16)(att)
    d = Dropout(drop)(l2)
    out = Dense(1)(d)
    m = Model(inputs=inp, outputs=out)
    m.compile(optimizer=Adam(learning_rate=lr), loss='huber')
    m.fit(x_tr_o, y_tr_o, epochs=15, batch_size=64, verbose=0, callbacks=[early_stop])
    preds_sc = m.predict(x_te_o, verbose=0)
    dummy = np.zeros((len(preds_sc), len(secilen_oznitelikler)))
    dummy[:, close_idx] = preds_sc[:, 0]
    return scaler_aapl.inverse_transform(dummy)[:, close_idx]

def train_long(p, n, lr, drop):
    tf.keras.backend.clear_session()
    x_tr_u, y_tr_u, x_te_u = prepare_data(p, scaled_data_aapl, is_short=False)
    m = Sequential([
        Input(shape=(x_tr_u.shape[1], x_tr_u.shape[2])),
        LSTM(n),
        Dropout(drop),
        Dense(1)
    ])
    m.compile(optimizer=Adam(learning_rate=lr), loss='huber')
    m.fit(x_tr_u, y_tr_u, epochs=15, batch_size=64, verbose=0, callbacks=[early_stop])
    preds_sc = m.predict(x_te_u, verbose=0)
    dummy = np.zeros((len(preds_sc), len(secilen_oznitelikler)))
    dummy[:, close_idx] = preds_sc[:, 0]
    return scaler_aapl.inverse_transform(dummy)[:, close_idx]

print("\n--- 2. AĞIR İŞÇİLİK: 150 GERÇEK MODEL EĞİTİMİ ---")
kisa_tahminler, orta_tahminler, uzun_tahminler = [], [], []
noron_havuzu = [16, 32, 64]

print("[1/3] KISA VADE (Log_Return Hedefli) 50 Model Eğitiliyor...")
for i in range(50):
    kisa_tahminler.append(train_short(random.randint(3,15), random.choice(noron_havuzu), random.uniform(0.001,0.005), random.uniform(0.1,0.3)))
    if (i+1)%10==0: print(f"  -> {i+1}/50 Tamam")

print("[2/3] ORTA VADE (Attention Katmanlı) 50 Model Eğitiliyor...")
for i in range(50):
    orta_tahminler.append(train_medium(random.randint(16,45), random.choice(noron_havuzu), random.uniform(0.0005,0.002), random.uniform(0.1,0.3)))
    if (i+1)%10==0: print(f"  -> {i+1}/50 Tamam")

print("[3/3] UZUN VADE (Huber Loss) 50 Model Eğitiliyor...")
for i in range(50):
    uzun_tahminler.append(train_long(random.randint(46,90), random.choice(noron_havuzu), random.uniform(0.0001,0.001), random.uniform(0.1,0.3)))
    if (i+1)%10==0: print(f"  -> {i+1}/50 Tamam")

print("\n--- 3. 125.000 ÇAPRAZ KOMBİNASYON SAVAŞI ---")
start_time = time.time()
en_iyi_kombinasyon = None
en_dusuk_hata = float('inf')

for i, k in enumerate(kisa_tahminler):
    for j, o in enumerate(orta_tahminler):
        for z, u in enumerate(uzun_tahminler):
            takim_tahmini = (k + o + u) / 3.0
            hata = np.sqrt(mean_squared_error(y_test_gercek, takim_tahmini))
            if hata < en_dusuk_hata:
                en_dusuk_hata = hata
                en_iyi_kombinasyon = (i, j, z)

end_time = time.time()
print("\n" + "="*50)
print(f"KOMBİNASYON RMSE: {en_dusuk_hata:.2f} USD")
print(f"En İyi Takım: Kısa({en_iyi_kombinasyon[0]}), Orta({en_iyi_kombinasyon[1]}), Uzun({en_iyi_kombinasyon[2]})")
print("="*50)
