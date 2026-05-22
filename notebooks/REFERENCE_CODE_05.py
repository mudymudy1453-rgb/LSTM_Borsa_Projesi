# ==========================================
# REFERANS KOD 05: BÜYÜK BİRLEŞİK (GRAND UNIFIED) OPTİMİZASYON
# ==========================================
# DİKKAT: Bu kod referanstır. Her çalışmada aynı sonucu vermesi için SEED eklenmiştir.
# Aşırı öğrenmeyi önlemek için EarlyStopping ve modelleri kaydetme mekanizması eklenmiştir.

import os
import random
os.environ['PYTHONHASHSEED'] = '42'
import pandas as pd
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import optuna
import pandas_ta as ta
import warnings
from scipy.optimize import minimize

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR') 

print("--- 1. VERİ YÜKLEME VE KUANTİTATİF ZENGİNLEŞTİRME ---")
VERI_YOLU_AAPL = '/content/drive/MyDrive/Data/Stocks/aapl.us.txt'

try:
    df_aapl = pd.read_csv(VERI_YOLU_AAPL)
except FileNotFoundError:
    print(f"HATA: {VERI_YOLU_AAPL} bulunamadı! Lütfen Google Drive'ı mount edin.")
    raise

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

print("\n--- 2. TAVİZSİZ (GRAND UNIFIED) HİPERPARAMETRE VE PERİYOT AVI ---")
optuna.logging.set_verbosity(optuna.logging.WARNING) 

def create_objective(min_pencere, max_pencere, is_short, loss_type):
    def objective(trial):
        pencere = trial.suggest_int('pencere', min_pencere, max_pencere)
        lstm_units = trial.suggest_categorical('lstm_units', [16, 32, 64, 128])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.4)
        lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        x_tr, y_tr, _ = prepare_data(pencere, scaled_data_aapl, y_raw_returns, is_short=is_short)
        
        tscv = TimeSeriesSplit(n_splits=2)
        fold_errors = []
        
        tr_len = len(x_tr)
        start_idx = max(0, tr_len - 1000)
        x_cv = x_tr[start_idx:]
        y_cv = y_tr[start_idx:]
        
        for train_idx, val_idx in tscv.split(x_cv):
            X_t, X_v = x_cv[train_idx], x_cv[val_idx]
            y_t, y_v = y_cv[train_idx], y_cv[val_idx]
            
            m = Sequential([
                Input(shape=(X_t.shape[1], X_t.shape[2])),
                LSTM(lstm_units),
                Dropout(dropout_rate),
                Dense(1)
            ])
            m.compile(optimizer=Adam(learning_rate=lr), loss=loss_type)
            m.fit(X_t, y_t, epochs=6, verbose=0, batch_size=32)
            
            preds = m.predict(X_v, verbose=0)
            fold_errors.append(mean_squared_error(y_v, preds))
            
        return np.mean(fold_errors)
    return objective

print("-> 1/3 KISA VADE (Panik Avcısı | Hedef: 3-15 Gün) - 50 Deneme Taranıyor...")
study_kisa = optuna.create_study(direction='minimize')
study_kisa.optimize(create_objective(3, 15, True, 'mse'), n_trials=50)
bp_k = study_kisa.best_params
print(f"[Kusursuz Kısa Vade] Periyot: {bp_k['pencere']} Gün, Nöron: {bp_k['lstm_units']}, LR: {bp_k['learning_rate']:.4f}")

print("-> 2/3 ORTA VADE (Momentum Avcısı | Hedef: 16-45 Gün) - 50 Deneme Taranıyor...")
study_orta = optuna.create_study(direction='minimize')
study_orta.optimize(create_objective(16, 45, False, 'huber'), n_trials=50)
bp_o = study_orta.best_params
print(f"[Kusursuz Orta Vade] Periyot: {bp_o['pencere']} Gün, Nöron: {bp_o['lstm_units']}, LR: {bp_o['learning_rate']:.4f}")

print("-> 3/3 UZUN VADE (Trend Avcısı | Hedef: 46-90 Gün) - 50 Deneme Taranıyor...")
study_uzun = optuna.create_study(direction='minimize')
study_uzun.optimize(create_objective(46, 90, False, 'huber'), n_trials=50)
bp_u = study_uzun.best_params
print(f"[Kusursuz Uzun Vade] Periyot: {bp_u['pencere']} Gün, Nöron: {bp_u['lstm_units']}, LR: {bp_u['learning_rate']:.4f}")

print("\n--- 3. KUSURSUZ PARAMETRELERLE NİHAİ EĞİTİM ---")
early_stop_final = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

# 1. KISA MODEL
x_tr_k, y_tr_k, x_te_k = prepare_data(bp_k['pencere'], scaled_data_aapl, y_raw_returns, is_short=True)
kisa_model = Sequential([
    Input(shape=(x_tr_k.shape[1], x_tr_k.shape[2])),
    LSTM(bp_k['lstm_units']),
    Dropout(bp_k['dropout_rate']),
    Dense(1)
])
kisa_model.compile(optimizer=Adam(learning_rate=bp_k['learning_rate']), loss='mse')
kisa_model.fit(x_tr_k, y_tr_k, batch_size=32, epochs=40, verbose=0, callbacks=[early_stop_final])
preds_returns = kisa_model.predict(x_te_k, verbose=0)[:, 0]
preds_kisa_fiyat = []
for i in range(len(preds_returns)):
    dun_fiyati = data_aapl[training_data_len + i - 1, close_idx] 
    preds_kisa_fiyat.append(dun_fiyati * np.exp(preds_returns[i]))
preds_kisa_fiyat = np.array(preds_kisa_fiyat)
kisa_model.save('champion_kisa.keras')

# 2. ORTA MODEL (ATTENTION)
x_tr_o, y_tr_o, x_te_o = prepare_data(bp_o['pencere'], scaled_data_aapl, is_short=False)
inputs_orta = Input(shape=(x_tr_o.shape[1], x_tr_o.shape[2]))
lstm_out1 = LSTM(bp_o['lstm_units'], return_sequences=True)(inputs_orta)
attention_out = Attention()([lstm_out1, lstm_out1])
lstm_out2 = LSTM(16, return_sequences=False)(attention_out) 
dropout_orta = Dropout(bp_o['dropout_rate'])(lstm_out2)
outputs_orta = Dense(1)(dropout_orta)
orta_model = Model(inputs=inputs_orta, outputs=outputs_orta)
orta_model.compile(optimizer=Adam(learning_rate=bp_o['learning_rate']), loss='huber')
orta_model.fit(x_tr_o, y_tr_o, batch_size=32, epochs=30, verbose=0, callbacks=[early_stop_final])
preds_orta_scaled = orta_model.predict(x_te_o, verbose=0)
dummy = np.zeros((len(preds_orta_scaled), len(secilen_oznitelikler)))
dummy[:, close_idx] = preds_orta_scaled[:, 0]
preds_orta_fiyat = scaler_aapl.inverse_transform(dummy)[:, close_idx]
orta_model.save('champion_orta.keras')

# 3. UZUN MODEL
x_tr_u, y_tr_u, x_te_u = prepare_data(bp_u['pencere'], scaled_data_aapl, is_short=False)
uzun_model = Sequential([
    Input(shape=(x_tr_u.shape[1], x_tr_u.shape[2])),
    LSTM(bp_u['lstm_units']),
    Dropout(bp_u['dropout_rate']),
    Dense(1)
])
uzun_model.compile(optimizer=Adam(learning_rate=bp_u['learning_rate']), loss='huber')
uzun_model.fit(x_tr_u, y_tr_u, batch_size=32, epochs=20, verbose=0, callbacks=[early_stop_final])
preds_uzun_scaled = uzun_model.predict(x_te_u, verbose=0)
dummy[:, close_idx] = preds_uzun_scaled[:, 0]
preds_uzun_fiyat = scaler_aapl.inverse_transform(dummy)[:, close_idx]
uzun_model.save('champion_uzun.keras')

print("\n--- 4. SCIPY MATEMATİKSEL FÜZYON OPTİMİZASYONU ---")
def ensemble_loss(weights):
    w_k, w_o, w_u = weights
    blended = (w_k * preds_kisa_fiyat) + (w_o * preds_orta_fiyat) + (w_u * preds_uzun_fiyat)
    return np.sqrt(mean_squared_error(y_test_gercek, blended))

init_weights = [0.33, 0.33, 0.34] 
bounds = ((0, 1), (0, 1), (0, 1)) 
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}) 

res = minimize(ensemble_loss, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
w_k_opt, w_o_opt, w_u_opt = res.x

np.save('champion_weights.npy', np.array([w_k_opt, w_o_opt, w_u_opt]))
print(f"✅ BULUNAN ALTIN ORANLAR: Kısa(%{w_k_opt*100:.1f}), Orta(%{w_o_opt*100:.1f}), Uzun(%{w_u_opt*100:.1f})")

ensemble_preds = (w_k_opt * preds_kisa_fiyat) + (w_o_opt * preds_orta_fiyat) + (w_u_opt * preds_uzun_fiyat)

print("\n--- 5. GRAND UNIFIED NİHAİ METRİKLER VE GRAFİK ---")
rmse = np.sqrt(mean_squared_error(y_test_gercek, ensemble_preds))
mae = mean_absolute_error(y_test_gercek, ensemble_preds)

print("-" * 50)
print(f"GRAND UNIFIED ENSEMBLE SONUÇLARI")
print(f"Ensemble RMSE: {rmse:.2f} USD")
print(f"Ensemble MAE:  {mae:.2f} USD")
print("-" * 50)

print("\n[BAŞARILI] Büyük Birleşik Optimizasyon tamamlandı ve Modeller (.keras) olarak Kaydedildi!")
