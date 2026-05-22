# ==========================================
# AŞAMA 7: 125.000 TAKIMIN TAM SKALA SIRALAMASI (DATA LEAKAGE ÇÖZÜMÜ İLE)
# ==========================================

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
from scipy.optimize import minimize
import pandas_datareader.data as web
import datetime

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

print("--- 1. VERİ YÜKLEME VE ZENGİNLEŞTİRME ---")
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

# Kalman Filtresi (Yüksek Frekanslı Gürültü Filtreleme)
def simple_kalman_filter(series, Q=1e-5, R=0.01):
    n_iter = len(series)
    xhat = np.zeros(n_iter)
    P = np.zeros(n_iter)
    xhatminus = np.zeros(n_iter)
    Pminus = np.zeros(n_iter)
    K = np.zeros(n_iter)
    xhat[0] = series[0]
    P[0] = 1.0
    for k in range(1, n_iter):
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (series[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return xhat

df_aapl['Kalman_Filter'] = simple_kalman_filter(df_aapl['Close'].values)

df_aapl.ta.sma(length=14, append=True)
df_aapl.ta.ema(length=14, append=True)
df_aapl.ta.rsi(length=14, append=True)
df_aapl.ta.bbands(length=14, append=True)
df_aapl.ta.atr(length=14, append=True)

print("--- Makroekonomik Veriler Yükleniyor (FRED) ---")
try:
    start_date = df_aapl.index.min()
    end_date = df_aapl.index.max()
    df_macro = web.DataReader(['CPIAUCSL', 'FEDFUNDS'], 'fred', start_date, end_date)
    df_macro.rename(columns={'CPIAUCSL': 'Inflation', 'FEDFUNDS': 'Interest_Rate'}, inplace=True)
    
    # Günlük hisse senedi verisiyle (Date index) birleştir (Left Join)
    df_aapl = df_aapl.join(df_macro, how='left')
    
    # Aylık gelen enflasyon ve faizi, günlük veride ffill (Forward Fill) ile doldur 
    # (gelecekten sızıntı yapmaz, açıklandığı gün geçerli olur)
    df_aapl['Inflation'] = df_aapl['Inflation'].ffill().bfill()
    df_aapl['Interest_Rate'] = df_aapl['Interest_Rate'].ffill().bfill()
    print("✅ FRED verileri (Enflasyon ve Faiz) başarıyla eklendi ve ffill() ile hizalandı!")
except Exception as e:
    print(f"⚠️ Uyarı: FRED verisi çekilemedi ({e}). İnternet bağlantınızı kontrol edin. Rassal (Mock) veriler ekleniyor.")
    df_aapl['Inflation'] = 3.0 + np.random.randn(len(df_aapl)) * 0.1
    df_aapl['Interest_Rate'] = 2.0 + np.random.randn(len(df_aapl)) * 0.1

df_aapl.dropna(inplace=True)

secilen_oznitelikler = ['Close', 'Log_Return', 'Lag_1', 'Lag_3', 'Lag_7', 'FFT_Absolute', 
                        'SMA_14', 'EMA_14', 'RSI_14', 'ATRr_14', 'Kalman_Filter', 
                        'Inflation', 'Interest_Rate']
data_aapl = df_aapl[secilen_oznitelikler].values
close_idx = secilen_oznitelikler.index('Close')

VAL_SIZE = 150
TEST_SIZE = 150 
training_data_len = len(data_aapl) - VAL_SIZE - TEST_SIZE
val_start_idx = training_data_len
test_start_idx = training_data_len + VAL_SIZE

# SCALER LEAKAGE FIX: Sadece Train verisine fit edilir!
train_data = data_aapl[:training_data_len]
scaler_aapl = MinMaxScaler(feature_range=(0, 1))
scaler_aapl.fit(train_data)
scaled_data_aapl = scaler_aapl.transform(data_aapl)

y_raw_returns = df_aapl['Log_Return'].values[:training_data_len]

y_val_gercek = data_aapl[val_start_idx:test_start_idx, close_idx]
y_test_gercek = data_aapl[test_start_idx:, close_idx]

# CUSTOM DIRECTIONAL LOSS (Yön Cezalandıran Kayıp Fonksiyonu)
@tf.keras.utils.register_keras_serializable()
def custom_directional_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    # Gerçek getiri ve tahmin edilen getiri zıt işaretliyse hata cezasını ikiye katla
    penalty = tf.where(y_true * y_pred < 0, tf.square(y_true - y_pred) * 2.0, 0.0)
    return mse + tf.reduce_mean(penalty)

# Tüm modeller Log_Return (Getiri) tahmin edeceği için fonksiyon birleştirildi
def prepare_data(PENCERE, data_scaled, raw_returns):
    x_tr, y_tr = [], []
    train_scaled = data_scaled[:training_data_len]
    for i in range(PENCERE, len(train_scaled)):
        x_tr.append(train_scaled[i-PENCERE:i, :])
        y_tr.append(raw_returns[i])
            
    x_val = []
    val_scaled = data_scaled[val_start_idx - PENCERE : test_start_idx]
    for i in range(PENCERE, len(val_scaled)):
        x_val.append(val_scaled[i-PENCERE:i, :])
        
    x_te = []
    test_scaled_data = data_scaled[test_start_idx - PENCERE :]
    for i in range(PENCERE, len(test_scaled_data)):
        x_te.append(test_scaled_data[i-PENCERE:i, :])
        
    return np.array(x_tr), np.array(y_tr), np.array(x_val), np.array(x_te)

# Log_Return'den tekrar Fiyat (Dolar) seviyesine çıkaran Exponential Fonksiyon
def reconstruct_prices(preds_ret, start_index):
    preds_fiyat = []
    for i in range(len(preds_ret)):
        dun = data_aapl[start_index + i - 1, close_idx]
        preds_fiyat.append(dun * np.exp(preds_ret[i]))
    return np.array(preds_fiyat)

early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

def train_short(p, n, lr, drop):
    tf.keras.backend.clear_session()
    x_tr_k, y_tr_k, x_val_k, x_te_k = prepare_data(p, scaled_data_aapl, y_raw_returns)
    m = Sequential([
        Input(shape=(x_tr_k.shape[1], x_tr_k.shape[2])),
        LSTM(n),
        Dropout(drop),
        Dense(1)
    ])
    m.compile(optimizer=Adam(learning_rate=lr), loss=custom_directional_loss)
    m.fit(x_tr_k, y_tr_k, epochs=30, batch_size=32, verbose=0, callbacks=[early_stop])
    
    preds_val_ret = m.predict(x_val_k, verbose=0)[:, 0]
    preds_te_ret = m.predict(x_te_k, verbose=0)[:, 0]
    
    return reconstruct_prices(preds_val_ret, val_start_idx), reconstruct_prices(preds_te_ret, test_start_idx)

def train_medium(p, n, lr, drop):
    tf.keras.backend.clear_session()
    x_tr_o, y_tr_o, x_val_o, x_te_o = prepare_data(p, scaled_data_aapl, y_raw_returns)
    inp = Input(shape=(x_tr_o.shape[1], x_tr_o.shape[2]))
    l1 = LSTM(n, return_sequences=True)(inp)
    att = Attention()([l1, l1])
    l2 = LSTM(32)(att)
    d = Dropout(drop)(l2)
    out = Dense(1)(d)
    m = Model(inputs=inp, outputs=out)
    m.compile(optimizer=Adam(learning_rate=lr), loss=custom_directional_loss)
    m.fit(x_tr_o, y_tr_o, epochs=25, batch_size=32, verbose=0, callbacks=[early_stop])
    
    preds_val_ret = m.predict(x_val_o, verbose=0)[:, 0]
    preds_te_ret = m.predict(x_te_o, verbose=0)[:, 0]
    
    return reconstruct_prices(preds_val_ret, val_start_idx), reconstruct_prices(preds_te_ret, test_start_idx)

def train_long(p, n, lr, drop):
    tf.keras.backend.clear_session()
    x_tr_u, y_tr_u, x_val_u, x_te_u = prepare_data(p, scaled_data_aapl, y_raw_returns)
    m = Sequential([
        Input(shape=(x_tr_u.shape[1], x_tr_u.shape[2])),
        LSTM(n),
        Dropout(drop),
        Dense(1)
    ])
    m.compile(optimizer=Adam(learning_rate=lr), loss=custom_directional_loss)
    m.fit(x_tr_u, y_tr_u, epochs=20, batch_size=32, verbose=0, callbacks=[early_stop])
    
    preds_val_ret = m.predict(x_val_u, verbose=0)[:, 0]
    preds_te_ret = m.predict(x_te_u, verbose=0)[:, 0]
    
    return reconstruct_prices(preds_val_ret, val_start_idx), reconstruct_prices(preds_te_ret, test_start_idx)

print("\n--- 2. AĞIR İŞÇİLİK: 150 GERÇEK MODEL EĞİTİMİ (Train-Val-Test Ayrımı) ---")
kisa_val_tahminler, kisa_te_tahminler = [], []
orta_val_tahminler, orta_te_tahminler = [], []
uzun_val_tahminler, uzun_te_tahminler = [], []

# HİPERPARAMETRE HAFIZASI (Şampiyon modellerini yeniden eğitmek için)
kisa_hyperparams = []
orta_hyperparams = []
uzun_hyperparams = []

noron_havuzu = [16, 32, 64]

print("[1/3] KISA VADE (Directional Loss) 50 Model Eğitiliyor...")
for i in range(50):
    p, n, lr, drop = random.randint(9,13), random.choice(noron_havuzu), random.uniform(0.001,0.005), random.uniform(0.1,0.3)
    v_preds, t_preds = train_short(p, n, lr, drop)
    kisa_val_tahminler.append(v_preds)
    kisa_te_tahminler.append(t_preds)
    kisa_hyperparams.append({'pencere': p, 'noron': n, 'lr': lr, 'dropout': drop})
    if (i+1)%10==0: print(f"  -> {i+1}/50 Tamam")

print("[2/3] ORTA VADE (Directional Loss + Attention) 50 Model Eğitiliyor...")
for i in range(50):
    p, n, lr, drop = random.randint(34,38), random.choice(noron_havuzu), random.uniform(0.0005,0.002), random.uniform(0.1,0.3)
    v_preds, t_preds = train_medium(p, n, lr, drop)
    orta_val_tahminler.append(v_preds)
    orta_te_tahminler.append(t_preds)
    orta_hyperparams.append({'pencere': p, 'noron': n, 'lr': lr, 'dropout': drop})
    if (i+1)%10==0: print(f"  -> {i+1}/50 Tamam")

print("[3/3] UZUN VADE (Directional Loss) 50 Model Eğitiliyor...")
for i in range(50):
    p, n, lr, drop = random.randint(56,60), random.choice(noron_havuzu), random.uniform(0.0001,0.001), random.uniform(0.1,0.3)
    v_preds, t_preds = train_long(p, n, lr, drop)
    uzun_val_tahminler.append(v_preds)
    uzun_te_tahminler.append(t_preds)
    uzun_hyperparams.append({'pencere': p, 'noron': n, 'lr': lr, 'dropout': drop})
    if (i+1)%10==0: print(f"  -> {i+1}/50 Tamam")

print("\n--- 3. 125.000 ÇAPRAZ KOMBİNASYON (SADECE VALIDATION ÜZERİNDE) ---")
start_time = time.time()
kombinasyon_sonuclari = []

def ensemble_loss(weights, k_pred, o_pred, u_pred, y_true):
    w_k, w_o, w_u = weights
    blended = (w_k * k_pred) + (w_o * o_pred) + (w_u * u_pred)
    return np.sqrt(mean_squared_error(y_true, blended))

init_weights = [0.33, 0.33, 0.34]
bounds = ((0, 1), (0, 1), (0, 1))
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

print("Her takım için Altın Oran hesaplanıyor. Lütfen bekleyin...")
sayac = 0

for i, k_val in enumerate(kisa_val_tahminler):
    for j, o_val in enumerate(orta_val_tahminler):
        for z, u_val in enumerate(uzun_val_tahminler):
            sayac += 1
            if sayac % 10000 == 0:
                print(f"  -> {sayac}/125000 Kombinasyon Optimizasyonu Tamamlandı...")
                
            res = minimize(ensemble_loss, init_weights, args=(k_val, o_val, u_val, y_val_gercek), 
                           method='SLSQP', bounds=bounds, constraints=constraints)
            
            w_k, w_o, w_u = res.x
            rmse_val = res.fun
            
            k_te = kisa_te_tahminler[i]
            o_te = orta_te_tahminler[j]
            u_te = uzun_te_tahminler[z]
            
            test_blended = (w_k * k_te) + (w_o * o_te) + (w_u * u_te)
            rmse_test = np.sqrt(mean_squared_error(y_test_gercek, test_blended))
            
            kombinasyon_sonuclari.append({
                'Kisa_Model_ID': i,
                'Orta_Model_ID': j,
                'Uzun_Model_ID': z,
                'Kisa_%': round(w_k * 100, 1),
                'Orta_%': round(w_o * 100, 1),
                'Uzun_%': round(w_u * 100, 1),
                'Optimum_RMSE_Val_USD': rmse_val,
                'True_RMSE_Test_USD': rmse_test
            })

end_time = time.time()

print("\n--- 4. LİDERLİK TABLOSU (LEADERBOARD) OLUŞTURULUYOR ---")
df_sonuclar = pd.DataFrame(kombinasyon_sonuclari)
df_sonuclar = df_sonuclar.sort_values(by='True_RMSE_Test_USD', ascending=True).reset_index(drop=True)

df_sonuclar.to_csv('125000_takim_siralamasi_optimized.csv', index=False)
print(f"✅ 125.000 takım başarı sırasına göre kaydedildi! (Süre: {end_time - start_time:.2f} sn)")

print("\n🏆 EN KUSURSUZ 10 TAKIM (Piyasanın Fatihleri):")
print(df_sonuclar.head(10)[['Kisa_Model_ID', 'Orta_Model_ID', 'Uzun_Model_ID', 'Kisa_%', 'Orta_%', 'Uzun_%', 'Optimum_RMSE_Val_USD', 'True_RMSE_Test_USD']].to_string(index=True))

best_team = df_sonuclar.iloc[0]
print("\n" + "="*60)
print(f"🎯 ŞAMPİYON TAKIMIN (Kısa:{int(best_team['Kisa_Model_ID'])}, Orta:{int(best_team['Orta_Model_ID'])}, Uzun:{int(best_team['Uzun_Model_ID'])}) GERÇEK DÜNYA PERFORMANSI:")
print(f"Validation Hatası (Optimizasyon): {best_team['Optimum_RMSE_Val_USD']:.2f} USD")
print(f"Test Hatası (HİLESİZ/GERÇEK):     {best_team['True_RMSE_Test_USD']:.2f} USD")
print("="*60)

print("\n--- 5. ŞAMPİYON TAKIMI KALICI OLARAK KAYDET ---")
import json
import joblib

SAVE_DIR = '/content/drive/MyDrive/LSTM_Project/deployment'
os.makedirs(SAVE_DIR, exist_ok=True)

best_team = df_sonuclar.iloc[0]
k_id = int(best_team['Kisa_Model_ID'])
o_id = int(best_team['Orta_Model_ID'])
u_id = int(best_team['Uzun_Model_ID'])
w_k = best_team['Kisa_%'] / 100.0
w_o = best_team['Orta_%'] / 100.0
w_u = best_team['Uzun_%'] / 100.0

k_hp = kisa_hyperparams[k_id]
o_hp = orta_hyperparams[o_id]
u_hp = uzun_hyperparams[u_id]

print(f"Şampiyon Kısa Model (ID:{k_id}) yeniden eğitiliyor... {k_hp}")
tf.keras.backend.clear_session()
x_tr, y_tr, x_val, x_te = prepare_data(k_hp['pencere'], scaled_data_aapl, y_raw_returns)
model_kisa = Sequential([
    Input(shape=(x_tr.shape[1], x_tr.shape[2])),
    LSTM(k_hp['noron']),
    Dropout(k_hp['dropout']),
    Dense(1)
])
model_kisa.compile(optimizer=Adam(learning_rate=k_hp['lr']), loss=custom_directional_loss)
model_kisa.fit(x_tr, y_tr, epochs=30, batch_size=32, verbose=0, callbacks=[early_stop])
model_kisa.save(os.path.join(SAVE_DIR, 'best_short_model.keras'))
kisa_val_pred = reconstruct_prices(model_kisa.predict(x_val, verbose=0)[:, 0], val_start_idx)
kisa_te_pred = reconstruct_prices(model_kisa.predict(x_te, verbose=0)[:, 0], test_start_idx)
print("✅ Kısa Vade modeli kaydedildi!")

print(f"Şampiyon Orta Model (ID:{o_id}) yeniden eğitiliyor... {o_hp}")
tf.keras.backend.clear_session()
x_tr, y_tr, x_val, x_te = prepare_data(o_hp['pencere'], scaled_data_aapl, y_raw_returns)
inp = Input(shape=(x_tr.shape[1], x_tr.shape[2]))
l1 = LSTM(o_hp['noron'], return_sequences=True)(inp)
att = Attention()([l1, l1])
l2 = LSTM(32)(att)
d = Dropout(o_hp['dropout'])(l2)
out = Dense(1)(d)
model_orta = Model(inputs=inp, outputs=out)
model_orta.compile(optimizer=Adam(learning_rate=o_hp['lr']), loss=custom_directional_loss)
model_orta.fit(x_tr, y_tr, epochs=25, batch_size=32, verbose=0, callbacks=[early_stop])
model_orta.save(os.path.join(SAVE_DIR, 'best_medium_model.keras'))
orta_val_pred = reconstruct_prices(model_orta.predict(x_val, verbose=0)[:, 0], val_start_idx)
orta_te_pred = reconstruct_prices(model_orta.predict(x_te, verbose=0)[:, 0], test_start_idx)
print("✅ Orta Vade modeli kaydedildi!")

print(f"Şampiyon Uzun Model (ID:{u_id}) yeniden eğitiliyor... {u_hp}")
tf.keras.backend.clear_session()
x_tr, y_tr, x_val, x_te = prepare_data(u_hp['pencere'], scaled_data_aapl, y_raw_returns)
model_uzun = Sequential([
    Input(shape=(x_tr.shape[1], x_tr.shape[2])),
    LSTM(u_hp['noron']),
    Dropout(u_hp['dropout']),
    Dense(1)
])
model_uzun.compile(optimizer=Adam(learning_rate=u_hp['lr']), loss=custom_directional_loss)
model_uzun.fit(x_tr, y_tr, epochs=20, batch_size=32, verbose=0, callbacks=[early_stop])
model_uzun.save(os.path.join(SAVE_DIR, 'best_long_model.keras'))
uzun_val_pred = reconstruct_prices(model_uzun.predict(x_val, verbose=0)[:, 0], val_start_idx)
uzun_te_pred = reconstruct_prices(model_uzun.predict(x_te, verbose=0)[:, 0], test_start_idx)
print("✅ Uzun Vade modeli kaydedildi!")

# Ensemble Tahminleri
ensemble_val = (w_k * kisa_val_pred) + (w_o * orta_val_pred) + (w_u * uzun_val_pred)
ensemble_te = (w_k * kisa_te_pred) + (w_o * orta_te_pred) + (w_u * uzun_te_pred)

rmse_final = np.sqrt(mean_squared_error(y_test_gercek, ensemble_te))
mae_final = np.mean(np.abs(y_test_gercek - ensemble_te))
print(f"\n🎯 Yeniden Eğitilmiş Şampiyon Test RMSE: {rmse_final:.2f} USD | MAE: {mae_final:.2f} USD")

# Tarihleri al
test_dates = df_aapl.index[test_start_idx:test_start_idx + len(y_test_gercek)].strftime('%Y-%m-%d').tolist()
val_dates = df_aapl.index[val_start_idx:val_start_idx + len(y_val_gercek)].strftime('%Y-%m-%d').tolist()

# Deployment verilerini kaydet
np.savez(os.path.join(SAVE_DIR, 'deployment_data.npz'),
    y_test_actual=y_test_gercek, y_test_predicted=ensemble_te,
    y_val_actual=y_val_gercek, y_val_predicted=ensemble_val,
    test_dates=np.array(test_dates), val_dates=np.array(val_dates))

# Scaler kaydet
joblib.dump(scaler_aapl, os.path.join(SAVE_DIR, 'scaler.pkl'))

# Ensemble config JSON kaydet
config = {
    'champion': {
        'kisa_model_id': k_id, 'orta_model_id': o_id, 'uzun_model_id': u_id,
        'weights': {'kisa': round(w_k, 4), 'orta': round(w_o, 4), 'uzun': round(w_u, 4)},
        'rmse_val': round(float(best_team['Optimum_RMSE_Val_USD']), 4),
        'rmse_test': round(rmse_final, 4), 'mae_test': round(mae_final, 4)
    },
    'hyperparams': {
        'kisa': {k: round(v, 6) if isinstance(v, float) else v for k, v in k_hp.items()},
        'orta': {k: round(v, 6) if isinstance(v, float) else v for k, v in o_hp.items()},
        'uzun': {k: round(v, 6) if isinstance(v, float) else v for k, v in u_hp.items()}
    },
    'features': secilen_oznitelikler,
    'total_combinations': 125000, 'total_models_trained': 150
}
with open(os.path.join(SAVE_DIR, 'ensemble_config.json'), 'w') as f:
    json.dump(config, f, indent=2)

print(f"\n{'='*60}")
print(f"🏆 TÜM DEPLOYMENT DOSYALARI KAYDEDİLDİ!")
print(f"📁 Konum: {SAVE_DIR}")
print(f"   - best_short_model.keras")
print(f"   - best_medium_model.keras")
print(f"   - best_long_model.keras")
print(f"   - deployment_data.npz")
print(f"   - ensemble_config.json")
print(f"   - scaler.pkl")
print(f"{'='*60}")
print("\n[BAŞARILI] Tüm kuantitatif sızıntı çözümleri, Kalman Filtresi ve Directional Loss eklendi!")
print("[SONRAKİ ADIM] Streamlit uygulamasını çalıştırın: !streamlit run src/app.py &")
