# ==========================================
# AŞAMA 3 (TAM OTONOM ŞAMPİYON MODEL): DİNAMİK PENCERE, OPTUNA VE FÜZYON
# ==========================================
# DİKKAT: Artık 10, 30, 60 gün gibi değerler yok.
# Yapay zeka en optimum gün sayısını (Pencereyi) kendisi bulur!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import optuna
import pandas_ta as ta
import warnings
from scipy.optimize import minimize
warnings.filterwarnings('ignore')

print("--- 1. VERİ YÜKLEME VE ZENGİNLEŞTİRME ---")
VERI_YOLU_AAPL = '/content/drive/MyDrive/Data/Stocks/aapl.us.txt'

try:
    df_aapl = pd.read_csv(VERI_YOLU_AAPL)
except FileNotFoundError:
    print(f"HATA: {VERI_YOLU_AAPL} bulunamadı!")
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

# Genel Test Verisi Boyutu (Son 150 Günlük bir test süreci tutalım ki tahminler hizalanabilsin)
# Tüm modeller son 150 günü test edecek.
TEST_SIZE = 150 
training_data_len = len(data_aapl) - TEST_SIZE
y_raw_returns = df_aapl['Log_Return'].values[:training_data_len]

# Dinamik Veri Üretici
def prepare_data(PENCERE, data_scaled, raw_returns=None, is_short=False):
    x_tr, y_tr = [], []
    # Eğitim Kısmı
    train_scaled = data_scaled[:training_data_len]
    for i in range(PENCERE, len(train_scaled)):
        x_tr.append(train_scaled[i-PENCERE:i, :])
        if is_short:
            y_tr.append(raw_returns[i])
        else:
            y_tr.append(train_scaled[i, close_idx])
            
    # Test Kısmı (Test süreci son TEST_SIZE kadar gün)
    x_te = []
    # Test verisi için geçmiş PENCERE kadar dataya ihtiyaç var
    test_start_idx = training_data_len - PENCERE
    test_scaled = data_scaled[test_start_idx:]
    for i in range(PENCERE, len(test_scaled)):
        x_te.append(test_scaled[i-PENCERE:i, :])
        
    return np.array(x_tr), np.array(y_tr), np.array(x_te)

y_test_gercek = data_aapl[training_data_len:, close_idx]

print("\n--- 2. OPTUNA İLE OTONOM PERİYOT (PENCERE) VE HİPERPARAMETRE AVI ---")
optuna.logging.set_verbosity(optuna.logging.WARNING) 

def create_objective(min_pencere, max_pencere, is_short, loss_type):
    def objective(trial):
        pencere = trial.suggest_int('pencere', min_pencere, max_pencere)
        lstm_units = trial.suggest_categorical('lstm_units', [32, 64])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
        lr = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
        
        # Otonom Periyota göre Dinamik Veri Üretimi
        x_tr, y_tr, _ = prepare_data(pencere, scaled_data_aapl, y_raw_returns, is_short=is_short)
        
        tscv = TimeSeriesSplit(n_splits=2)
        fold_errors = []
        
        # Zaman tasarrufu için son 800 gün üzerinde TSCV yapıyoruz
        tr_len = len(x_tr)
        start_idx = max(0, tr_len - 800)
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
            m.fit(X_t, y_t, epochs=5, verbose=0, batch_size=32)
            
            preds = m.predict(X_v, verbose=0)
            fold_errors.append(mean_squared_error(y_v, preds))
            
        return np.mean(fold_errors)
    return objective

print("1/3 -> Kısa Vade (3-14 Gün) Otonom Olarak Taranıyor...")
study_kisa = optuna.create_study(direction='minimize')
study_kisa.optimize(create_objective(3, 14, True, 'mse'), n_trials=10)
bp_kisa = study_kisa.best_params
print(f"✅ BULUNAN KISA VADE PERİYODU: {bp_kisa['pencere']} Gün")

print("2/3 -> Orta Vade (15-40 Gün) Otonom Olarak Taranıyor...")
study_orta = optuna.create_study(direction='minimize')
study_orta.optimize(create_objective(15, 40, False, 'huber'), n_trials=5)
bp_orta = study_orta.best_params
print(f"✅ BULUNAN ORTA VADE PERİYODU: {bp_orta['pencere']} Gün")

print("3/3 -> Uzun Vade (41-90 Gün) Otonom Olarak Taranıyor...")
study_uzun = optuna.create_study(direction='minimize')
study_uzun.optimize(create_objective(41, 90, False, 'huber'), n_trials=5)
bp_uzun = study_uzun.best_params
print(f"✅ BULUNAN UZUN VADE PERİYODU: {bp_uzun['pencere']} Gün")


print("\n--- 3. OTONOM BULUNAN PERİYOTLARLA NİHAİ MODELLERİN EĞİTİMİ ---")

# 1. KISA MODEL
x_tr_k, y_tr_k, x_te_k = prepare_data(bp_kisa['pencere'], scaled_data_aapl, y_raw_returns, is_short=True)
kisa_model = Sequential([
    Input(shape=(x_tr_k.shape[1], x_tr_k.shape[2])),
    LSTM(bp_kisa['lstm_units']),
    Dropout(bp_kisa['dropout_rate']),
    Dense(1)
])
kisa_model.compile(optimizer=Adam(learning_rate=bp_kisa['learning_rate']), loss='mse')
kisa_model.fit(x_tr_k, y_tr_k, batch_size=32, epochs=30, verbose=0)
preds_returns = kisa_model.predict(x_te_k, verbose=0)[:, 0]
preds_kisa_fiyat = []
for i in range(len(preds_returns)):
    dun_fiyati = data_aapl[training_data_len + i - 1, close_idx] 
    preds_kisa_fiyat.append(dun_fiyati * np.exp(preds_returns[i]))
preds_kisa_fiyat = np.array(preds_kisa_fiyat)

# 2. ORTA MODEL (ATTENTION EKLİ)
x_tr_o, y_tr_o, x_te_o = prepare_data(bp_orta['pencere'], scaled_data_aapl, is_short=False)
inputs_orta = Input(shape=(x_tr_o.shape[1], x_tr_o.shape[2]))
lstm_out1 = LSTM(bp_orta['lstm_units'], return_sequences=True)(inputs_orta)
attention_out = Attention()([lstm_out1, lstm_out1])
lstm_out2 = LSTM(32, return_sequences=False)(attention_out)
dropout_orta = Dropout(bp_orta['dropout_rate'])(lstm_out2)
outputs_orta = Dense(1)(dropout_orta)
orta_model = Model(inputs=inputs_orta, outputs=outputs_orta)
orta_model.compile(optimizer=Adam(learning_rate=bp_orta['learning_rate']), loss='huber')
orta_model.fit(x_tr_o, y_tr_o, batch_size=32, epochs=25, verbose=0)
preds_orta_scaled = orta_model.predict(x_te_o, verbose=0)
dummy = np.zeros((len(preds_orta_scaled), len(secilen_oznitelikler)))
dummy[:, close_idx] = preds_orta_scaled[:, 0]
preds_orta_fiyat = scaler_aapl.inverse_transform(dummy)[:, close_idx]

# 3. UZUN MODEL
x_tr_u, y_tr_u, x_te_u = prepare_data(bp_uzun['pencere'], scaled_data_aapl, is_short=False)
uzun_model = Sequential([
    Input(shape=(x_tr_u.shape[1], x_tr_u.shape[2])),
    LSTM(bp_uzun['lstm_units']),
    Dropout(bp_uzun['dropout_rate']),
    Dense(1)
])
uzun_model.compile(optimizer=Adam(learning_rate=bp_uzun['learning_rate']), loss='huber')
uzun_model.fit(x_tr_u, y_tr_u, batch_size=32, epochs=20, verbose=0)
preds_uzun_scaled = uzun_model.predict(x_te_u, verbose=0)
dummy[:, close_idx] = preds_uzun_scaled[:, 0]
preds_uzun_fiyat = scaler_aapl.inverse_transform(dummy)[:, close_idx]

print("\n--- 4. SCIPY MATEMATİKSEL FÜZYON OPTİMİZASYONU ---")

def ensemble_loss(weights):
    w_k, w_o, w_u = weights
    blended = (w_k * preds_kisa_fiyat) + (w_o * preds_orta_fiyat) + (w_u * preds_uzun_fiyat)
    return np.sqrt(mean_squared_error(y_test_gercek, blended))

init_weights = [0.33, 0.33, 0.34] 
bounds = ((0, 1), (0, 1), (0, 1)) 
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1}) 

print("SciPy SLSQP algoritması en düşük hatayı veren yüzdelik oranları arıyor...")
res = minimize(ensemble_loss, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
w_k_opt, w_o_opt, w_u_opt = res.x

print(f"✅ BULUNAN ALTIN ORANLAR: Kısa(%{w_k_opt*100:.1f}), Orta(%{w_o_opt*100:.1f}), Uzun(%{w_u_opt*100:.1f})")

ensemble_preds = (w_k_opt * preds_kisa_fiyat) + (w_o_opt * preds_orta_fiyat) + (w_u_opt * preds_uzun_fiyat)

print("\n--- 5. AKADEMİK NİHAİ METRİKLER VE GRAFİK ---")
rmse = np.sqrt(mean_squared_error(y_test_gercek, ensemble_preds))
mae = mean_absolute_error(y_test_gercek, ensemble_preds)

print("-" * 50)
print(f"TAM OTONOM TRIPLE ENSEMBLE SONUÇLARI")
print(f"Ensemble RMSE: {rmse:.2f} USD")
print(f"Ensemble MAE:  {mae:.2f} USD")
print("-" * 50)

plt.figure(figsize=(16,8))
plt.title('Tam Otonom Yapay Zeka: Optuna Tarafından Bulunan Periyotlarla Tahmin', fontsize=16)

plt.plot(y_test_gercek, label='Gerçek AAPL Fiyatı', color='black', linewidth=2)
plt.plot(ensemble_preds, label='OTONOM FÜZYON TAHMİNİ', color='darkmagenta', linewidth=2.5)

plt.xlabel('Zaman (Son 150 Gün)', fontsize=12)
plt.ylabel('Fiyat (USD)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
