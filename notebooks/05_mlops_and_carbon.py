# ==========================================
# AŞAMA 5: MLOps VE ÇEVRE FARKINDALIĞI (CodeCarbon & MLFlow)
# ==========================================

import mlflow
import mlflow.tensorflow
from codecarbon import EmissionsTracker
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta

print("--- 1. VERİ HAZIRLIĞI (Değerlendirme İçin) ---")
# Colab runtime'ı yeniden başladığında bellek silindiği için veriyi hızlıca tekrar çekiyoruz.
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

secilen_oznitelikler = ['Close', 'Log_Return', 'Lag_1', 'Lag_3', 'Lag_7', 'FFT_Absolute', 'SMA_14', 'EMA_14', 'RSI_14', 'ATRr_14']
data_aapl = df_aapl[secilen_oznitelikler].values

scaler_aapl = MinMaxScaler(feature_range=(0, 1))
scaled_data_aapl = scaler_aapl.fit_transform(data_aapl)

training_data_len_aapl = int(np.ceil(len(data_aapl) * .8))
test_data_aapl = scaled_data_aapl[training_data_len_aapl - 60:, :]

x_test_aapl = []
y_test_gercek_aapl = data_aapl[training_data_len_aapl:, 0] # Close

for i in range(60, len(test_data_aapl)):
    x_test_aapl.append(test_data_aapl[i-60:i, :])
x_test_aapl = np.array(x_test_aapl)


print("\n--- 2. MLOps: MLFLOW İLE DENEY TAKİBİ ---")
mlflow.tensorflow.autolog()

MODEL_YOLU = '/content/drive/MyDrive/Data/lstm_aapl_finetuned.keras'
model_mlops = load_model(MODEL_YOLU)

with mlflow.start_run(run_name="LSTM_AAPL_Evaluation") as run:
    print("MLflow kaydı başlatıldı. Model değerlendirmesi yapılıyor...")
    loss = model_mlops.evaluate(x_test_aapl, y_test_gercek_aapl, verbose=0)
    print(f"MLflow'a Kaydedilen Güncel Huber Loss Değeri: {loss}")
    print("Modelin hiperparametreleri (optimizer, katmanlar) MLflow tarafından arka planda otomatik loglandı.")

print("\n--- 3. ÇEVRE FARKINDALIĞI: CODECARBON İLE CO2 AYAK İZİ ÖLÇÜMÜ ---")
tracker = EmissionsTracker(project_name="LSTM_AAPL_Inference")
tracker.start()

print("Model tahminleri yapılırken tüketilen enerji ölçülüyor...")
# Ölçümü netleştirmek için tahmini 10 kez üst üste yapıyoruz
for _ in range(10):
    predictions = model_mlops.predict(x_test_aapl, verbose=0)

emissions = tracker.stop()

print("\n" + "=" * 50)
print(f"🌍 ÇEVRESEL ETKİ RAPORU (IEEE YZ ETİK STANDARDI)")
print("=" * 50)
print(f"Bu işlemin Karbon Ayak İzi: {emissions:.6f} kg CO2")
print("Not: Büyük çaplı eğitimlerde gereksiz epoch'lardan kaçınmak ve")
print("EarlyStopping kullanmak dünyamız için hayati önem taşır!")
print("=" * 50)
print("\n[BAŞARILI] Aşama 5 MLOps ve Etik kısmı başarıyla tamamlandı!")
print("Son adım olarak Streamlit Arayüzüne geçiyoruz.")
