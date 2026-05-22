# ==========================================
# AŞAMA 2: 3 FARKLI BASELINE MODEL KARŞILAŞTIRMASI VE DEĞERLENDİRME
# ==========================================
# DİKKAT: Bu hücrenin çalışması için 1. Hücrenin başarıyla tamamlanmış olması gerekir.
# Bellekte kalan veriler (scaled_data, x_train, scaler vb.) doğrudan kullanılacaktır.
# Eğer XGBoost hata verirse yeni bir hücrede: !pip install xgboost

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

print("--- 1. TEST VERİSİNİN HAZIRLANMASI ---")
# 1. Aşamadaki 'scaled_data' ve 'training_data_len' değişkenlerini kullanıyoruz.
# Verinin eğitimden geriye kalan kısmını (%20) Test Verisi yapıyoruz.
test_data = scaled_data[training_data_len - PENCERE_BOYUTU:, :]

x_test = []
y_test_gercek = data[training_data_len:, close_idx] # Ölçeklenmemiş orijinal gerçek kapanış fiyatları

for i in range(PENCERE_BOYUTU, len(test_data)):
    x_test.append(test_data[i-PENCERE_BOYUTU:i, :])

x_test = np.array(x_test)
print(f"Test Verisi Boyutu (LSTM için 3D): {x_test.shape}")

# KURAL: Makine öğrenmesi (Baseline) modelleri 3D (Zaman Adımı) veriyi desteklemez. 
# Bu yüzden LSTM'e verilen 60 günlük pencereyi 2D düz bir vektöre (Flatten) çeviriyoruz.
# Örn: (Örnek Sayısı, 60, 10) -> (Örnek Sayısı, 600)
x_train_2d = x_train.reshape(x_train.shape[0], -1)
x_test_2d = x_test.reshape(x_test.shape[0], -1)
print(f"Baseline (Geleneksel) Modeller için Eğitim Boyutu (2D): {x_train_2d.shape}")

print("\n--- 2. BASELINE MODELLERİN EĞİTİLMESİ ---")

# A. Linear Regression (Doğrusal Regresyon)
print("1/3 -> Linear Regression eğitiliyor...")
lr_model = LinearRegression()
lr_model.fit(x_train_2d, y_train)

# B. Support Vector Regression (SVR)
print("2/3 -> SVR (Support Vector Regression) eğitiliyor...")
svr_model = SVR(kernel='rbf', C=100, gamma=0.1)
svr_model.fit(x_train_2d, y_train)

# C. XGBoost Regressor (Ağaç Tabanlı Gradyan İnişi)
print("3/3 -> XGBoost eğitiliyor...")
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, objective='reg:squarederror')
xgb_model.fit(x_train_2d, y_train)

print("\n--- 3. TAHMİNLER VE TERS ÖLÇEKLENDİRME (INVERSE SCALING) ---")
# Tahminleri alıyoruz (Çıkan sonuçlar 0-1 arasında ölçeklidir)
lr_preds = lr_model.predict(x_test_2d)
svr_preds = svr_model.predict(x_test_2d)
xgb_preds = xgb_model.predict(x_test_2d)

# 1. Hücrede eğittiğimiz LSTM modeli ile tahmin yapıyoruz
lstm_preds = model.predict(x_test) 

# Bütün tahminleri tek bir matriste toplayıp orijinal fiyatlara geri çeviren (Ters Ölçekleme) Fonksiyon
def inverse_scale_predictions(preds, scaler, close_idx, feature_count):
    preds = preds.reshape(-1, 1)
    dummy_mat = np.zeros((len(preds), feature_count))
    dummy_mat[:, close_idx] = preds[:, 0]
    return scaler.inverse_transform(dummy_mat)[:, close_idx]

feature_count = len(available_features)
lr_preds = inverse_scale_predictions(lr_preds, scaler, close_idx, feature_count)
svr_preds = inverse_scale_predictions(svr_preds, scaler, close_idx, feature_count)
xgb_preds = inverse_scale_predictions(xgb_preds, scaler, close_idx, feature_count)
lstm_preds = inverse_scale_predictions(lstm_preds, scaler, close_idx, feature_count)

print("\n--- 4. SONUÇLAR VE METRİKLER (RMSE / MAE) ---")
def print_metrics(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name:>22} | RMSE: {rmse:7.2f} | MAE: {mae:7.2f}")

print("-" * 50)
print(f"{'MODEL':>22} | {'RMSE':>7} | {'MAE':>7}")
print("-" * 50)
print_metrics("Linear Regression", y_test_gercek, lr_preds)
print_metrics("SVR", y_test_gercek, svr_preds)
print_metrics("XGBoost", y_test_gercek, xgb_preds)
print_metrics("LSTM (Huber Loss - v4)", y_test_gercek, lstm_preds)
print("-" * 50)

print("\n--- 5. KARŞILAŞTIRMALI GRAFİK ÇİZİLİYOR ---")
plt.figure(figsize=(16,8))
plt.title('Baseline Modeller vs LSTM (Huber Loss) Fiyat Tahmini (Test Verisi)', fontsize=16)

# Grafik karmaşık olmasın diye sadece son 150 günü çizdiriyoruz
son_x_gun = 150
plt.plot(y_test_gercek[-son_x_gun:], label='Gerçek S&P 500 (SPY)', color='black', linewidth=3)
plt.plot(lr_preds[-son_x_gun:], label='Linear Regression', alpha=0.5, linestyle='--')
plt.plot(svr_preds[-son_x_gun:], label='SVR', alpha=0.7)
plt.plot(xgb_preds[-son_x_gun:], label='XGBoost', alpha=0.7)
plt.plot(lstm_preds[-son_x_gun:], label='LSTM (Bizim Modelimiz)', color='red', linewidth=2)

plt.xlabel('Zaman (Gün)', fontsize=12)
plt.ylabel('Fiyat (USD)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

print("\n[BAŞARILI] Aşama 2: Model Kıyaslamaları tamamlandı. Lütfen metrikleri ve grafiği inceleyin.")
