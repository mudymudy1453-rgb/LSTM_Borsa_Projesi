# ==========================================
# HAFTA 7: DEĞERLENDİRME, METRİKLER VE GRAFİK
# ==========================================

# 1. TEST VERİSİNİN HAZIRLANMASI
# Eğitim verisinin bittiği yerden sonrasını Test verisi olarak alıyoruz.
# Ancak modelin 60 günlük pencereye ihtiyacı olduğu için, test setinin başlangıcından 60 gün geriye gidiyoruz.
test_data = scaled_data[training_data_len - PENCERE_BOYUTU: , :]

x_test = []
y_test_gercek = data[training_data_len:, :] # Ölçeklenmemiş Orijinal Gerçek Fiyatlar

for i in range(PENCERE_BOYUTU, len(test_data)):
    x_test.append(test_data[i-PENCERE_BOYUTU:i, 0])

x_test = np.array(x_test)

# LSTM için 3 Boyutlu Reshape
x_test_lstm = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# ==========================================
# 2. TAHMİNLERİN YAPILMASI
# ==========================================
print("Tahminler (Predictions) oluşturuluyor...")

# A. Baseline (Linear Regression) Tahmini
lr_predictions = lr_model.predict(x_test)
# Scikit-learn scaler'ı 2D matris bekler, lr_model ise 1D (düz liste) sonuç verir. 
# Bu yüzden (Reshape) ile onu [N, 1] boyutuna çeviriyoruz:
lr_predictions = lr_predictions.reshape(-1, 1)
# Linear Regression sonuçlarını orijinal fiyat aralığına geri döndürüyoruz (Inverse Transform)
lr_predictions = scaler.inverse_transform(lr_predictions)

# B. LSTM Tahmini
lstm_predictions = lstm_model.predict(x_test_lstm)
# LSTM sonuçlarını orijinal fiyat aralığına geri döndürüyoruz
lstm_predictions = scaler.inverse_transform(lstm_predictions)

# ==========================================
# 3. HATA METRİKLERİNİN HESAPLANMASI (RMSE & MAE)
# ==========================================
# Kılavuzun zorunlu tuttuğu akademik başarı kriterleri
print("\n--- MODEL BAŞARI SONUÇLARI (Ne kadar düşük, o kadar iyi) ---")

# Baseline (LR) Metrikleri
lr_rmse = np.sqrt(mean_squared_error(y_test_gercek, lr_predictions))
lr_mae = mean_absolute_error(y_test_gercek, lr_predictions)
print(f"Linear Regression RMSE: {lr_rmse:.2f}")
print(f"Linear Regression MAE:  {lr_mae:.2f}")

# LSTM Metrikleri
lstm_rmse = np.sqrt(mean_squared_error(y_test_gercek, lstm_predictions))
lstm_mae = mean_absolute_error(y_test_gercek, lstm_predictions)
print(f"\nLSTM RMSE: {lstm_rmse:.2f}")
print(f"LSTM MAE:  {lstm_mae:.2f}")

# Modelimizi Kaydetme (Gelecek hafta Streamlit'te canlıya almak için çok önemli!)
import os
KAYIT_YERI = '/content/drive/MyDrive/Data/lstm_spy_model.keras'
lstm_model.save(KAYIT_YERI)
print(f"\n[BAŞARILI] LSTM Modeli Drive'a kaydedildi: {KAYIT_YERI}")

# ==========================================
# 4. GÖRSELLEŞTİRME (RAPOR İÇİN GRAFİK)
# ==========================================
# Eğitim, Gerçek Test Fiyatları ve Model Tahminlerini tek grafikte çizdiriyoruz
train = df[:training_data_len]
valid = df[training_data_len:].copy() # Hata almamak için copy() kullanıyoruz
valid['Baseline_Predictions'] = lr_predictions
valid['LSTM_Predictions'] = lstm_predictions

plt.figure(figsize=(16,8))
plt.title('S&P 500 (SPY) Hisse Fiyatı Tahmin Modeli', fontsize=18, fontweight='bold')
plt.xlabel('Zaman Adımı (Gün)', fontsize=14)
plt.ylabel('Kapanış Fiyatı (USD)', fontsize=14)

# Verileri çizdir
plt.plot(train['Close'], color='blue', label='Eğitim Verisi (Train)')
plt.plot(valid['Close'], color='black', label='Gerçek Fiyat (Test)', linewidth=2)
plt.plot(valid['LSTM_Predictions'], color='green', label='LSTM Tahmini', linewidth=2)
plt.plot(valid['Baseline_Predictions'], color='red', label='Linear Reg. (Baseline) Tahmini', linestyle='dashed')

plt.legend(loc='lower right', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()
