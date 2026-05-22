# ==========================================
# AŞAMA 4: AÇIKLANABİLİR YAPAY ZEKA (SHAP İLE KARA KUTU KIRMA)
# ==========================================
# DİKKAT: Bu hücrenin çalışması için shap kütüphanesi gereklidir.
# Eğer yüklü değilse, en üstte yeni bir hücre açıp şunu çalıştırın:
# !pip install shap

import shap
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

print("--- 1. SHAP İÇİN MODEL VE VERİ HAZIRLIĞI ---")
# 3. Aşamada eğittiğimiz AAPL modelini kullanacağız
MODEL_YOLU = '/content/drive/MyDrive/Data/lstm_aapl_finetuned.keras'
model_xai = load_model(MODEL_YOLU)

# SHAP hesaplaması derin öğrenme modellerinde çok ağır olduğu için
# Eğitim verisinden (background) rastgele 100 örnek seçiyoruz.
np.random.seed(42)
background_indices = np.random.choice(x_train_aapl.shape[0], 100, replace=False)
background_data = x_train_aapl[background_indices]

# Test verisinden de açıklaması yapılacak 20 örnek seçiyoruz
test_indices = np.random.choice(x_test_aapl.shape[0], 20, replace=False)
test_data_sample = x_test_aapl[test_indices]

print("\n--- 2. SHAP DEĞERLERİNİN HESAPLANMASI (GRADIENT EXPLAINER) ---")
print("SHAP hesaplaması Keras LSTM modeli için biraz zaman alabilir, lütfen bekleyin...")

# LSTM modelleri için matematikteki en uygun açıklayıcı olan GradientExplainer kullanıyoruz
explainer = shap.GradientExplainer(model_xai, background_data)
shap_values = explainer.shap_values(test_data_sample)

# Keras modellerinde shap_values bazen liste şeklinde dönebilir
if isinstance(shap_values, list):
    shap_values = shap_values[0]

# shap_values boyutu normalde: (20 örnek, 60 gün, 10 özellik) olacaktır
print(f"SHAP Değerleri Başarıyla Hesaplandı. Matris Boyutu: {shap_values.shape}")

print("\n--- 3. ÖZNİTELİK (FEATURE) ETKİLERİNİN GÖRSELLEŞTİRİLMESİ ---")

# SHAP değerlerini 3D'den 2D'ye (Zaman boyutunu ezerek) indirgiyoruz ki klasik grafikleri çizebilelim
# Yani her bir günü ayrı bir örneklem gibi değerlendireceğiz: (20*60, 10)
shap_values_2d = shap_values.reshape(-1, shap_values.shape[2])
test_data_sample_2d = test_data_sample.reshape(-1, test_data_sample.shape[2])

# İlgili özellik isimlerini (Aşama 1'den hatırlayarak) yazıyoruz
feature_names = ['Close', 'Log_Return', 'Lag_1', 'Lag_3', 'Lag_7', 'FFT_Absolute', 
                 'SMA_14', 'EMA_14', 'RSI_14', 'ATRr_14']

plt.figure(figsize=(12, 8))
plt.title("Hisse Fiyatı Tahmininde Özelliklerin (Features) Etki Ağırlığı", fontsize=14)

# 1. Grafik: Summary Plot (Nokta Dağılımı)
shap.summary_plot(shap_values_2d, test_data_sample_2d, feature_names=feature_names, show=False)
plt.tight_layout()
plt.show()

# 2. Grafik: Ortalama Mutlak Etki (Bar Plot - Hangi özellik daha önemli?)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_2d, test_data_sample_2d, feature_names=feature_names, plot_type="bar", show=False)
plt.title("Hangi Öznitelik Kararlarda Daha Baskın? (Mutlak SHAP Değerleri)", fontsize=14)
plt.tight_layout()
plt.show()

print("\n[BAŞARILI] Aşama 4: SHAP Açıklanabilirlik Raporu tamamlandı!")
print("Grafiklerde 'FFT' veya 'RSI' gibi teknik göstergelerin, modelin 'Karar Kapılarına' nasıl etki ettiğini görebilirsiniz.")
