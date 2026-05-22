# 📈 LSTM Borsa Tahmin Motoru - V8 Akademik Proje Raporu

## 📌 Proje Özeti
Bu proje, Apple (AAPL) hisse senedi fiyatlarını analiz edip gelecekteki yönünü ve fiyatını tahmin eden yüksek performanslı, akademik standartlarda bir **Kuantitatif Finans Yapay Zeka (AI) Motoru** inşa etme sürecidir. 

Projenin temel amacı, klasik "al-tut" stratejilerini aşan, rastgele piyasa gürültüsünü filtreleyen ve piyasadaki panik/trend döngülerini okuyabilen elit bir ensemble (topluluk) modeli geliştirmekti. Proje, son V8 güncellemesiyle makroekonomik verileri de içine alarak tam bir endüstri standardına ulaşmıştır.

---

## 🛠️ Adım Adım Neler Yaptık? Duvarları Nasıl Yıktık?

### 🧱 Duvar 1: Klasik Algoritmaların Yetersizliği ve Baseline
- **Ne Yaptık?** AAPL hisse verilerini yükledik ve en temel algoritmaları (Linear Regression, SVR, XGBoost) test ettik.
- **Sorun Neydi?** Standart makine öğrenmesi algoritmaları borsadaki "zaman serisi" mantığını ve verilerin hafızasını kavramakta zorlanıyordu. Hata payı (RMSE) **15.30 USD** civarındaydı.

### 🧱 Duvar 2: Veri Zenginleştirme (Mikro) ve FFT
- **Çözüm:** Çıplak fiyat verisinin yanına piyasanın matematiğini (teknik göstergeleri) ekledik (RSI, ATR vb.). Ayrıca **FFT (Fast Fourier Transform)** dalga analizi ile fiyatın gizli frekans boyutunu çıkardık.

### 🧱 [YENİ] Duvar 2.5: Makroekonomik Sentez (Time Alignment)
- **Sorun:** Model dünyadan habersiz, sadece tekniğe bakarak işlem yapıyordu. Oysa Enflasyon ve FED Faiz kararları fiyatın yönünü belirleyen asıl temellerdir.
- **Çözüm:** `pandas_datareader` ile Amerikan Merkez Bankası veri tabanından (FRED) **TÜFE (Enflasyon)** ve **Faiz (FEDFUNDS)** verilerini otomatik çektik.
- **Frekans Hizalama Zekası:** Aylık açıklanan bu verileri, günlük hisse verisiyle sızıntı yapmadan birleştirmek için **Forward Fill (`ffill()`)** kullandık. (Veri açıklandığı gün sisteme düşer ve bir sonraki aya kadar geçerli kalır). Şampiyon takım artık makroekonomiyi okuyor!

### 🧱 Duvar 3: Piyasa Gürültüsü ve Hatalı Sinyaller
- **Mühendislik Harikası Çözümlerimiz:** 
  1. **Kalman Filtresi:** Fiyat verisindeki anlamsız günlük zikzaklar temizlenerek hissenin "gerçek durumu" ortaya çıkarıldı.
  2. **Custom Directional Loss (Yön Cezalandıran Kayıp Fonksiyonu):** Model sadece fiyatta yanıldığı için değil, **hisse düşerken çıkacak yönünde tahmin yaptığı için ekstra 2 kat ceza** yedi.

### 🧱 Duvar 4: Triple Ensemble ve 125.000 Kombinasyon
- **Sorun:** Veriyi hazırlarken Ölçekleme (Scaler) adımında gelecekteki test verisi eğitim verisine sızıyordu. (Data Leakage)
- **Çözüm:** 
  - Strict (Katı) Train-Validation-Test ayrımı yapıldı. 
  - 150 farklı rastgele uzman model eğitildi. 125.000 farklı takım kombinasyonu oluşturuldu.
  - "Bu takımlardaki Kısa, Orta, Uzun modeller hangi ağırlıkla birleşmeli?" sorusu SciPy (Altın Oran) ile matematiksel optimizasyona tabi tutuldu.
- **Sonuç:** Gerçek ve hilesiz test verisinde **1.77 USD Hata** oranıyla şampiyon takım bulundu.

### 👑 Final Aşama: Streamlit Dashboard (V8)
Sistemi akademik bir son ürüne dönüştüren arayüz modülleri eklendi:

1. **📊 EDA (Keşifçi Veri Analizi) Sekmesi:** Modelin veriyi nasıl gördüğünün ispatı (Getiri Histogramları, Fat-Tail Analizi ve Makro/Mikro Korelasyon Matrisi).
2. **🧠 Şeffaflık - XAI (SHAP) Sekmesi:** Yapay zekanın "Kara Kutu" (Black Box) olduğu önyargısını kıran karar açıklayıcı sekme. Son tahminin arkasındaki itici güçlerin (Enflasyon %30 etkiledi, RSI %40 etkiledi gibi) şelale grafiği (Waterfall) ile kanıtlanması.
3. **💰 Gerçekçi ROI Backtest (Slippage + Komisyon):** Önceden kusursuz piyasada çalışan sistem, artık **Slippage (Fiyat Kayması)** ve **İşlem Komisyonu** gibi gerçek dünyanın acımasız maliyetlerini de dahil ederek "Gerçekte ne kazanırdım?" sorusuna kesin cevap veriyor.

---
Masaüstünüzdeki güncel kodlarla harika bir Akademik Borsa Tahmin Motoru yaratıldı! 🚀
