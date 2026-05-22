# 📈 Deep Quant: Triple Ensemble LSTM Borsa Tahmin Motoru & Büyük Birleşik Optimizasyon

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%202.x-orange.svg)
![SciPy](https://img.shields.io/badge/Optimization-SciPy%20SLSQP-green.svg)
![RMSE](https://img.shields.io/badge/True%20Test%20RMSE-1.77%20USD-success.svg)

## 📌 Proje Hakkında
Bu proje, finansal piyasalardaki rastgele yürüyüş (Random Walk) ve yüksek frekanslı gürültü sorunlarını aşmak amacıyla Apple (AAPL) ve S&P 500 (SPY) verileri üzerinde test edilmiş, akademik standartlarda bir **Kuantitatif Finans Yapay Zeka Motoru** mimarisidir. 

Proje, tek bir derin öğrenme modeline güvenmek yerine; kısa, orta ve uzun vadeli piyasa dinamiklerine yalıtılmış uzman LSTM ağlarını paralel eğitip, bu ağların tahminlerini **SciPy SLSQP Altın Oran** optimizasyonuyla birleştiren bir **Triple Ensemble (Üçlü Topluluk)** yapısına sahiptir.

---

## 🏗️ Projenin Evrimsel Basamakları (Adım Adım Yolculuk)

Proje, doğrudan son ürünü yazmak yerine, makine öğrenmesi literatüründeki engelleri (duvarları) adım adım yıkarak bir Ar-Ge laboratuvarı titizliğiyle geliştirilmiştir:

1. **Baseline Kurulumu & Örütçülük:** İlk olarak en basit doğrusal modeller kurulmuş ve çıplak fiyat serilerinin zaman hafızasını kavrayamadığı ispatlanmıştır. Ardından ACF/PACF otokorelasyon analizleri yapılarak fiyata yön veren anlamlı geçmiş günler tespit edilmiştir.
2. **Sinyal İşleme ve Öznitelik Mühendisliği:** Veriye sadece teknik indikatörler değil, Hızlı Fourier Dönüşümü (FFT) gibi sinyal işleme algoritmaları eklenerek fiyatın frekans boyutundaki gizli dalgaları yakalanmıştır.
3. **Piyasa Gürültüsü ve Yön Kaybı Çözümü:** Günlük anlamsız zikzakları temizlemek için Kalman Filtresi entegre edilmiş; modelin sadece fiyatta yanılmasını değil, yönü yanlış tahmin etmesini (hisse düşerken çıkacak demesini) asimetrik olarak 2 kat ağır cezalandıran `Custom Directional Loss` fonksiyonu geliştirilmiştir.
4. **Veri Sızıntısı (Data Leakage) Engelleme:** Ölçekleme (MinMaxScaler) ve doğrulama süreçlerinde gelecekteki verilerin eğitim setine sızması (Scaler Leakage) tamamen engellenmiş, katı bir Train-Validation-Test izolasyonu kurulmuştur.
5. **Büyük Birleşik Optimizasyon (Grand Unified Matrix):** Optuna ile her bir uzman LSTM modelinin periyodu, nöron sayısı, öğrenme hızı ve dropout oranları otonom olarak optimize edilmiş; ardından 125.000 takım kombinasyonu taranarak en kararlı şampiyon model seçilmiştir.

---

## 📂 Dosya Rehberi & Mimari Dağılım

Repoda bulunan tüm araştırma (`notebooks_and_research/`) ve canlı sistem (`app/`) dosyalarının teknik görevleri ve projeye katkıları aşağıda detaylandırılmıştır:

### 🔬 Araştırma ve Geliştirme (Ar-Ge) Dosyaları:

* **`01_baseline_and_lstm.py`**
    * **Görevi:** Projenin en temel baseline (referans) noktasıdır. S&P 500 (SPY) kapanış fiyatlarını MinMaxScaler ile ölçekleyip 60 günlük pencereler halinde basit bir LSTM modeline (50 nöron, 0.2 Dropout) besler. 
    * **Amacı:** Derin öğrenmenin, geleneksel Linear Regression modellerine karşı ilksel üstünlüğünü ve hata marjlarını ölçmek için bir kıyas zeminidir.
* **`01_pattern_and_feature_engineering.py`**
    * **Görevi:** İstatistiki ve ekonometrik zenginleştirme aşamasıdır. `statsmodels` kütüphanesiyle ACF analizi yaparak fiyata en çok etki eden geçmiş günleri (Lag 1, 3, 7) belirler ve mevsimsel ayrıştırma (Decomposition) yapar. Getirileri durağanlaştırmak için Log Getiri (Log Return) hesaplar, gürültüyü süzmek için **FFT Absolute** dalga analizini ve `pandas_ta` ile teknik göstergeleri (SMA, EMA, RSI, Bollinger, ATR) öznitelik olarak ekler. Modeli borsa şoklarına dayanıklı **Huber Loss** ile eğitip kaydeder.
* **`02_baseline_comparison.py`**
    * **Görevi:** Geleneksel makine öğrenmesi algoritmaları ile derin öğrenmenin çok boyutlu kıyaslamasını yapar. Zaman serisi pencerelerini (3D Tensor) doğrusal modellerin anlayabileceği 2D düz bir vektöre (Flatten) dönüştürür. **Linear Regression, Support Vector Regression (SVR-RBF) ve XGBoost Regressor** modellerini eğitir. Tüm tahminleri gerçeğe dönüştürerek (Inverse Scaling) RMSE ve MAE metrikleriyle LSTM'in başarısını matematiksel grafiklerle ilan eder.
* **`02_evaluation_and_metrics.py`**
    * **Görevi:** İlk aşamalarda üretilen temel modellerin doğruluğunu, test setinin başlangıcından 60 gün geriye giderek kesintisiz test eden, RMSE/MAE metrik hesaplama ve rapor grafikleri üreten temel doğrulama modülüdür.
* **`03_transfer_learning_aapl.py`**
    * **Görüvi:** Projenin Apple (AAPL) hissesine uyarlandığı ve ilk otonom ensemble denemesinin yapıldığı yerdir. Optuna kütüphanesini kullanarak Kısa Vade (3-14 gün), Orta Vade (15-40 gün) ve Uzun Vade (41-90 gün) için en ideal bakış pencerelerini (Lookback Periods) tırnak ucuyla arar. **Orta vade modeline Self-Attention (Dikkat) mekanizması** ekleyerek piyasa şoklarına odaklanmasını sağlar. Son aşamada 3 modelin tahminlerini **SciPy SLSQP** algoritmasıyla optimize ederek hata payını düşüren ilk "Altın Oran" ağırlıklarını bulur.
* **`04_bisection_window_search.py`**
    * **Görevi:** Modelin hiperparametrelerini sabitleyerek (yalıtarak) sadece "Kusursuz Bakış Penceresi" gün sayısını bulmaya odaklanan özel bir arama algoritmasıdır. İki uç yaklaşımı mantığıyla çalışan **Ternary Search** algoritmasını kullanarak, 4 ile 16 gün arasındaki aralığı matematiksel olarak daraltır ve hatanın türevinin (eğiminin) sıfır olduğu mutlak dip noktayı (en optimum gün sayısını) otonom keşfeder.
* **`04_xai_shap_lime.py`**
    * **Görevi:** Yapay zekanın "Kara Kutu" (Black Box) önyargısını kıran akademik açıklanabilirlik modülüdür. Eğitilmiş Keras LSTM modelinin karar kapılarını aralamak için **SHAP (GradientExplainer)** algoritmasını çalıştırır. 3 boyutlu zaman serisi etkilerini 2D düzleme indirgeyerek modelin kararlarında FFT, RSI veya geçmiş fiyatların (Lags) mutlak etki ağırlıklarını nokta dağılımı (Summary Plot) ve önem barları (Bar Plot) ile jüriye kanıtlar.
* **`05_grand_unified_optimization.py`**
    * **Görevi:** Projenin en ağır işçiliğini yapan, kısıtlamaların kaldırıldığı **Büyük Birleşik Optimizasyon** kodudur. Zaman serisi sızıntısını önleyen `TimeSeriesSplit` cross-validation altyapısıyla her 3 uzman model (Kısa, Orta-Attention, Uzun) için 50'şer adet olmak üzere toplamda 150 denemelik devasa bir Optuna araması gerçekleştirir. Nöron sayılarını (16-128 arası), öğrenme hızlarını ve dropout oranlarını aynı anda tarayıp kusursuz DNA'ya sahip şampiyon ensemble model kombinasyonunu belirler.

### 💻 Kullanıcı Arayüzü & Ürün Katmanı:

* **`app.py` (Streamlit Canlı Dashboard V8)**
    * **Görevi:** Ar-Ge aşamasında üretilen tüm ağır matematiksel modelleri son kullanıcının veya jürinin saniyeler içinde test edebileceği kurumsal bir simülatöre dönüştürür.
    * **İçerdiği Modüller:**
        1.  *Canlı Simülatör:* Gerçek fiyat ve Ensemble tahmin grafiklerini Plotly ile interaktif, cam tasarımlı (glassmorphism) bir arayüzde sunar.
        2.  *Akademik ROI Backtest:* Kusursuz piyasa illüzyonunu yıkarak, gerçek dünyanın acımasız **Slippage (Fiyat Kayması)** ve **İşlem Komisyonu (%0.1)** maliyetlerini hesaplamaya dahil eder; "Al ve Tut" stratejisine karşı üretilen Alpha (fazla kazanç) değerini simüle eder.
        3.  *EDA Sekmesi:* Getirilerin istatistiksel histogram dağılımlarını (Fat-Tail analizi) ve makro/mikro verilerin korelasyon ısı haritasını görselleştirir.
        4.  *SHAP XAI Butonu:* Canlı veride üretilen son tahminin arkasındaki itici güçlerin (Teknik indikatörler, Enflasyon verisi veya FFT) fiyata olan USD bazlı etkisini anlık şelale (Waterfall) grafiği ile açıklar.

### 📝 Akademik Dokümantasyon:

* **`PROJE_RAPORU.md`**
    * **Görevi:** Projenin 1. haftasından 12. haftasına kadar yapılan tüm mühendislik hamlelerini, yıkılan duvarları, makroekonomik sentez mantığını ve sızıntı çözümlerini barındıran, IEEE standartlarına uygun 15 sayfalık resmi proje raporudur.

---

## 🛠️ Kurulum ve Canlı Çalıştırma

Projeyi yerel bilgisayarınızda veya sunucunuzda çalıştırmak için aşağıdaki adımları takip edebilirsiniz:

```bash
# 1. Repoyu bilgisayarınıza indirin
git clone [https://github.com/KULLANICI_ADIN/LSTM-Borsa-Tahmin-Motoru.git](https://github.com/KULLANICI_ADIN/LSTM-Borsa-Tahmin-Motoru.git)
cd LSTM-Borsa-Tahmin-Motoru

# 2. Gerekli tüm kütüphaneleri indirin
pip install -r requirements.txt

# 3. Streamlit ROI Dashboard uygulamasını başlatın
streamlit run app/app.py
