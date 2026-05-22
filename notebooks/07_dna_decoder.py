# ==========================================
# ŞAMPİYON MODELLERİN DNA'SINI (PARAMETRELERİNİ) ÇÖZEN KOD
# ==========================================
# Bu kodu Colab'da yeni bir hücrede çalıştırın. Eğitim GEREKTİRMEZ! 
# 07 koduyla aynı Random Seed (42) kullanıldığı için, geçmişi %100 doğrulukla çözer.

import random
import pandas as pd

random.seed(42)
noron_havuzu = [16, 32, 64]

kisa_dna, orta_dna, uzun_dna = [], [], []

for i in range(50):
    kisa_dna.append({
        'Model_ID': i, 'Tip': 'Kısa (Panik)', 'Pencere_Gun': random.randint(3,15),
        'Noron': random.choice(noron_havuzu), 'Learning_Rate': round(random.uniform(0.001,0.005), 5)
    })

for i in range(50):
    orta_dna.append({
        'Model_ID': i, 'Tip': 'Orta (Attention)', 'Pencere_Gun': random.randint(16,45),
        'Noron': random.choice(noron_havuzu), 'Learning_Rate': round(random.uniform(0.0005,0.002), 5)
    })

for i in range(50):
    uzun_dna.append({
        'Model_ID': i, 'Tip': 'Uzun (Trend)', 'Pencere_Gun': random.randint(46,90),
        'Noron': random.choice(noron_havuzu), 'Learning_Rate': round(random.uniform(0.0001,0.001), 5)
    })

df_dna = pd.DataFrame(kisa_dna + orta_dna + uzun_dna)
df_dna.to_csv('150_Modelin_DNASI.csv', index=False)

print("🏆 ŞAMPİYON TAKIMIN DNA'SI (Parametreleri) 🏆\n")
print(f"KISA VADE (ID 36): {kisa_dna[36]['Pencere_Gun']} Günlük Veriye Bakıyor, {kisa_dna[36]['Noron']} Nöronu Var. (LR: {kisa_dna[36]['Learning_Rate']})")
print(f"ORTA VADE (ID 2) : {orta_dna[2]['Pencere_Gun']} Günlük Veriye Bakıyor, {orta_dna[2]['Noron']} Nöronu Var. (LR: {orta_dna[2]['Learning_Rate']})")
print(f"UZUN VADE (ID 44): {uzun_dna[44]['Pencere_Gun']} Günlük Veriye Bakıyor, {uzun_dna[44]['Noron']} Nöronu Var. (LR: {uzun_dna[44]['Learning_Rate']})")

print("\n(Tüm 150 modelin parametreleri '150_Modelin_DNASI.csv' dosyasına kaydedildi!)")
