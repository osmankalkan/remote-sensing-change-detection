# End-to-End Remote Sensing Change Detection (Uydu Görüntüleriyle Değişim Tespiti)

Bu proje, aynı coğrafi alanın iki farklı zamanda çekilmiş bitemporal (**T1** ve **T2**) uydu görüntülerini karşılaştırarak kentsel alan ve bina değişimlerini piksel düzeyinde ikili maske (`0: Değişim Yok`, `1: Değişim Var`) olarak tespit eden uçtan uca derin öğrenme boru hattıdır.

---

## 🚀 Proje Mimarisi ve Teknik Tercihler

### 1. Modeller
- **Baseline Model (Early Fusion):**
  - T1 ve T2 görüntülerinin kanal bazında birleştirilmesi (`[B, 6, H, W]`).
  - 6-kanallı girdi alan `ResNet18` omurgalı U-Net (`segmentation_models_pytorch`).
- **Hedef Model (Siamese U-Net / Late Fusion):**
  - **Twin Encoder:** T1 ve T2 bağımsız olarak ağırlık paylaşan (`shared-weights`) `ResNet34`/`ResNet50` encoder'dan geçer.
  - **Feature Fusion:** Multi-scale skip connection seviyelerinde mutlak fark $|f_1 - f_2|$ ve kanal birleşimi $[f_1; f_2; |f_1 - f_2|]$.
  - **CBAM Attention:** Mevsimsel ve radyometrik parazitleri filtrelemek için Convolutional Block Attention Module.

### 2. Donanım ve Ortam Stratejisi
- **Yerel (MacBook Air M2):** Apple Silicon Metal Performance Shaders (`torch.device("mps")`) üzerinde duman testleri (smoke testing), veri ön işleme/yamalama (CPU + SSD optimizasyonu) ve web arayüzü demosu.
- **Bulut (Colab / Kaggle):** 30-50 epoch'luk tam ölçekli model eğitimleri (CUDA, Mixed Precision `torch.amp`).

### 3. Kayıp Fonksiyonu ve Metrikler
- **Sınıf Dengesizliği Çözümü:** Değişen pikseller tüm görüntünün yalnızca %3-5'ini oluşturduğundan $\mathcal{L}_{Total} = 0.5 \cdot \mathcal{L}_{BCE} + 0.5 \cdot \mathcal{L}_{Dice}$ (veya Focal + Dice) kullanılmaktadır.
- **Metrikler:** Precision, Recall, F1-Score ve Change-IoU.

---

## 📁 Dizin Yapısı

```text
remote-sensing-change-detection/
├── configs/
│   └── config.yaml              # Hiperparametreler, dosya yolları, batch size
├── data/
│   ├── raw/                     # LEVIR-CD ham veri (1024x1024)
│   └── processed/               # 256x256 yamalanmış veri (train/val/test)
│       ├── train/ {A, B, label}
│       ├── val/   {A, B, label}
│       └── test/  {A, B, label}
├── notebooks/
│   ├── 01_eda.ipynb             # Keşifsel veri analizi
│   └── 02_error_analysis.ipynb  # Hata analizi ve görselleştirme
├── scripts/
│   ├── patch_images.py          # 1024x1024 -> 256x256 kesme scripti
│   ├── prepare_sample_data.py   # Test amaçlı örnek sentetik veri üretici
│   ├── verify_dataset.py        # Albumentations augmentasyon doğrulama scripti
│   ├── smoke_test_mps.py        # 1-batch MPS forward/backward duman testi
│   └── evaluate_large.py        # 1024x1024 dikişsiz kayar pencere testi
├── src/
│   ├── __init__.py
│   ├── datasets.py              # DualAugmentationDataset & DataLoader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_unet.py     # 6-kanallı early fusion U-Net
│   │   ├── siamese_unet.py      # Dual-branch Siamese U-Net
│   │   └── attention.py         # CBAM / Attention blokları
│   ├── losses.py                # BCE, Dice, Focal, Compound Loss
│   ├── metrics.py               # IoU, F1-score hesaplayıcıları
│   ├── utils.py                 # Cihaz seçimi, görselleştirme araçları
│   └── train.py                 # Eğitim ve doğrulama döngüsü
├── app.py                       # Gradio interaktif web demosu
├── requirements.txt
└── README.md
```

---

## 🛠️ Kurulum ve Hızlı Başlangıç

### 1. Gün: Ortam Kurulumu
```bash
# Sanal ortamı aktifleştirin
source .venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 2. Gün: Veri Seti ve 256x256 Yamalama (Patching)
Hızlı duman testleri için sentetik bitemporal uydu çiftleri üretebilir veya LEVIR-CD ham verisini `data/raw` altına yerleştirebilirsiniz:

```bash
# 1. Test verisi oluştur (veya gerçek LEVIR-CD'yi data/raw'a koy)
python scripts/prepare_sample_data.py

# 2. 1024x1024 görüntüleri 256x256 yamalara böl
python scripts/patch_images.py --raw_dir data/raw --processed_dir data/processed --patch_size 256 --stride 256
```

### 3. Gün: Çiftli Veri Yükleyici ve Augmentasyon Doğrulaması
Geometrik dönüşümlerin (Flip, Rotate90) T1, T2 ve Maske arasında birebir senkronize, renk/ışık dönüşümlerinin ise bağımsız çalıştığını görsel olarak doğrulamak için:

```bash
python scripts/verify_dataset.py
```
*Oluşturulan görsel `visualizations/augmentation_verification.png` yoluna kaydedilir.*

### 4. Gün: Early Fusion Baseline & Apple Silicon MPS Duman Testi
Modelin Mac MPS (`torch.device("mps")`) hızlandırıcısı üzerinde 1 batch ileri ve geri (forward/backward) geçişini test etmek için:

```bash
python scripts/smoke_test_mps.py
```

### Modeli Eğitme
```bash
python src/train.py --config configs/config.yaml
```

### İnteraktif Web Arayüzü (Gradio)
```bash
python app.py
```
