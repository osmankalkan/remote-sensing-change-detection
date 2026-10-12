# 🛰️ Remote Sensing Change Detection (Uydu Görüntülerinde Değişim Tespiti)

Bu proje, aynı coğrafi alanın farklı zamanlarda çekilmiş (**T1** ve **T2** - öncesi ve sonrası) yüksek çözünürlüklü bitemporal uydu görüntülerini karşılaştırarak kentsel alan ve bina değişimlerini piksel düzeyinde ikili maske (`0: Değişim Yok`, `1: Değişim Var`) olarak tespit eden uçtan uca bir derin öğrenme (Deep Learning) pipeline'ıdır.

---

## 🚀 Proje Mimarisi ve Teknik Tercihler

### 1. Model Mimarileri
- **Baseline Model (Early Fusion):**
  - T1 ve T2 görüntüleri kanal bazında birleştirilerek (`[B, 6, H, W]`) modele beslenir.
  - 6 kanallı girdi alacak şekilde uyarlanan `ResNet18` omurgalı U-Net mimarisi (`segmentation_models_pytorch`).
- **Gelişmiş Model (Siamese U-Net / Late Fusion):**
  - **Twin Encoder:** T1 ve T2 görüntüleri, ağırlıkları paylaşımlı (`shared-weights`) `ResNet34` omurgasından bağımsız dallar olarak geçer.
  - **Feature Fusion:** Multi-scale skip connection seviyelerinde mutlak fark $|f_1 - f_2|$ ve kanal birleşimi $[f_1; f_2; |f_1 - f_2|]$ ile zengin değişim öznitelikleri çıkarılır.
  - **CBAM Attention:** Mevsimsel ışık/ton farkları ile radyometrik gürültüleri filtreleyip bina sınırlarına ve gerçek yapısal değişimlere odaklanmak için kanal ve uzamsal dikkat modülü (Convolutional Block Attention Module) entegre edilmiştir.

### 2. Donanım ve Ortam Optimizasyonu
- **Yerel Geliştirme (MacBook Air / Apple Silicon):** Apple Silicon Metal Performance Shaders (`torch.device("mps")`) üzerinde hızlı smoke testleri, veri hazırlığı / patch çıkarma (CPU + SSD optimizasyonu) ve Gradio demo arayüzü.
- **Bulut Eğitimi (Colab / Kaggle / Cloud GPU):** 30–50 epoch'luk tam ölçekli eğitimler için mixed precision (`torch.amp`) ve CUDA hızlandırması.

### 3. Kayıp Fonksiyonları ve Değerlendirme Metrikleri
- **Sınıf Dengesizliği (Class Imbalance) Çözümü:** Değişen binalar tüm uydu görüntüsünün genellikle yalnızca %3–5'lik küçük bir kısmını oluşturur. Bu dengesizliği aşmak için bileşik kayıp fonksiyonu kullanılır:
  $$\mathcal{L}_{Total} = 0.5 \cdot \mathcal{L}_{BCE} + 0.5 \cdot \mathcal{L}_{Dice}$$
  *(Ayrıca zor pikseller için Focal + Dice loss desteği mevcuttur).*
- **Metrikler:** Precision, Recall, F1-Score ve Change-IoU (özellikle değişen pikseller üzerindeki IoU skoru).

---

## 📁 Dizin Yapısı

```text
remote-sensing-change-detection/
├── configs/
│   └── config.yaml              # Hiperparametreler, dosya yolları ve eğitim ayarları
├── data/
│   ├── raw/                     # LEVIR-CD ham uydu görüntüleri (1024x1024)
│   └── processed/               # 256x256 patch'lere ayrılmış veri seti (train/val/test)
│       ├── train/ {A, B, label}
│       ├── val/   {A, B, label}
│       └── test/  {A, B, label}
├── notebooks/
│   ├── 01_eda.ipynb             # Keşifsel veri analizi ve sınıf dağılımları
│   ├── 02_error_analysis.ipynb  # Hata analizi ve model tahmin görselleştirmeleri
│   └── colab_train.ipynb        # Google Colab T4 GPU + Google Drive eğitim defteri
├── scripts/
│   ├── download_levir_cd.py     # LEVIR-CD veri seti indirme / örnek veri scripti
│   ├── patch_images.py          # 1024x1024 -> 256x256 patch çıkarma (tiling) scripti
│   ├── prepare_sample_data.py   # Hızlı testler için sentetik bitemporal veri üreteci
│   ├── verify_dataset.py        # Albumentations augmentasyon pipeline'ı doğrulama scripti
│   ├── smoke_test_mps.py        # 1-batch MPS/CUDA forward & backward smoke testi
│   └── evaluate_large.py        # 1024x1024 görüntüler için kesintisiz kayar pencere (sliding window) çıkarımı
├── src/
│   ├── __init__.py
│   ├── datasets.py              # Senkronize çiftli veri yükleyici (DualAugmentationDataset)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_unet.py     # 6 kanallı Early Fusion U-Net
│   │   ├── siamese_unet.py      # Dual-branch Siamese U-Net
│   │   └── attention.py         # CBAM (Channel & Spatial Attention) blokları
│   ├── losses.py                # BCE, Dice, Focal ve Compound Loss fonksiyonları
│   ├── metrics.py               # IoU ve F1-score hesaplayıcıları
│   ├── utils.py                 # Cihaz seçimi (MPS/CUDA/CPU) ve görselleştirme araçları
│   └── train.py                 # Eğitim ve doğrulama (train/val) döngüsü
├── app.py                       # Gradio tabanlı interaktif web demosu
├── requirements.txt
└── README.md
```

---

## 🛠️ Kurulum ve Hızlı Başlangıç

### 1. Ortam Kurulumu
Gereksinimleri yüklemek için bir sanal ortam oluşturup aktifleştirin:

```bash
# Sanal ortamı oluşturun ve aktifleştirin
python3 -m venv .venv
source .venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt
```

### 2. Veri Hazırlığı ve Patching (256x256)
Modeli test etmek için sentetik uydu çiftleri üretebilir veya gerçek LEVIR-CD veri setini `data/raw` klasörüne yerleştirebilirsiniz:

```bash
# Test amaçlı hızlı sentetik veri oluştur (veya gerçek LEVIR-CD'yi data/raw altına koy)
python scripts/prepare_sample_data.py

# 1024x1024 büyük görüntüleri 256x256 boyutlu patch'lere böl
python scripts/patch_images.py --raw_dir data/raw --processed_dir data/processed --patch_size 256 --stride 256
```

### 3. Çiftli Veri Yükleyici ve Augmentasyon Kontrolü
Geometrik dönüşümlerin (Flip, Rotate90) T1, T2 ve Değişim Maskesi arasında senkronize; fotometrik dönüşümlerin (parlaklık, kontrast, renk tonu) ise zamansal farkları simüle etmek için bağımsız uygulandığını görsel olarak doğrulamak için:

```bash
python scripts/verify_dataset.py
```
*Üretilen kontrol görseli `visualizations/augmentation_verification.png` dosyasına kaydedilir.*

### 4. Donanım ve Pipeline Doğrulama (Smoke Test)
Model mimarisinin ve tensör akışının donanımınızda (Apple Silicon MPS, CUDA veya CPU) sorunsuz çalıştığını 1 batch'lik forward/backward geçişiyle hızlıca teyit etmek için:

```bash
python scripts/smoke_test_mps.py
```

### 5. Modeli Eğitme (Yerel)
Yapılandırma dosyasındaki (`configs/config.yaml`) hiperparametrelerle eğitim döngüsünü başlatın:

```bash
python src/train.py --config configs/config.yaml
```

### 6. Bulutta Eğitim: Google Colab (T4 GPU + Google Drive)
Google Colab'de ücretsiz T4 GPU kullanarak eğitmek ve ağırlıkları doğrudan Google Drive'a kaydetmek için hazırlanan hazır defteri kullanabilirsiniz:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/osmankalkan/remote-sensing-change-detection/blob/main/notebooks/colab_train.ipynb)

**Tek seferlik hazırlık & çalıştırma:**
1. `Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU`
2. `Çalışma zamanı → Tümünü çalıştır` (Run All)

> 💡 **Veri seti için Kaggle token/hesabı zorunlu değildir!** Defter varsayılan olarak açık Hugging Face sunucusundan (`satellite-image-deep-learning/LEVIR-CD`) doğrudan indirir (403 hatası vermez, anında iner ve 256x256 patch'lere dilimler). İsteğe bağlı olarak Kaggle token'ı (`KAGGLE_API_TOKEN`) da kullanılabilir.

Bu notebook (`notebooks/colab_train.ipynb`):
- Tüm ayarları (model, encoder, epoch, batch size, LR, loss) tek bir form hücresinde toplar.
- Google Drive'ı bağlar (`/content/drive/MyDrive/remote_sensing_change_detection/checkpoints/<model>_<encoder>/`); bağlanamazsa yerel diske düşer, hiçbir hücre çökmez.
- GitHub reposunu klonlar / günceller, sadece eksik kütüphaneleri kurar (Colab'in PyTorch/NumPy sürümlerine dokunmaz).
- LEVIR-CD veri setini token gerektirmeden indirir; `scripts/prepare_kaggle_levir.py` ile veriyi otomatik algılar, gerekiyorsa 256x256 patch'lere dilimler ve `data/processed/{train,val,test}/{A,B,label}` yapısına yerleştirip doğrular.
- `src/train.py`'yi **ayrı bir süreçte** çalıştırır (hücreyi durdurmak eğitimi durdurmaz), AMP (fp16) ile T4'te hızlandırır.
- Colab bağlantısı koparsa tekrar **Run all** → `--resume` ile Drive'daki `latest_model.pth`'den kaldığı epoch'tan devam eder.
- `best_model.pth`, `history.csv`, eğitim grafikleri, tahmin görselleri ve test metriklerini Drive'a kaydeder.

Yerelde de aynı şekilde devam ettirilebilir:
```bash
python src/train.py --config configs/config.yaml --resume
```

### 7. Büyük Görüntülerde Çıkarım (Seamless Sliding Window Inference)
Model 256x256 patch'ler üzerinde eğitilir. 1024x1024 tam boyutlu uydu görüntülerinde kenar kesme (seam) izlerini engellemek için Gaussian kernel ağırlıklı kayar pencere algoritması kullanılır:

```bash
python scripts/evaluate_large.py --t1 <T1_path> --t2 <T2_path> --checkpoint checkpoints/best_model.pth
```

### 7. İnteraktif Web Arayüzü (Gradio)
Görüntüleri sürükle-bırak yöntemiyle yükleyip değişim maskesini ve T2 üzerindeki görsel bindirmeyi (overlay) anlık incelemek için:

```bash
python app.py
```
