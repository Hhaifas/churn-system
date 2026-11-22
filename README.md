# Telco Customer Churn — Analisis Data & Prediksi

Analisis dan pemodelan *customer churn* (pelanggan berhenti berlangganan) pada perusahaan telekomunikasi, menggunakan dataset **Telco Customer Churn** dari IBM (via Kaggle). Seluruh analisis ada di [`notebooks/telco_churn.ipynb`](notebooks/telco_churn.ipynb).

**Tujuan:** memahami karakteristik pelanggan yang churn dan membangun model klasifikasi untuk memprediksi apakah seorang pelanggan akan churn (`Churn = Yes/No`).

---

## 1. Dataset

| Item | Keterangan |
|---|---|
| Sumber | Kaggle — [`blastchar/telco-customer-churn`](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| File | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Ukuran awal | 7.043 baris × 21 kolom |
| Setelah cleaning | 7.021 baris × 20 kolom |
| Target | `Churn` (Yes / No) |

**Kelompok fitur:**

- **Demografi:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Layanan:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Akun & tagihan:** `tenure` (bulan), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`

---

## 2. Data Cleaning

| Langkah | Detail |
|---|---|
| Hapus ID | `customerID` di-drop karena tidak informatif. |
| Duplikat | Ditemukan **22 baris duplikat** (setelah `customerID` dibuang) → dihapus. |
| Tipe data `TotalCharges` | Awalnya bertipe `object` karena ada nilai string kosong. Dikonversi dengan `pd.to_numeric(errors="coerce")` → muncul **11 nilai NaN**. |
| Missing value | 11 NaN pada `TotalCharges` diisi dengan **rata-rata** kolom. Ke-11 baris ini adalah pelanggan dengan `tenure = 0` (pelanggan baru yang belum pernah ditagih). |

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Statistik ringkas

| Fitur | Mean | Median | Min | Max |
|---|---|---|---|---|
| `tenure` (bulan) | 32,5 | 29 | 0 | 72 |
| `MonthlyCharges` | 64,85 | 70,40 | 18,25 | 118,75 |
| `SeniorCitizen` | 16,3% pelanggan lansia | | | |

Profil umum pelanggan (nilai modus):

- **Gender** seimbang: 3.541 laki-laki vs 3.480 perempuan.
- Mayoritas pelanggan memakai kontrak **Month-to-month** (3.853 / 54,9%).
- Internet paling banyak **Fiber optic** (3.090 / 44,0%).
- Metode bayar terbanyak **Electronic check** (2.359 / 33,6%).
- Mayoritas memakai **Paperless Billing** (4.161 / 59,3%).

### 3.2 Distribusi target — data tidak seimbang

| Churn | Jumlah | Proporsi |
|---|---|---|
| No | 5.164 | 73,6% |
| Yes | 1.857 | **26,4%** |

Kelas churn hanya ~1/4 data, sehingga model cenderung bias ke kelas mayoritas. Ini ditangani dengan SMOTE (lihat bagian 4).

### 3.3 Korelasi fitur terhadap `Churn`

Untuk analisis korelasi, setiap kolom kategorikal di-*map* ke angka secara manual, lalu dihitung korelasi Pearson.

| Fitur | Korelasi dengan Churn | Interpretasi |
|---|---|---|
| `tenure` | **−0,35** | Semakin lama berlangganan, semakin kecil kemungkinan churn. |
| `OnlineSecurity` / `TechSupport` | −0,31 | Pelanggan tanpa layanan keamanan/dukungan teknis lebih rentan churn. |
| `InternetService` | −0,30 | Pelanggan Fiber optic lebih banyak churn dibanding DSL. |
| `OnlineBackup` / `DeviceProtection` | −0,28 / −0,27 | Layanan tambahan menurunkan risiko churn. |
| `TotalCharges` | −0,20 | Sejalan dengan `tenure` (korelasi `tenure`–`TotalCharges` = 0,82). |
| `MonthlyCharges` | +0,19 | Tagihan bulanan lebih tinggi → risiko churn lebih tinggi. |
| `PaperlessBilling` | +0,19 | Pengguna tagihan digital sedikit lebih rentan churn. |
| `Contract` | −0,18 | Kontrak lebih panjang → churn lebih rendah. |
| `SeniorCitizen` | +0,15 | Pelanggan lansia sedikit lebih rentan churn. |
| `Partner` / `Dependents` | −0,15 / −0,16 | Pelanggan yang punya pasangan/tanggungan lebih loyal. |
| `gender`, `PhoneService`, `MultipleLines` | ≈ 0,01 | **Tidak berpengaruh** terhadap churn. |

Temuan lain:

- `PhoneService` dan `MultipleLines` berkorelasi sangat kuat (−0,85), sehingga redundan.
- Layanan add-on internet (`OnlineSecurity` … `StreamingMovies`) saling berkorelasi tinggi (0,82–0,91) karena semuanya bernilai *"No internet service"* untuk pelanggan tanpa internet.

### 3.4 Insight bisnis

1. **Pelanggan baru adalah kelompok paling berisiko.** Program *onboarding* dan retensi di bulan-bulan awal berpotensi paling efektif.
2. **Kontrak bulanan rawan churn.** Insentif untuk beralih ke kontrak 1–2 tahun bisa menurunkan churn.
3. **Layanan pendukung (security, tech support, backup) bersifat "lengket".** Bundling atau trial gratis layanan ini bisa meningkatkan retensi.
4. **Pelanggan Fiber optic dengan tagihan tinggi** perlu dicek dari sisi harga dan kualitas layanan.

---

## 4. Feature Engineering & Preprocessing

1. **Seleksi fitur.** `gender`, `PhoneService`, dan `MultipleLines` di-drop karena korelasinya dengan churn ≈ 0. Hasilnya 16 fitur + target.
2. **Encoding:**
   - *One-hot encoding* untuk `InternetService`, `Contract`, `PaymentMethod`.
   - *Label encoding* untuk kolom kategorikal lainnya (Yes/No/No internet service → 0/1/2) dan target `Churn`.
   - Total: **23 fitur** setelah encoding.
3. **Penyeimbangan kelas (SMOTE).** Kelas 0 : 1 = 5.164 : 1.857 → 5.164 : 5.164 (total 10.328 baris).
4. **Train-test split** 80:20 (`random_state=42`): 8.262 data latih, 2.066 data uji.
5. **Scaling.** `StandardScaler` pada `tenure`, `MonthlyCharges`, `TotalCharges` (di-*fit* hanya pada data latih).

---

## 5. Modeling

Empat algoritma dibandingkan:

| Kode | Model |
|---|---|
| KNN | `KNeighborsClassifier` |
| RF | `RandomForestClassifier` |
| XBoost | `GradientBoostingClassifier` (scikit-learn, **bukan** library XGBoost) |
| SVM | `SVC` |

### 5.1 Hyperparameter tuning (GridSearchCV, 5-fold, metrik accuracy)

| Model | Best parameters | CV accuracy |
|---|---|---|
| Random Forest | `max_depth=20`, `min_samples_split=5`, `n_estimators=100` | **0,854** |
| KNN | `n_neighbors=7`, `p=1`, `weights="distance"`, `leaf_size=10`, `algorithm="auto"`, `metric="minkowski"` | 0,832 |
| Gradient Boosting | Belum ada hasil (eksekusi grid search terhenti) | — |

Random Forest terbaik pada data uji: **test accuracy = 0,7851**.

### 5.2 Ringkasan evaluasi (parameter default)

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| KNN | 0,760 | 0,523 | 0,477 |
| RF | 0,756 | 0,514 | 0,469 |
| **Gradient Boosting** | **0,781** | 0,570 | **0,511** |
| SVM | 0,778 | **0,570** | 0,463 |

Dari tabel ini, Gradient Boosting memberikan accuracy dan recall tertinggi, walaupun selisih antar model kecil. Namun lihat catatan di bagian 6 sebelum tabel ini dijadikan acuan.

---

## 6. Catatan & Keterbatasan

Beberapa hal pada notebook perlu diperhatikan sebelum hasilnya dipakai sebagai acuan:

1. **SMOTE diterapkan sebelum train-test split.** Akibatnya data uji berisi sampel sintetis yang dibuat dari data latih (*data leakage*), sehingga skor evaluasi cenderung terlalu optimistis. Ini juga terlihat dari selisih CV accuracy RF (0,854) dengan test accuracy (0,785). Sebaiknya SMOTE dijalankan **hanya pada data latih**, misalnya dengan `imblearn.pipeline.Pipeline` di dalam cross-validation.
2. **Output tabel evaluasi (5.2) kemungkinan berasal dari eksekusi lama.**
   - `KNeighborsClassifier(random_state=42)` akan menghasilkan error karena KNN tidak punya parameter `random_state`, jadi sel itu tidak mungkin menghasilkan output tersebut dengan kode saat ini.
   - Pada data uji yang seimbang (hasil SMOTE), accuracy ~0,78 tidak konsisten dengan precision/recall ~0,5.

   Notebook perlu dijalankan ulang dari awal agar angkanya valid.
3. **Grid search Gradient Boosting** belum selesai. Selain itu, nilai `criterion="friendman_mse"` salah ketik (seharusnya `"friedman_mse"`), dan ruang parameternya besar (≈ 8.700 kombinasi × 5 fold).
4. **Metrik.** Untuk kasus churn, *recall* kelas churn biasanya lebih penting daripada accuracy (lebih baik salah menandai pelanggan loyal daripada melewatkan pelanggan yang akan churn). Pertimbangkan `scoring="recall"`, `"f1"`, atau ROC-AUC, serta tambahkan confusion matrix.
5. **Encoding untuk korelasi.** Mapping manual (mis. `"No internet service" → 3`, `"Two year" → 3`) bersifat ordinal dan sembarang, jadi nilai korelasi pada kolom tersebut hanya indikatif.
6. **Imputasi `TotalCharges`.** 11 nilai kosong berasal dari pelanggan dengan `tenure = 0`, sehingga imputasi dengan `0` (atau `MonthlyCharges`) lebih masuk akal daripada rata-rata.
7. **Feature importance.** Belum ada analisis feature importance dari model (mis. `best_estimator_.feature_importances_` atau SHAP) untuk memvalidasi temuan korelasi.

---

## 7. Cara Menjalankan

Notebook ini dibuat di **Google Colab**.

```bash
pip install pandas scikit-learn matplotlib seaborn imbalanced-learn kagglehub
```

1. Buka `notebooks/telco_churn.ipynb` di Colab atau Jupyter.
2. Dataset diunduh otomatis lewat `kagglehub.dataset_download("blastchar/telco-customer-churn")`. Simpan kredensial Kaggle di `~/.kaggle/kaggle.json` dan **jangan menulis API key langsung di notebook**.
3. Jalankan semua sel secara berurutan (*Run all*).

---

## 8. Struktur Proyek

```
churn-system/
├── notebooks/
│   └── telco_churn.ipynb   # EDA, preprocessing, modeling
├── src/                    # modul data, features, utils
├── tests/                  # unit test
├── data/{raw,processed}/
├── models/
├── reports/
├── configs/
├── requirements.txt
└── requirements-dev.txt
```
