# Maternal Health Risk Classification

Proyek ini bertujuan untuk mengklasifikasikan tingkat risiko kesehatan maternal menggunakan data kesehatan ibu hamil dan model machine learning/deep learning.

## Dataset

- [Maternal Health Risk Data (Kaggle)](https://www.kaggle.com/datasets/csafrit2/maternal-health-risk-data/data)
- Dataset diunduh dan juga tersedia dalam format CSV.

## Struktur Project

- `Model_Klasifikasi_Maternal_Health_Risk_CC25_CF302.ipynb` — Notebook utama untuk EDA, preprocessing, training, dan evaluasi model.
- `requirements.txt` — Daftar dependensi Python.
- `best_model.keras`, `model_klasifikasi_maternal/` — Model hasil training.
- Data CSV (`Maternal_Health_Risk _Dataset.csv`, dll).

## Instalasi

1. **Clone repository**
   ```sh
   git clone https://github.com/kia-digital/kia-digital-machine-learning.git
   cd kia-digital-machine-learning
   ```

2. **Buat environment dan install dependencies**
   ```sh
   python -m venv venv
   source venv/bin/activate  # atau `venv\Scripts\activate` di Windows
   pip install -r requirements.txt
   ```

   Jika menggunakan conda:
   ```sh
   conda create -n tf python=3.9
   conda activate tf
   pip install -r requirements.txt
   ```

3. **(Opsional) Install Jupyter dan ipykernel**
   ```sh
   pip install notebook ipykernel
   python -m ipykernel install --user --name=tf
   ```

## Cara Menjalankan

1. **Buka notebook**
   - Jalankan Jupyter Notebook/Lab atau buka di VS Code.
   - Buka file [Model_Klasifikasi_Maternal_Health_Risk_CC25_CF302.ipynb](Model_Klasifikasi_Maternal_Health_Risk_CC25_CF302.ipynb).

2. **Jalankan setiap cell secara berurutan**
   - Mulai dari import library, EDA, preprocessing, training, hingga evaluasi model.
   - Model yang digunakan: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, dan Deep Learning (TensorFlow/Keras).

3. **Fine Tuning**
   - Hyperparameter tuning menggunakan `keras-tuner`.

4. **Inference**
   - Gunakan fungsi `predict_risk_level_input()` untuk prediksi manual.

## Catatan

- Jika terjadi error terkait `keras_tuner` atau `protobuf`, install/downgrade dengan:
  ```sh
  pip install keras-tuner
  pip install "protobuf<4"
  ```

- Model hasil training akan tersimpan di folder `model_klasifikasi_maternal/`.

## Kontak

Untuk pertanyaan lebih lanjut, silakan hubungi [email Anda].

# 🩺 Sistem Rekomendasi Artikel Berdasarkan Kondisi Kesehatan

Proyek ini membangun sistem rekomendasi artikel kesehatan berbasis prediksi kondisi pengguna menggunakan data medis dan Multi-Layer Perceptron (MLP).

## Dataset

- **Artikel:** `data_artikel_new.csv`  
  Berisi judul, deskripsi, dan tag artikel kesehatan.
- **Pengguna:** Data dummy 20.000 baris, fitur: tekanan darah, gula darah, suhu tubuh, trimester, denyut jantung, dan gejala (yes/no).

## Struktur Project

- `Model_Rekomendasi_Artikel_CC25-CF302.ipynb` — Notebook utama (end-to-end pipeline).
- `requirements.txt` — Daftar dependensi Python.
- `data_artikel_new.csv` — Dataset artikel.

## Instalasi

1. **Clone repository**
   ```sh
   git clone https://github.com/kia-digital/kia-digital-machine-learning.git
   cd kia-digital-machine-learning
   ```

2. **Buat environment dan install dependencies**
   ```sh
   python -m venv venv
   venv\Scripts\activate   # (Windows) atau source venv/bin/activate (Linux/Mac)
   pip install -r requirements.txt
   ```

3. **(Opsional) Install Jupyter**
   ```sh
   pip install notebook ipykernel
   python -m ipykernel install --user --name=venv
   ```

## Cara Menjalankan

1. **Buka notebook**
   - Jalankan Jupyter Notebook/Lab atau buka di VS Code.
   - Buka file `Model_Rekomendasi_Artikel_CC25-CF302.ipynb`.

2. **Jalankan setiap cell secara berurutan:**
   - Import library
   - Load dan eksplorasi data
   - Data preprocessing (scaling, encoding)
   - Training model MLP
   - Evaluasi model (F1, Precision, Recall, Accuracy, Hamming Loss, Jaccard Score)
   - Visualisasi loss dan metrik evaluasi
   - Inference & rekomendasi artikel

3. **Inference**
   - Pilih user dari data test, prediksi kondisi, dan tampilkan rekomendasi artikel berdasarkan tag.

4. **Simpan Model**
   - Model dapat disimpan dengan:
     ```python
     model.save('model_rekomendasi_artikel', save_format='tf')
     ```

## Catatan

- Jika terjadi error terkait dependensi, pastikan versi library sesuai dengan `requirements.txt`.
- Dataset artikel harus tersedia di folder project dengan nama `data_artikel_new.csv`.