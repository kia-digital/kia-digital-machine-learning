import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_articles(file_path='edukasi_artikel.csv'):
    """Memuat data artikel dari file CSV."""
    df = pd.read_csv(file_path)
    # Gabungkan semua kolom teks yang relevan menjadi satu kolom 'content'
    df['content'] = df['Kategori'] + " " + df['Kondisi'] + " " + df['Judul Artikel'] + " " + df['Isi Artikel']
    return df

def map_user_input_to_conditions(age, systolicBP, diastolicBP, bloodsugar, heartrate):
    """
    Memetakan input numerik pengguna ke kondisi tekstual yang relevan dengan artikel.
    Menggunakan threshold umum yang relevan dengan kondisi di edukasi_artikel.csv.
    """
    conditions = []

    # Kondisi Usia
    if age < 18:
        conditions.append("Usia Ibu Hamil Kurang dari 18 Tahun")
    elif age > 35:
        conditions.append("Usia Ibu Hamil Lebih dari 35 Tahun")
    # Tidak ada kondisi spesifik untuk usia 18-35 di artikel yang diberikan,
    # sehingga tidak ditambahkan jika di rentang normal ini.

    # Kondisi Gula Darah (BS)
    # Mengacu pada string 'Kondisi' di artikel untuk Gula Darah Ibu Hamil (BS)
    if bloodsugar < 70:
        conditions.append("Gula Darah Ibu Hamil Rendah (Contoh <70 mg/dL)")
    elif 70 <= bloodsugar <= 100:
        conditions.append("Gula Darah Ibu Hamil Normal (Contoh puasa 70-100 mg/dL)")
    elif bloodsugar > 100:
        conditions.append("Gula Darah Ibu Hamil Tinggi (Contoh >126 mg/dL puasa atau >140 mg/dL 2 jam pp)")

    # Kondisi Tekanan Darah (BP)
    # Mengacu pada string 'Kondisi' di artikel untuk Tekanan Darah Ibu Hamil (BP)
    if systolicBP < 90 or diastolicBP < 60:
        conditions.append("Tekanan Darah Ibu Hamil Rendah (Sistolik <90 mmHg atau Diastolik <60 mmHg)")
    elif (90 <= systolicBP <= 120) and (60 <= diastolicBP <= 80):
        conditions.append("Tekanan Darah Ibu Hamil Normal (Sistolik 90-120 mmHg dan Diastolik 60-80 mmHg)")
    elif systolicBP > 120 or diastolicBP > 80: # Mengasumsikan >120 SBP atau >80 DBP sudah masuk kategori tinggi
        conditions.append("Tekanan Darah Ibu Hamil Tinggi (Sistolik >140 mmHg atau Diastolik >90 mmHg)")

    # Kondisi Detak Jantung (HR)
    # Mengacu pada string 'Kondisi' di artikel untuk Detak Jantung Ibu Hamil (HR)
    if heartrate < 60:
        conditions.append("Detak Jantung Ibu Hamil Rendah (Contoh <60 detak/menit)")
    elif 60 <= heartrate <= 100:
        conditions.append("Detak Jantung Ibu Hamil Normal (Contoh 60-100 detak/menit)")
    elif heartrate > 100:
        conditions.append("Detak Jantung Ibu Hamil Tinggi (Contoh >100 detak/menit)")

    return " ".join(conditions)

def recommend_articles(user_input_dict, df_articles, vectorizer, article_vectors, top_n=3):
    """
    Merekomendasikan artikel berdasarkan input pengguna.

    Args:
        user_input_dict (dict): Dictionary berisi 'age', 'systolicBP', 'diastolicBP',
                               'bloodsugar', 'heartrate'.
        df_articles (pd.DataFrame): DataFrame yang berisi data artikel.
        vectorizer (TfidfVectorizer): Model TF-IDF yang sudah dilatih.
        article_vectors (sparse matrix): Vektor TF-IDF dari semua artikel.
        top_n (int): Jumlah artikel teratas yang akan direkomendasikan.

    Returns:
        pd.DataFrame: DataFrame berisi artikel yang direkomendasikan.
    """
    # 1. Buat profil tekstual pengguna
    user_profile_text = map_user_input_to_conditions(
        user_input_dict['age'],
        user_input_dict['systolicBP'],
        user_input_dict['diastolicBP'],
        user_input_dict['bloodsugar'],
        user_input_dict['heartrate']
    )

    if not user_profile_text:
        print("Tidak ada kondisi relevan yang ditemukan dari input pengguna. Tidak dapat merekomendasikan artikel.")
        return pd.DataFrame()

    # 2. Ubah profil pengguna menjadi vektor TF-IDF menggunakan vectorizer yang sama
    user_vector = vectorizer.transform([user_profile_text])

    # 3. Hitung kemiripan kosinus antara profil pengguna dan semua artikel
    cosine_similarities = cosine_similarity(user_vector, article_vectors).flatten()

    # 4. Dapatkan indeks artikel berdasarkan skor kemiripan tertinggi
    # Menggunakan np.argsort untuk mendapatkan indeks yang akan mengurutkan array
    # Kemudian membalik urutan agar menjadi dari tertinggi ke terendah
    # np.argsort returns indices that would sort an array.
    # [::-1] reverses the array so it's descending order of similarity.
    # [:top_n] takes the top N indices.
    recommended_article_indices = cosine_similarities.argsort()[::-1][:top_n]

    # 5. Ambil artikel yang direkomendasikan
    recommended_articles = df_articles.iloc[recommended_article_indices]

    # Tambahkan skor kemiripan ke hasil rekomendasi untuk analisis
    recommended_articles['Skor Kemiripan'] = cosine_similarities[recommended_article_indices]

    return recommended_articles[['Judul Artikel', 'Kategori', 'Kondisi', 'Skor Kemiripan', 'Isi Artikel']]

# --- Main Program ---
if __name__ == "__main__":
    print("Memuat data artikel...")
    df_articles = load_articles()

    if df_articles.empty:
        print("Gagal memuat artikel atau file kosong.")
    else:
        print(f"Berhasil memuat {len(df_articles)} artikel.")

        # Inisialisasi dan latih TF-IDF Vectorizer
        print("Melatih model TF-IDF...")
        vectorizer = TfidfVectorizer()
        article_vectors = vectorizer.fit_transform(df_articles['content'])
        print("Model TF-IDF selesai dilatih.")

        # --- Contoh Penggunaan ---
        print("\n--- Sistem Rekomendasi Artikel ---")

        # Contoh 1: Ibu hamil dengan kondisi normal
        print("\nContoh 1: Input Normal")
        user_input_normal = {
            'age': 28,
            'systolicBP': 110,
            'diastolicBP': 70,
            'bloodsugar': 90,
            'heartrate': 80
        }
        print(f"Input Pengguna: {user_input_normal}")
        recommended_normal = recommend_articles(user_input_normal, df_articles, vectorizer, article_vectors)
        if not recommended_normal.empty:
            print("Artikel Rekomendasi:")
            for index, row in recommended_normal.iterrows():
                print(f"- Judul: {row['Judul Artikel']} (Kategori: {row['Kategori']}, Kondisi: {row['Kondisi']:.50}..., Skor: {row['Skor Kemiripan']:.2f})")
                print(f"  Isi Ringkas: {row['Isi Artikel'][:150]}...") # Tampilkan sebagian isi artikel

        # Contoh 2: Ibu hamil dengan usia muda dan gula darah tinggi
        print("\nContoh 2: Input Usia Muda & Gula Darah Tinggi")
        user_input_high_risk = {
            'age': 17,
            'systolicBP': 115,
            'diastolicBP': 75,
            'bloodsugar': 150,
            'heartrate': 90
        }
        print(f"Input Pengguna: {user_input_high_risk}")
        recommended_high_risk = recommend_articles(user_input_high_risk, df_articles, vectorizer, article_vectors)
        if not recommended_high_risk.empty:
            print("Artikel Rekomendasi:")
            for index, row in recommended_high_risk.iterrows():
                print(f"- Judul: {row['Judul Artikel']} (Kategori: {row['Kategori']}, Kondisi: {row['Kondisi']:.50}..., Skor: {row['Skor Kemiripan']:.2f})")
                print(f"  Isi Ringkas: {row['Isi Artikel'][:150]}...")

        # Contoh 3: Ibu hamil dengan tekanan darah tinggi dan detak jantung tinggi
        print("\nContoh 3: Input Tekanan Darah Tinggi & Detak Jantung Tinggi")
        user_input_bp_hr = {
            'age': 30,
            'systolicBP': 145,
            'diastolicBP': 95,
            'bloodsugar': 95,
            'heartrate': 110
        }
        print(f"Input Pengguna: {user_input_bp_hr}")
        recommended_bp_hr = recommend_articles(user_input_bp_hr, df_articles, vectorizer, article_vectors)
        if not recommended_bp_hr.empty:
            print("Artikel Rekomendasi:")
            for index, row in recommended_bp_hr.iterrows():
                print(f"- Judul: {row['Judul Artikel']} (Kategori: {row['Kategori']}, Kondisi: {row['Kondisi']:.50}..., Skor: {row['Skor Kemiripan']:.2f})")
                print(f"  Isi Ringkas: {row['Isi Artikel'][:150]}...")