import streamlit as st

st.title("Tentang Aplikasi")

st.markdown(
    """
Aplikasi ini adalah luaran penelitian **"Perbandingan Algoritma Naive Bayes, Decision Tree, dan K-Nearest
Neighbor untuk Klasifikasi Tingkat Risiko Kesehatan Ibu Hamil dengan Penanganan Data Duplikat"**, yang disusun
untuk UTS–UAS mata kuliah Penambangan Data.
"""
)

c1, c2 = st.columns(2)
with c1, st.container(border=True):
    st.markdown(
        """
**Penyusun**

| | |
|---|---|
| Nama | Fahmi Fatmawati Azzahra |
| NIM | A11.2024.15831 |
| Kelas | Penambangan Data / A11.4505 |
| Dosen pengampu | Junta Zeniarja, M.Kom |
| Program studi | Teknik Informatika S1, Fakultas Ilmu Komputer, Universitas Dian Nuswantoro |
"""
    )
with c2, st.container(border=True):
    st.markdown(
        """
**Dataset**

*Maternal Health Risk* — Ahmed, M. (2020). UCI Machine Learning Repository.
https://doi.org/10.24432/C5DP5D · Lisensi CC BY 4.0.

1.014 baris, 6 atribut (usia, tekanan darah sistolik & diastolik, gula darah, suhu tubuh,
denyut jantung), 1 target (risiko rendah/sedang/tinggi).
"""
    )

st.subheader("Ringkasan metode")
st.markdown(
    """
1. **Pemahaman data.** Ditemukan 562 baris duplikat eksak (55,4%) dan 35 kombinasi atribut identik yang
   berlabel berbeda.
2. **Pra-pemrosesan.** Deduplikasi; nilai tidak wajar secara fisiologis (mis. denyut jantung 7 bpm)
   dianggap kosong lalu diimputasi median yang dihitung **hanya dari data latih** di setiap fold.
3. **Pemodelan.** Naive Bayes (Gaussian dan Kategorikal dengan Laplace smoothing), Decision Tree (CART),
   dan KNN, dengan grid search hyperparameter.
4. **Evaluasi.** Nested cross-validation 10×10 dengan pembagian berbasis grup (vektor atribut identik
   tidak pernah berada di data latih dan data uji sekaligus); metrik utama macro-F1 dan recall kelas
   Risiko Tinggi; uji signifikansi corrected resampled t-test dengan koreksi Holm.
5. **Tiga skenario.** A = data mentah (praktik umum), B = deduplikasi (usulan), C = B tanpa label konflik.
"""
)

st.subheader("Batasan")
st.markdown(
    """
- Data berasal dari pedesaan Bangladesh; distribusi pasien Indonesia bisa berbeda, sehingga model perlu
  divalidasi dengan data lokal (mis. data Puskesmas) sebelum dipakai di lapangan.
- Dataset tidak memiliki ID pasien. Deduplikasi menghapus salinan persis, tetapi pengukuran berulang pasien
  yang sama dengan nilai sedikit berbeda masih mungkin ada.
- Label konflik menunjukkan ketidakpastian pelabelan pada data sumber.
- Aplikasi ini **bukan** alat diagnosis; hasilnya hanya pendukung edukasi dan skrining awal.
"""
)

st.subheader("Cara menjalankan ulang")
st.code(
    """python -m venv .venv
.venv\\Scripts\\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python -m src.eda                 # profil data + grafik EDA
python -m src.experiment          # eksperimen 10x10 nested CV (± 30-60 menit)
python -m src.train_final         # model final untuk aplikasi
streamlit run app/streamlit_app.py""",
    language="bash",
)
