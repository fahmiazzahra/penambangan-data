"""Aplikasi Streamlit: Klasifikasi Tingkat Risiko Kesehatan Ibu Hamil.

Jalankan dari folder proyek:
    streamlit run app/streamlit_app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Risiko Kesehatan Ibu Hamil",
    page_icon=":material/pregnant_woman:",
    layout="wide",
)

pages = [
    st.Page("views/prediksi.py", title="Prediksi Risiko", icon=":material/monitor_heart:", default=True),
    st.Page("views/eksplorasi.py", title="Eksplorasi Data", icon=":material/insights:"),
    st.Page("views/evaluasi.py", title="Evaluasi Model", icon=":material/fact_check:"),
    st.Page("views/tentang.py", title="Tentang", icon=":material/info:"),
]

with st.sidebar:
    st.caption(
        "Alat bantu skrining edukatif hasil penelitian Data Mining. "
        "**Bukan** pengganti pemeriksaan oleh bidan/dokter."
    )

st.navigation(pages).run()
