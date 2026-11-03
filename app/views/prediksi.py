import io

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import (
    CLASS_LABEL_ID,
    CLASS_ORDER,
    FEATURES,
    MODEL_CODES,
    MODEL_LABEL,
    RISK_STATUS,
    angka,
    angka_ringkas,
    explain_categorical_nb,
    explain_gaussian_nb,
    explain_knn,
    explain_tree,
    is_dark,
    load_metadata,
    load_models,
    out_of_range,
    palette,
    predict_one,
    tabel,
)

meta = load_metadata()
models = load_models()
recommended = meta.get("model_rekomendasi", "DT")
default_index = MODEL_CODES.index(recommended) if recommended in MODEL_CODES else 0


def model_option_label(code):
    return MODEL_LABEL[code] + (" (rekomendasi)" if code == recommended else "")


def proba_chart(probs: dict):
    pal = palette()
    df = pd.DataFrame(
        {
            "kelas": [CLASS_LABEL_ID[c] for c in CLASS_ORDER],
            "kode": CLASS_ORDER,
            "prob": [probs[c] for c in CLASS_ORDER],
        }
    )
    df["label"] = (df.prob * 100).map(lambda v: angka(v, 1)) + "%"
    base = alt.Chart(df).encode(
        y=alt.Y("kelas:N", sort=[CLASS_LABEL_ID[c] for c in CLASS_ORDER], title=None,
                scale=alt.Scale(paddingInner=0.3), axis=alt.Axis(labelOverlap=False)),
        x=alt.X("prob:Q", scale=alt.Scale(domain=[0, 1.12]),
                axis=alt.Axis(format="%", title="Probabilitas", values=[0, 0.25, 0.5, 0.75, 1])),
        tooltip=[alt.Tooltip("kelas:N", title="Kelas"), alt.Tooltip("label:N", title="Probabilitas")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.Color(
            "kode:N",
            scale=alt.Scale(domain=CLASS_ORDER, range=[pal["risk"][c] for c in CLASS_ORDER]),
            legend=None,
        )
    )
    text = base.mark_text(align="left", dx=4, color="#f0efec" if is_dark() else "#0b0b0b").encode(text="label:N")
    return (bars + text).properties(height=alt.Step(34))


def clinical_flags(age, sbp, dbp, temp_f, hr) -> list[str]:
    flags = []
    if sbp >= 140 or dbp >= 90:
        flags.append("Tekanan darah ≥ 140/90 mmHg termasuk hipertensi dalam kehamilan (rekomendasi ISSHP 2021; "
                     "Magee et al., 2022).")
    if temp_f >= 100.4:
        flags.append("Suhu ≥ 38 °C (100,4 °F) menandakan demam.")
    if hr > 100:
        flags.append("Denyut jantung > 100 bpm (takikardia).")
    if age < 20 or age > 35:
        flags.append("Usia < 20 atau > 35 tahun merupakan faktor risiko kehamilan.")
    return flags


def render_result(code: str, x: np.ndarray):
    model = models[code]
    pred, probs = predict_one(model, x)
    status = RISK_STATUS[pred]
    msg = f"**Prediksi: {CLASS_LABEL_ID[pred]}** · {MODEL_LABEL[code]} · keyakinan {probs[pred]:.0%}"
    {"green": st.success, "orange": st.warning, "red": st.error}[status["warna"]](msg, icon=status["ikon"])
    st.altair_chart(proba_chart(probs), width="stretch")

    ood = out_of_range(x)
    if ood:
        st.warning(
            "Input berada di luar rentang data latih, sehingga prediksi kurang dapat diandalkan:\n\n- "
            + "\n- ".join(ood),
            icon=":material/travel_explore:",
        )

    age, sbp, dbp, _, temp_f, hr = x
    flags = clinical_flags(age, sbp, dbp, temp_f, hr)
    if flags:
        st.info("**Catatan klinis umum (bukan keluaran model):**\n\n- " + "\n- ".join(flags),
                icon=":material/stethoscope:")

    with st.expander("Mengapa model memprediksi ini?", icon=":material/psychology:"):
        if code == "DT":
            st.markdown("Jalur keputusan yang dilalui data ini pada pohon keputusan:")
            for i, s in enumerate(explain_tree(model, x), 1):
                st.markdown(f"{i}. {s}")
            st.markdown(f"→ daun berlabel **{CLASS_LABEL_ID[pred]}**")
        elif code in ("NB-G", "NB-K"):
            st.markdown(
                "Naive Bayes menjumlahkan log-likelihood tiap atribut, log P(atribut | kelas), "
                "ditambah log prior kelas. Nilai yang lebih besar (kurang negatif) berarti atribut "
                "tersebut lebih mendukung kelas itu."
            )
            fn = explain_gaussian_nb if code == "NB-G" else explain_categorical_nb
            st.dataframe(tabel(fn(model, x), 3, {"Nilai": angka_ringkas}), hide_index=True)
        else:
            k = model.named_steps["clf"].n_neighbors
            st.markdown(f"KNN memakai suara mayoritas dari **{k} tetangga terdekat** pada data latih:")
            tetangga = explain_knn(model, x)
            kolom = {c: angka_ringkas for c in tetangga.select_dtypes("float").columns if c != "Jarak"}
            st.dataframe(tabel(tetangga, 3, kolom), hide_index=True)

    with st.expander("Bandingkan semua algoritma", icon=":material/compare_arrows:"):
        rows = []
        for c in MODEL_CODES:
            p, pr = predict_one(models[c], x)
            rows.append({"Algoritma": MODEL_LABEL[c], "Prediksi": CLASS_LABEL_ID[p],
                         **{CLASS_LABEL_ID[k]: 100 * pr[k] for k in CLASS_ORDER}})
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                CLASS_LABEL_ID[k]: st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)
                for k in CLASS_ORDER
            },
        )


def render_single():
    col_in, col_out = st.columns([1, 1.25], gap="large")
    with col_in:
        st.subheader("Data pemeriksaan")
        code = st.selectbox(
            "Algoritma", MODEL_CODES, index=default_index, format_func=model_option_label,
            help="Dipilih dari evaluasi Skenario B: macro-F1 tertinggi; jika selisihnya tidak signifikan, dipilih recall Risiko Tinggi tertinggi.",
        )
        age = st.number_input("Usia (tahun)", min_value=10, max_value=70, value=25, step=1)
        c1, c2 = st.columns(2)
        sbp = c1.number_input("Tekanan darah sistolik (mmHg)", 60, 220, 120, step=1)
        dbp = c2.number_input("Tekanan darah diastolik (mmHg)", 30, 140, 80, step=1)

        u1, u2 = st.columns(2)
        bs_unit = u1.segmented_control("Satuan gula darah", ["mmol/L", "mg/dL"], default="mmol/L", required=True)
        t_unit = u2.segmented_control("Satuan suhu", ["°C", "°F"], default="°C", required=True)
        c1, c2 = st.columns(2)
        if bs_unit == "mg/dL":
            bs = c1.number_input("Gula darah (mg/dL)", 36.0, 540.0, 135.0, step=1.0) / 18.0
        else:
            bs = c1.number_input("Gula darah (mmol/L)", 2.0, 30.0, 7.5, step=0.1, format="%.1f")
        if t_unit == "°C":
            temp_f = c2.number_input("Suhu tubuh (°C)", 34.0, 42.0, 36.7, step=0.1, format="%.1f") * 9 / 5 + 32
        else:
            temp_f = c2.number_input("Suhu tubuh (°F)", 93.0, 108.0, 98.0, step=0.1, format="%.1f")

        hr = st.number_input("Denyut jantung (bpm)", 40, 200, 76, step=1)
        st.caption(
            f"Nilai yang dikirim ke model: gula darah **{angka(bs, 2)} mmol/L**, suhu **{angka(temp_f, 1)} °F** "
            "(satuan sesuai dataset)."
        )

    with col_out:
        st.subheader("Hasil")
        if sbp <= dbp:
            st.error("Tekanan sistolik harus lebih besar daripada diastolik. Periksa kembali input.",
                     icon=":material/error:")
        else:
            render_result(code, np.array([age, sbp, dbp, bs, temp_f, hr], dtype=float))


def render_batch():
    st.subheader("Prediksi banyak data sekaligus")
    st.markdown(
        "Unggah file CSV dengan kolom "
        + ", ".join(f"`{f}`" for f in FEATURES)
        + ". Satuan mengikuti dataset: gula darah **mmol/L**, suhu **°F**."
    )
    template = pd.DataFrame([[25, 120, 80, 7.5, 98.0, 76], [35, 140, 90, 13.0, 98.0, 70]], columns=FEATURES)
    st.download_button("Unduh contoh CSV", template.to_csv(index=False), "contoh_input.csv", "text/csv",
                       icon=":material/download:")
    code = st.selectbox("Algoritma", MODEL_CODES, index=default_index, format_func=model_option_label,
                        key="model_batch")
    up = st.file_uploader("File CSV", type=["csv"])
    if up is None:
        return
    try:
        df_in = pd.read_csv(io.BytesIO(up.getvalue()), encoding="utf-8-sig")
    except Exception as e:
        st.error(f"File tidak dapat dibaca sebagai CSV: {e}")
        return
    missing = [f for f in FEATURES if f not in df_in.columns]
    if missing:
        st.error("Kolom berikut tidak ditemukan: " + ", ".join(missing))
        return
    X = df_in[FEATURES].apply(pd.to_numeric, errors="coerce")
    bad = X.isna().any(axis=1) | (X.SystolicBP <= X.DiastolicBP)
    if bad.any():
        st.warning(f"{int(bad.sum())} baris dilewati karena nilai kosong/bukan angka atau sistolik ≤ diastolik.")
    if bad.all():
        return
    Xv = X[~bad].to_numpy(dtype=float)
    m = models[code]
    proba = m.predict_proba(Xv)
    out = df_in.loc[~bad].copy()
    out["Prediksi"] = [CLASS_LABEL_ID[p] for p in m.predict(Xv)]
    for i, c in enumerate(m.classes_):
        out[f"P({CLASS_LABEL_ID[c]})"] = proba[:, i].round(4)
    st.dataframe(tabel(out, 4, {c: angka_ringkas for c in FEATURES if out[c].dtype.kind == "f"}), hide_index=True)
    st.download_button("Unduh hasil prediksi", out.to_csv(index=False), "hasil_prediksi.csv", "text/csv",
                       icon=":material/download:")


st.title("Prediksi Tingkat Risiko Kehamilan")
st.caption(
    "Masukkan hasil pemeriksaan ibu hamil. Model memperkirakan tingkat risiko "
    "(rendah / sedang / tinggi) berdasarkan pola pada dataset *Maternal Health Risk* (UCI)."
)
tab_single, tab_batch = st.tabs([":material/person: Satu pasien", ":material/table: Banyak data (CSV)"])
with tab_single:
    render_single()
with tab_batch:
    render_batch()

st.divider()
st.caption(
    ":material/info: Hasil ini adalah alat bantu skrining edukatif dari model data mining dan **tidak** "
    "menggantikan diagnosis bidan atau dokter. Model dilatih pada data dari Bangladesh sehingga "
    "belum divalidasi untuk populasi Indonesia."
)
