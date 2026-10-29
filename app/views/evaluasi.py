import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from common import (
    CLASS_LABEL_ID,
    CLASS_ORDER,
    MODEL_LABEL,
    SCENARIO_LABEL,
    is_dark,
    load_metadata,
    load_results,
    palette,
)

st.title("Evaluasi Model")

res = load_results()
summ, folds = res["ringkasan"], res["per_fold"]
if summ is None or folds is None:
    st.warning("Hasil eksperimen belum tersedia. Jalankan `python -m src.experiment` terlebih dahulu.")
    st.stop()

meta_exp = res["meta"] or {}
pal = palette()
MODELS = ["BASELINE", "NB-G", "NB-K", "DT", "KNN"]
METRICS = {
    "f1_macro": "Macro-F1",
    "accuracy": "Akurasi",
    "balanced_accuracy": "Balanced accuracy",
    "recall_high": "Recall Risiko Tinggi",
    "recall_mid": "Recall Risiko Sedang",
    "recall_low": "Recall Risiko Rendah",
    "precision_high": "Presisi Risiko Tinggi",
    "auc_ovr_macro": "ROC-AUC (OvR, macro)",
}

st.caption(
    f"Protokol: {meta_exp.get('repeats', 10)}×{meta_exp.get('outer_folds', 10)} repeated stratified (group) "
    f"k-fold; tuning hyperparameter dengan grid search {meta_exp.get('inner_folds', 5)}-fold di dalam setiap "
    "fold latih (nested CV). Nilai = rata-rata ± simpangan baku dari seluruh fold uji."
)

tuned = summ[(summ.konfigurasi == "tuned") | (summ.model == "BASELINE")]

# ----------------------------------------------------------------------------- ringkasan utama
b = tuned[(tuned.skenario == "B") & (tuned.model != "BASELINE")].set_index("model")
a = tuned[tuned.skenario == "A"].set_index("model")
best = load_metadata().get("model_rekomendasi") or b.f1_macro_mean.idxmax()
overlap = meta_exp.get("baris_uji_dengan_salinan_di_data_latih", {})
infl = (a.f1_macro_mean - b.f1_macro_mean).drop("BASELINE", errors="ignore")
row1, row2 = st.columns(2), st.columns(2)
row1[0].metric("Model rekomendasi (skenario B)", MODEL_LABEL[best], border=True,
               help="Macro-F1 tertinggi; bila selisih antarmodel tidak signifikan, dipilih recall Risiko Tinggi tertinggi.")
row1[1].metric("Macro-F1 model rekomendasi", f"{b.loc[best, 'f1_macro_mean']:.3f} ± {b.loc[best, 'f1_macro_std']:.3f}",
               f"recall Risiko Tinggi {b.loc[best, 'recall_high_mean']:.3f}", delta_color="off", delta_arrow="off",
               border=True)
row2[0].metric("Inflasi macro-F1 terbesar (A − B)", f"{infl.max():+.3f}", MODEL_LABEL[infl.idxmax()],
               delta_color="off", delta_arrow="off", border=True)
if "A" in overlap:
    row2[1].metric("Baris uji A yang punya salinan di data latih", f"{overlap['A']['rata2_persen']:.1f}%",
                   f"skenario B: {overlap.get('B', {}).get('rata2_persen', 0):.1f}%", delta_color="off",
                   delta_arrow="off", border=True)

# ----------------------------------------------------------------------------- perbandingan skenario
st.subheader("Perbandingan skenario")
metric = st.selectbox("Metrik", list(METRICS), format_func=METRICS.get)
plot = tuned[["skenario", "model", f"{metric}_mean", f"{metric}_std"]].rename(
    columns={f"{metric}_mean": "mean", f"{metric}_std": "sd"}
)
plot["Algoritma"] = plot.model.map(MODEL_LABEL)
plot["Skenario"] = plot.skenario.map(SCENARIO_LABEL)
plot["lo"] = plot["mean"] - plot.sd
plot["hi"] = plot["mean"] + plot.sd
order = [MODEL_LABEL[m] for m in MODELS]
enc = dict(
    x=alt.X("Algoritma:N", sort=order, title=None, axis=alt.Axis(labelAngle=0, labelLimit=160)),
    xOffset=alt.XOffset("Skenario:N"),
)
tooltip = [alt.Tooltip("Skenario:N"), alt.Tooltip("Algoritma:N"), alt.Tooltip("mean:Q", title="Rata-rata", format=".3f"),
           alt.Tooltip("sd:Q", title="Simpangan baku", format=".3f")]
bars = alt.Chart(plot).mark_bar(size=22, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
    **enc,
    y=alt.Y("mean:Q", title=METRICS[metric], scale=alt.Scale(domain=[0, 1])),
    color=alt.Color("Skenario:N", scale=alt.Scale(domain=list(SCENARIO_LABEL.values()),
                                                  range=[pal["scenario"][s] for s in "ABC"]),
                    legend=alt.Legend(title=None, orient="top")),
    tooltip=tooltip,
)
err = alt.Chart(plot).mark_rule(color="#898781").encode(**enc, y="lo:Q", y2="hi:Q", tooltip=tooltip)
st.altair_chart((bars + err).properties(height=340), width="stretch")
st.caption(
    "Skenario A memakai data mentah sehingga salinan baris yang sama bisa berada di data latih dan data uji. "
    "Selisih A − B menunjukkan seberapa besar kinerja ‘menggelembung’ akibat duplikasi."
)

with st.expander("Tabel inflasi kinerja (A − B)"):
    inf = res["inflasi"]
    if inf is not None:
        inf = inf.copy()
        inf["model"] = inf["model"].map(MODEL_LABEL)
        st.dataframe(inf.round(3), hide_index=True)

# ----------------------------------------------------------------------------- tabel lengkap
st.subheader("Tabel ringkasan")
c1, c2 = st.columns(2)
scn = c1.radio("Skenario", list(SCENARIO_LABEL), index=1, format_func=SCENARIO_LABEL.get)
cfg = c2.radio("Konfigurasi", ["tuned", "default"],
               format_func={"tuned": "Hasil tuning", "default": "Parameter bawaan"}.get)
t = summ[(summ.skenario == scn) & ((summ.konfigurasi == cfg) | (summ.model == "BASELINE"))].copy()
t["model"] = pd.Categorical(t.model, MODELS, ordered=True)
t = t.sort_values("model")
show = pd.DataFrame({"Algoritma": t.model.map(MODEL_LABEL)})
for m, label in METRICS.items():
    show[label] = t[f"{m}_mean"].map("{:.3f}".format) + " ± " + t[f"{m}_std"].map("{:.3f}".format)
st.dataframe(show, hide_index=True)

st.markdown("**Kompleksitas** (rata-rata per fold)")
cx = pd.DataFrame({
    "Algoritma": t.model.map(MODEL_LABEL),
    "Waktu latih (ms)": t.waktu_latih_ms_mean.round(2),
    "Waktu prediksi per 1.000 data (ms)": t.waktu_prediksi_ms_per_1000_mean.round(2),
    "Ukuran model (nilai tersimpan)": t.ukuran_model_mean.round(0),
})
st.dataframe(cx, hide_index=True)
st.caption(
    "Ukuran model: Decision Tree = jumlah simpul; KNN = jumlah data latih × atribut (KNN menyimpan seluruh "
    "data latih); Naive Bayes = jumlah parameter probabilitas."
)

# ----------------------------------------------------------------------------- confusion matrix
st.subheader("Confusion matrix (dinormalisasi per kelas aktual)")
sub = folds[(folds.skenario == scn) & (folds.konfigurasi == cfg) & (folds.model != "BASELINE")]
rows = []
short = {c: CLASS_LABEL_ID[c].replace("Risiko ", "") for c in CLASS_ORDER}
for m, g in sub.groupby("model"):
    cm = np.sum([np.array(json.loads(x)) for x in g.cm], axis=0)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    for i, ta in enumerate(CLASS_ORDER):
        for j, pr in enumerate(CLASS_ORDER):
            rows.append({"Algoritma": MODEL_LABEL[m], "Aktual": short[ta], "Prediksi": short[pr],
                         "proporsi": cmn[i, j], "jumlah": int(cm[i, j])})
cmdf = pd.DataFrame(rows)
lab = [short[c] for c in CLASS_ORDER]
base = alt.Chart(cmdf).encode(
    x=alt.X("Prediksi:N", sort=lab, axis=alt.Axis(labelAngle=0, labelOverlap=False)),
    y=alt.Y("Aktual:N", sort=lab, axis=alt.Axis(labelOverlap=False)),
    tooltip=["Algoritma", "Aktual", "Prediksi", alt.Tooltip("proporsi:Q", format=".1%"),
             alt.Tooltip("jumlah:Q", title="Jumlah (akumulasi fold)")],
)
heat = base.mark_rect().encode(
    color=alt.Color("proporsi:Q", scale=alt.Scale(domain=[0, 1], range=pal["seq"], interpolate="lab"), legend=None)
)
strong, weak = ("#0b0b0b", "#f0efec") if is_dark() else ("#ffffff", "#0b0b0b")
txt = base.mark_text(fontSize=12).encode(
    text=alt.Text("proporsi:Q", format=".2f"),
    color=alt.when(alt.datum.proporsi > 0.55).then(alt.value(strong)).otherwise(alt.value(weak)),
)
st.altair_chart(
    (heat + txt).properties(width=210, height=170).facet(
        facet=alt.Facet("Algoritma:N", sort=[MODEL_LABEL[m] for m in MODELS[1:]], title=None), columns=2
    ),
)

# ----------------------------------------------------------------------------- uji statistik
st.subheader("Uji signifikansi antaralgoritma")
uji = res["uji"]
if uji is not None:
    met = st.segmented_control("Metrik uji", ["f1_macro", "recall_high"], default="f1_macro",
                               format_func=METRICS.get, required=True)
    u = uji[(uji.skenario == scn) & (uji.metrik == met)].copy()
    u["Perbandingan"] = u.model_1.map(MODEL_LABEL) + " vs " + u.model_2.map(MODEL_LABEL)
    st.dataframe(
        u[["Perbandingan", "selisih_rata2", "t", "p", "p_holm", "signifikan_0.05"]].rename(
            columns={"selisih_rata2": "Selisih rata-rata", "p_holm": "p (Holm)", "signifikan_0.05": "Signifikan (α=0,05)"}
        ).round(4),
        hide_index=True,
    )
    st.caption(
        "Corrected resampled t-test (Nadeau & Bengio, 2003) untuk skor k-fold berulang; nilai p dikoreksi "
        "dengan metode Holm karena ada banyak perbandingan sekaligus. Hanya konfigurasi hasil tuning."
    )

# ----------------------------------------------------------------------------- model final
st.subheader("Model final di aplikasi")
meta = load_metadata()
if meta:
    rows = []
    for code, info in meta["model"].items():
        rows.append({
            "Algoritma": MODEL_LABEL[code],
            "Hyperparameter terpilih": ", ".join(f"{k.split('__')[-1]}={v}" for k, v in info["parameter_terbaik"].items()),
            "Macro-F1 nested CV (B)": info.get("nested_cv_skenario_B", {}).get("f1_macro", {}).get("mean"),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        f"Model dilatih ulang pada seluruh {meta['dataset']['n_latih']} baris unik. Rekomendasi: "
        f"**{MODEL_LABEL[meta['model_rekomendasi']]}** ({meta['dasar_rekomendasi']})."
    )
