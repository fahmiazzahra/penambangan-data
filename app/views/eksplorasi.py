import altair as alt
import pandas as pd
import streamlit as st

from common import (
    CLASS_LABEL_ID,
    CLASS_ORDER,
    FEATURES,
    TARGET,
    feature_label,
    is_dark,
    load_raw_df,
    load_results,
    palette,
)

st.title("Eksplorasi Data")
st.caption(
    "Dataset *Maternal Health Risk* (Ahmed, 2020; UCI Machine Learning Repository, CC BY 4.0): "
    "data pemeriksaan ibu hamil dari rumah sakit, klinik komunitas, dan layanan kesehatan ibu "
    "di pedesaan Bangladesh yang dikumpulkan melalui sistem pemantauan berbasis IoT."
)

raw = load_raw_df()
dedup = raw.drop_duplicates()
prof = load_results()["profil"] or {}
pal = palette()

row1, row2 = st.columns(2), st.columns(2)
row1[0].metric("Baris data mentah", f"{len(raw):,}".replace(",", "."), border=True)
row1[1].metric("Duplikat eksak", f"{raw.duplicated().sum()}",
               f"{100 * raw.duplicated().mean():.1f}% dari data", delta_color="off", delta_arrow="off", border=True)
row2[0].metric("Baris unik", f"{len(dedup)}", border=True)
row2[1].metric("Atribut identik, label berbeda", f"{prof.get('n_vektor_konflik', '–')} vektor",
               f"{prof.get('n_baris_unik_konflik', '–')} baris unik", delta_color="off", delta_arrow="off",
               border=True)

st.subheader("Distribusi kelas: sebelum dan sesudah deduplikasi")
dist = pd.concat(
    [
        raw[TARGET].value_counts().rename("jumlah").to_frame().assign(versi="Data mentah"),
        dedup[TARGET].value_counts().rename("jumlah").to_frame().assign(versi="Setelah deduplikasi"),
    ]
).reset_index(names="kode")
dist["kelas"] = dist.kode.map(CLASS_LABEL_ID)
dist["persen"] = dist.jumlah / dist.groupby("versi").jumlah.transform("sum")
order = [CLASS_LABEL_ID[c] for c in CLASS_ORDER]
chart = (
    alt.Chart(dist)
    .mark_bar(size=24, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    .encode(
        x=alt.X("kelas:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("versi:N"),
        y=alt.Y("jumlah:Q", title="Jumlah baris"),
        color=alt.Color("versi:N", scale=alt.Scale(range=[pal["scenario"]["A"], pal["scenario"]["B"]]),
                        legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("versi:N", title="Versi"), alt.Tooltip("kelas:N", title="Kelas"),
                 alt.Tooltip("jumlah:Q", title="Jumlah"), alt.Tooltip("persen:Q", title="Proporsi", format=".1%")],
    )
    .properties(height=300)
)
st.altair_chart(chart, width="stretch")
st.markdown(
    f"Sebanyak **{raw.duplicated().sum()} baris ({100 * raw.duplicated().mean():.1f}%)** adalah salinan persis "
    "baris lain, dan satu kombinasi nilai bisa berulang hingga "
    f"**{prof.get('pengulangan_maksimum', '–')} kali**. Duplikasi paling banyak terjadi pada kelas *Risiko Sedang*, "
    "sehingga proporsi kelas berubah setelah deduplikasi. Bila data mentah langsung dibagi menjadi data "
    "latih dan uji, salinan yang sama dapat muncul di keduanya dan membuat akurasi tampak lebih tinggi "
    "daripada kemampuan model yang sebenarnya (kebocoran data)."
)

st.subheader("Sebaran atribut per tingkat risiko")
feat = st.segmented_control("Atribut", FEATURES, default="BS", format_func=feature_label, required=True)
src = st.radio("Data", ["Setelah deduplikasi", "Data mentah"], horizontal=True)
d = (dedup if src == "Setelah deduplikasi" else raw).copy()
d["kelas"] = d[TARGET].map(CLASS_LABEL_ID)
box = (
    alt.Chart(d)
    .mark_boxplot(size=40, extent=1.5, median={"color": "white"}, rule={"color": "#898781"},
                  ticks={"color": "#898781"}, outliers={"color": "#898781"})
    .encode(
        x=alt.X("kelas:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{feat}:Q", title=feature_label(feat), scale=alt.Scale(zero=False)),
        color=alt.Color(f"{TARGET}:N", scale=alt.Scale(domain=CLASS_ORDER, range=[pal["risk"][c] for c in CLASS_ORDER]),
                        legend=None),
    )
    .properties(height=320)
)
st.altair_chart(box, width="stretch")
med = d.groupby("kelas")[feat].median().reindex(order)
st.caption("Median per kelas: " + " · ".join(f"{k}: **{v:g}**" for k, v in med.items()))

st.subheader("Korelasi antaratribut (Spearman, data unik)")
cd = dedup.copy()
cd["Tingkat risiko"] = cd[TARGET].map({c: i for i, c in enumerate(CLASS_ORDER)})
cols = FEATURES + ["Tingkat risiko"]
corr = cd[cols].corr(method="spearman")
names = {f: feature_label(f) for f in FEATURES} | {"Tingkat risiko": "Tingkat risiko (ordinal)"}
long = corr.rename(index=names, columns=names).reset_index(names="a").melt("a", var_name="b", value_name="r")
heat = alt.Chart(long).encode(
    x=alt.X("b:N", sort=list(names.values()), title=None, axis=alt.Axis(labelAngle=-35)),
    y=alt.Y("a:N", sort=list(names.values()), title=None),
    tooltip=[alt.Tooltip("a:N", title="Atribut 1"), alt.Tooltip("b:N", title="Atribut 2"),
             alt.Tooltip("r:Q", title="ρ", format=".2f")],
)
rect = heat.mark_rect().encode(
    color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 0, 1], range=pal["div"], interpolate="lab"),
                    legend=alt.Legend(title="ρ"))
)
if is_dark():
    txt_color = alt.value("#f0efec")
else:
    txt_color = alt.when("abs(datum.r) > 0.6").then(alt.value("#ffffff")).otherwise(alt.value("#0b0b0b"))
txt = heat.mark_text(fontSize=11).encode(text=alt.Text("r:Q", format=".2f"), color=txt_color)
st.altair_chart((rect + txt).properties(height=380), width="stretch")

st.subheader("Data")
only_dup = st.toggle("Tampilkan hanya baris yang memiliki duplikat")
view = raw[raw.duplicated(keep=False)] if only_dup else raw
st.dataframe(
    view.rename(columns={f: feature_label(f) for f in FEATURES} | {TARGET: "Tingkat risiko"}),
    hide_index=False,
    height=320,
)
st.caption(f"{len(view)} baris ditampilkan. Indeks = nomor baris pada file CSV asli (mulai dari 0).")
