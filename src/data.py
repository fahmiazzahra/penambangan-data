"""Memuat dataset dan menyiapkan tiga versi data untuk skenario eksperimen.

Skenario A  : data mentah (1.014 baris), mengikuti praktik yang umum di studi terdahulu.
Skenario B  : duplikat eksak dihapus; baris dengan vektor atribut identik dikunci
              dalam satu grup agar tidak pernah muncul di data latih dan data uji sekaligus.
Skenario C  : seperti B, tetapi vektor atribut yang labelnya saling bertentangan dibuang
              (analisis sensitivitas terhadap inkonsistensi label).
"""
from __future__ import annotations

import pandas as pd

from .config import CLASS_ORDER, FEATURES, PLAUSIBLE_RANGE, RAW_CSV, TARGET


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Kolom tidak ditemukan: {sorted(missing_cols)}")
    df[TARGET] = df[TARGET].str.strip().str.lower()
    unknown = set(df[TARGET].unique()) - set(CLASS_ORDER)
    if unknown:
        raise ValueError(f"Label tidak dikenal: {sorted(unknown)}")
    return df[FEATURES + [TARGET]]


def feature_vector_id(df: pd.DataFrame) -> pd.Series:
    """ID grup: baris dengan nilai atribut identik mendapat ID yang sama."""
    return df.groupby(FEATURES, sort=False).ngroup()


def conflicting_vectors(df: pd.DataFrame) -> pd.DataFrame:
    """Vektor atribut yang muncul dengan lebih dari satu label kelas."""
    n_labels = df.drop_duplicates().groupby(FEATURES)[TARGET].nunique()
    return n_labels[n_labels > 1].reset_index()[FEATURES]


def implausible_mask(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {c: ~df[c].between(*PLAUSIBLE_RANGE[c]) for c in FEATURES}, index=df.index
    )


def build_scenarios(df_raw: pd.DataFrame | None = None) -> dict[str, dict]:
    df_raw = load_raw() if df_raw is None else df_raw

    a = df_raw.reset_index(drop=True)
    b = df_raw.drop_duplicates().reset_index(drop=True)
    conflicts = conflicting_vectors(df_raw)
    c = (
        b.merge(conflicts.assign(_conflict=True), on=FEATURES, how="left")
        .query("_conflict != True")
        .drop(columns="_conflict")
        .reset_index(drop=True)
    )

    return {
        "A": {
            "nama": "A - Data mentah (praktik umum)",
            "data": a,
            # Setiap baris adalah grupnya sendiri -> setara stratified k-fold biasa,
            # sehingga duplikat bisa tersebar ke data latih dan data uji.
            "groups": pd.Series(range(len(a))),
        },
        "B": {
            "nama": "B - Deduplikasi + validasi berbasis grup (usulan)",
            "data": b,
            "groups": feature_vector_id(b),
        },
        "C": {
            "nama": "C - B tanpa vektor berlabel konflik (sensitivitas)",
            "data": c,
            "groups": feature_vector_id(c),
        },
    }


def profile(df_raw: pd.DataFrame | None = None) -> dict:
    """Ringkasan kualitas data yang dikutip di proposal dan ditampilkan di aplikasi."""
    df_raw = load_raw() if df_raw is None else df_raw
    dedup = df_raw.drop_duplicates()
    conflicts = conflicting_vectors(df_raw)
    conflict_rows = dedup.merge(conflicts, on=FEATURES, how="inner")
    bad = implausible_mask(df_raw)
    repeats = df_raw.groupby(FEATURES + [TARGET]).size()
    return {
        "n_baris_mentah": int(len(df_raw)),
        "n_atribut": len(FEATURES),
        "n_missing": int(df_raw.isna().sum().sum()),
        "n_duplikat_eksak": int(df_raw.duplicated().sum()),
        "persen_duplikat": round(100 * df_raw.duplicated().mean(), 2),
        "n_baris_terlibat_duplikasi": int(df_raw.duplicated(keep=False).sum()),
        "pengulangan_maksimum": int(repeats.max()),
        "n_baris_unik": int(len(dedup)),
        "n_vektor_konflik": int(len(conflicts)),
        "n_baris_unik_konflik": int(len(conflict_rows)),
        "n_baris_tanpa_konflik": int(len(dedup) - len(conflict_rows)),
        "nilai_tidak_wajar_mentah": {c: int(bad[c].sum()) for c in FEATURES if bad[c].any()},
        "nilai_tidak_wajar_unik": {
            c: int(implausible_mask(dedup)[c].sum())
            for c in FEATURES
            if implausible_mask(dedup)[c].any()
        },
        "distribusi_kelas_mentah": {k: int(df_raw[TARGET].eq(k).sum()) for k in CLASS_ORDER},
        "distribusi_kelas_unik": {k: int(dedup[TARGET].eq(k).sum()) for k in CLASS_ORDER},
        "usia_di_bawah_15_unik": int((dedup.Age < 15).sum()),
        "usia_di_atas_50_unik": int((dedup.Age > 50).sum()),
        "sistolik_tidak_lebih_besar_dari_diastolik": int(
            (df_raw.SystolicBP <= df_raw.DiastolicBP).sum()
        ),
    }
