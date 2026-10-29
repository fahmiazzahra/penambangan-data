"""Fungsi bersama untuk semua halaman aplikasi: memuat model, data, hasil eksperimen,
palet warna, dan penjelasan prediksi."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.preprocessing import MinMaxScaler, StandardScaler  # noqa: E402

from src.config import (  # noqa: E402
    CLASS_LABEL_ID,
    CLASS_ORDER,
    FEATURE_INFO,
    FEATURES,
    MODELS_DIR,
    PLAUSIBLE_RANGE,
    RESULTS_DIR,
    TARGET,
)
from src.data import build_scenarios, load_raw  # noqa: E402
from src.models import build_models  # noqa: E402

MODEL_CODES = ["NB-G", "NB-K", "DT", "KNN"]
MODEL_LABEL = {
    "BASELINE": "Baseline (kelas mayoritas)",
    "NB-G": "Naive Bayes (Gaussian)",
    "NB-K": "Naive Bayes (Kategorikal)",
    "DT": "Decision Tree",
    "KNN": "K-Nearest Neighbor",
}
SCENARIO_LABEL = {
    "A": "A · Data mentah",
    "B": "B · Deduplikasi (usulan)",
    "C": "C · B tanpa label konflik",
}
RISK_STATUS = {
    "low risk": {"warna": "green", "ikon": ":material/check_circle:"},
    "mid risk": {"warna": "orange", "ikon": ":material/warning:"},
    "high risk": {"warna": "red", "ikon": ":material/error:"},
}


# ----------------------------------------------------------------------------- tema & warna
def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def palette() -> dict:
    """Palet tervalidasi (lolos uji buta warna) untuk mode terang dan gelap."""
    if is_dark():
        return {
            "risk": {"low risk": "#184f95", "mid risk": "#3987e5", "high risk": "#9ec5f4"},
            "scenario": {"A": "#3987e5", "B": "#d95926", "C": "#199e70"},
            "seq": ["#1a1a19", "#104281", "#256abf", "#5598e7", "#9ec5f4"],
            "div": ["#3987e5", "#383835", "#e66767"],
        }
    return {
        "risk": {"low risk": "#86b6ef", "mid risk": "#2a78d6", "high risk": "#104281"},
        "scenario": {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"},
        "seq": ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#104281"],
        "div": ["#256abf", "#f0efec", "#c23a3a"],
    }


def feature_label(f: str) -> str:
    return f"{FEATURE_INFO[f]['nama']} ({FEATURE_INFO[f]['satuan']})"


# ----------------------------------------------------------------------------- data & model
@st.cache_data
def load_raw_df() -> pd.DataFrame:
    return load_raw()


@st.cache_data
def load_training_df() -> pd.DataFrame:
    """Data skenario B (unik) dengan urutan baris yang sama seperti saat model final dilatih."""
    return build_scenarios()["B"]["data"]


@st.cache_data
def load_metadata() -> dict:
    path = MODELS_DIR / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _params_from_meta(params: dict) -> dict:
    lookup = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler()}
    return {k: lookup.get(v, v) if isinstance(v, str) else v for k, v in params.items()}


@st.cache_resource
def load_models() -> dict:
    """Memuat model .joblib. Jika gagal (mis. beda versi scikit-learn di server),
    model dilatih ulang dari data dengan hyperparameter yang tersimpan di metadata."""
    meta = load_metadata()
    train = load_training_df()
    X = train[FEATURES].to_numpy(dtype=float)
    y = train[TARGET].to_numpy()
    specs = build_models()
    models = {}
    for code in MODEL_CODES:
        try:
            models[code] = joblib.load(MODELS_DIR / f"{code}.joblib")
        except Exception:
            params = meta.get("model", {}).get(code, {}).get("parameter_terbaik", {})
            pipe = specs[code]["pipeline"].set_params(**_params_from_meta(params))
            models[code] = pipe.fit(X, y)
    return models


@st.cache_data
def load_results() -> dict:
    def csv(name):
        p = RESULTS_DIR / name
        return pd.read_csv(p) if p.exists() else None

    def js(name):
        p = RESULTS_DIR / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    return {
        "ringkasan": csv("ringkasan.csv"),
        "per_fold": csv("skor_per_fold.csv"),
        "inflasi": csv("inflasi_A_vs_B.csv"),
        "uji": csv("uji_statistik.csv"),
        "profil": js("data_profile.json"),
        "meta": js("meta_eksperimen.json"),
        "parameter": js("parameter_terpilih.json"),
    }


# ----------------------------------------------------------------------------- prediksi
def predict_one(model, x: np.ndarray) -> tuple[str, dict]:
    proba = model.predict_proba(x.reshape(1, -1))[0]
    probs = {c: float(p) for c, p in zip(model.classes_, proba)}
    return model.predict(x.reshape(1, -1))[0], {c: probs[c] for c in CLASS_ORDER}


def out_of_range(x: np.ndarray) -> list[str]:
    """Atribut yang nilainya di luar rentang data latih (prediksi kurang dapat diandalkan)."""
    ranges = load_metadata().get("dataset", {}).get("rentang_atribut")
    if not ranges:
        train = load_training_df()
        ranges = {f: {"min": train[f].min(), "max": train[f].max()} for f in FEATURES}
    msgs = []
    for f, v in zip(FEATURES, x):
        lo, hi = ranges[f]["min"], ranges[f]["max"]
        if f == "HeartRate":
            lo = max(lo, PLAUSIBLE_RANGE[f][0])
        if not lo <= v <= hi:
            msgs.append(f"{feature_label(f)} = {v:g} di luar rentang data latih ({lo:g}–{hi:g})")
    return msgs


def explain_tree(model, x: np.ndarray) -> list[str]:
    pre, tree = model[:-1], model.named_steps["clf"]
    xt = pre.transform(x.reshape(1, -1))
    node_ids = tree.decision_path(xt).indices
    t = tree.tree_
    steps = []
    for node in node_ids:
        if t.children_left[node] == t.children_right[node]:  # daun
            continue
        f = t.feature[node]
        thr = t.threshold[node]
        val = xt[0, f]
        sign = "≤" if val <= thr else ">"
        steps.append(f"{feature_label(FEATURES[f])} = **{val:g}** {sign} {thr:.2f}")
    return steps


def explain_gaussian_nb(model, x: np.ndarray) -> pd.DataFrame:
    """Log-likelihood tiap atribut untuk tiap kelas: log P(x_i | kelas)."""
    pre, nb = model[:-1], model.named_steps["clf"]
    xt = pre.transform(x.reshape(1, -1))[0]
    ll = -0.5 * np.log(2 * np.pi * nb.var_) - (xt - nb.theta_) ** 2 / (2 * nb.var_)
    return _nb_table(ll, nb.classes_, x)


def explain_categorical_nb(model, x: np.ndarray) -> pd.DataFrame:
    pre, nb = model[:-1], model.named_steps["clf"]
    xt = pre.transform(x.reshape(1, -1))[0].astype(int)
    ll = np.array([[nb.feature_log_prob_[i][k, xt[i]] for i in range(len(FEATURES))]
                   for k in range(len(nb.classes_))])
    return _nb_table(ll, nb.classes_, x)


def _nb_table(ll: np.ndarray, classes, x) -> pd.DataFrame:
    df = pd.DataFrame(ll.T, columns=[CLASS_LABEL_ID[c] for c in classes])
    df = df[[CLASS_LABEL_ID[c] for c in CLASS_ORDER]]
    df.insert(0, "Nilai", x)
    df.insert(0, "Atribut", [feature_label(f) for f in FEATURES])
    df["Paling mendukung"] = df[[CLASS_LABEL_ID[c] for c in CLASS_ORDER]].idxmax(axis=1)
    return df


def explain_knn(model, x: np.ndarray) -> pd.DataFrame:
    pre, knn = model[:-1], model.named_steps["clf"]
    xt = pre.transform(x.reshape(1, -1))
    dist, idx = knn.kneighbors(xt)
    train = load_training_df().reset_index(drop=True)
    nb = train.iloc[idx[0]].copy()
    nb[TARGET] = nb[TARGET].map(CLASS_LABEL_ID)
    nb.insert(0, "Jarak", dist[0].round(3))
    return nb.rename(columns={f: feature_label(f) for f in FEATURES} | {TARGET: "Tingkat risiko"})
