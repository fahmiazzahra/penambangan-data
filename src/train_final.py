"""Melatih model final (skenario B: data unik) untuk dipakai aplikasi Streamlit.

Hyperparameter dipilih dengan grid search + 10-fold stratified group CV pada seluruh
data unik, lalu model dilatih ulang pada seluruh data tersebut. Estimasi kinerja yang
jujur TIDAK diambil dari skor grid search ini (karena optimistis), melainkan dari hasil
nested CV di reports/results/ringkasan.csv (src.experiment).

Jalankan dari folder proyek:  python -m src.train_final
"""
from __future__ import annotations

import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.tree import export_text

from .config import FEATURE_INFO, FEATURES, MODELS_DIR, RANDOM_STATE, RAW_CSV, RESULTS_DIR, TARGET
from .data import build_scenarios
from .models import build_models

MODEL_CODES = ["NB-G", "NB-K", "DT", "KNN"]


def _jsonable(v):
    if hasattr(v, "fit"):
        return type(v).__name__
    if isinstance(v, np.generic):
        return v.item()
    return v


def nested_cv_summary() -> dict:
    path = RESULTS_DIR / "ringkasan.csv"
    if not path.exists():
        return {}
    s = pd.read_csv(path)
    s = s[(s.skenario == "B") & (s.konfigurasi == "tuned")]
    cols = ["accuracy", "balanced_accuracy", "f1_macro", "recall_high", "recall_mid", "recall_low",
            "precision_high", "auc_ovr_macro"]
    return {
        row.model: {c: {"mean": round(getattr(row, f"{c}_mean"), 4), "std": round(getattr(row, f"{c}_std"), 4)}
                    for c in cols}
        for row in s.itertuples()
    }


def choose_recommended(nested: dict, meta: dict) -> tuple[str, str]:
    """Aturan pemilihan model (ditetapkan di proposal, Subbab 5.3):
    1) ambil model dengan macro-F1 nested CV tertinggi pada skenario B;
    2) model lain yang selisih macro-F1-nya tidak signifikan (p Holm >= 0,05) ikut menjadi kandidat;
    3) dari kandidat, pilih recall kelas Risiko Tinggi tertinggi (kesalahan paling berbahaya)."""
    tests_path = RESULTS_DIR / "uji_statistik.csv"
    if not nested or not tests_path.exists():
        best = max(meta["model"], key=lambda m: meta["model"][m]["skor_cv_grid_search_f1_macro"])
        return best, "skor grid search (nested CV belum dijalankan)"
    top = max(nested, key=lambda m: nested[m]["f1_macro"]["mean"])
    t = pd.read_csv(tests_path)
    t = t[(t.skenario == "B") & (t.metrik == "f1_macro")]
    candidates = [top]
    for m in nested:
        if m == top:
            continue
        row = t[((t.model_1 == top) & (t.model_2 == m)) | ((t.model_1 == m) & (t.model_2 == top))]
        if not row.empty and row.p_holm.iloc[0] >= 0.05:
            candidates.append(m)
    best = max(candidates, key=lambda m: nested[m]["recall_high"]["mean"])
    return best, (
        f"macro-F1 tertinggi pada skenario B adalah {top}; kandidat yang tidak berbeda signifikan "
        f"(p Holm >= 0,05): {', '.join(candidates)}; dipilih recall Risiko Tinggi tertinggi"
    )


def main():
    warnings.filterwarnings("ignore")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    sd = build_scenarios()["B"]
    X = sd["data"][FEATURES].to_numpy(dtype=float)
    y = sd["data"][TARGET].to_numpy()
    groups = sd["groups"].to_numpy()
    specs = build_models()

    nested = nested_cv_summary()
    meta = {
        "dibuat_pada": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "file": RAW_CSV.name,
            "sha256": hashlib.sha256(RAW_CSV.read_bytes()).hexdigest(),
            "n_latih": int(len(y)),
            "distribusi_kelas": {k: int(v) for k, v in pd.Series(y).value_counts().items()},
            "rentang_atribut": {
                f: {"min": float(X[:, i].min()), "max": float(X[:, i].max())} for i, f in enumerate(FEATURES)
            },
        },
        "versi": {"python": platform.python_version(), "scikit-learn": sklearn.__version__},
        "fitur": FEATURES,
        "model": {},
    }

    cv = StratifiedGroupKFold(10, shuffle=True, random_state=RANDOM_STATE)
    for code in MODEL_CODES:
        spec = specs[code]
        gs = GridSearchCV(spec["pipeline"], spec["grid"], cv=cv, scoring="f1_macro", n_jobs=-1)
        gs.fit(X, y, groups=groups)
        joblib.dump(gs.best_estimator_, MODELS_DIR / f"{code}.joblib", compress=3)
        meta["model"][code] = {
            "nama": spec["nama"],
            "parameter_terbaik": {k: _jsonable(v) for k, v in gs.best_params_.items()},
            "skor_cv_grid_search_f1_macro": round(float(gs.best_score_), 4),
            "nested_cv_skenario_B": nested.get(code, {}),
        }
        print(f"{code:5s} f1(grid-search CV)={gs.best_score_:.3f}  params={gs.best_params_}")

        if code == "DT":
            tree = gs.best_estimator_.named_steps["clf"]
            names = [f"{FEATURE_INFO[f]['nama']} ({FEATURE_INFO[f]['satuan']})" for f in FEATURES]
            rules = export_text(tree, feature_names=names, show_weights=True, decimals=2)
            (MODELS_DIR / "aturan_decision_tree.txt").write_text(rules, encoding="utf-8")
            meta["model"][code]["kedalaman"] = int(tree.get_depth())
            meta["model"][code]["jumlah_daun"] = int(tree.get_n_leaves())

    best, reason = choose_recommended(nested, meta)
    meta["model_rekomendasi"] = best
    meta["dasar_rekomendasi"] = reason
    (MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Model rekomendasi:", best)


if __name__ == "__main__":
    main()
