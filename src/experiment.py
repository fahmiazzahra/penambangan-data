"""Eksperimen utama: perbandingan NB, DT, dan KNN pada tiga skenario data.

Protokol
- Validasi luar  : 10x10 repeated stratified (group) k-fold (Bouckaert & Frank, 2004).
- Tuning         : grid search di dalam setiap fold latih (nested CV, 5 fold, skor macro-F1)
                   agar estimasi kinerja tidak bias oleh pemilihan hyperparameter
                   (Varma & Simon, 2006).
- Konfigurasi    : "default" (parameter bawaan) dan "tuned" (hasil grid search).
- Uji statistik  : corrected resampled t-test (Nadeau & Bengio, 2003) + koreksi Holm.

Jalankan dari folder proyek:  python -m src.experiment  [--repeats 10] [--jobs -1]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from . import viz
from .config import CLASS_LABEL_ID, CLASS_ORDER, FEATURES, FIGURES_DIR, RANDOM_STATE, RESULTS_DIR, TARGET
from .data import build_scenarios
from .models import build_models

N_OUTER_FOLDS = 10
N_INNER_FOLDS = 5
MODEL_ORDER = ["BASELINE", "NB-G", "NB-K", "DT", "KNN"]


# ----------------------------------------------------------------------------- metrik
def model_size(est) -> int:
    """Jumlah parameter/nilai yang disimpan model (ukuran kompleksitas ruang)."""
    pipe = est.best_estimator_ if hasattr(est, "best_estimator_") else est
    clf = pipe.named_steps["clf"]
    name = type(clf).__name__
    if name == "DecisionTreeClassifier":
        return int(clf.tree_.node_count)
    if name == "KNeighborsClassifier":
        return int(clf.n_samples_fit_ * clf.n_features_in_)
    if name == "GaussianNB":
        return int(clf.theta_.size + clf.var_.size + clf.class_prior_.size)
    if name == "CategoricalNB":
        return int(sum(a.size for a in clf.feature_log_prob_) + clf.class_count_.size)
    return int(clf.n_classes_ if hasattr(clf, "n_classes_") else 0)


def compute_metrics(y_true, y_pred, proba, classes) -> dict:
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_ovr_macro": roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes),
    }
    rec = recall_score(y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0)
    prec = precision_score(y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=CLASS_ORDER, average=None, zero_division=0)
    for i, c in enumerate(CLASS_ORDER):
        key = c.split()[0]
        m[f"recall_{key}"] = rec[i]
        m[f"precision_{key}"] = prec[i]
        m[f"f1_{key}"] = f1[i]
    return m


# ----------------------------------------------------------------------------- satu fold
def run_fold(scn, rep, fold, code, config, spec, X, y, groups, tr, te):
    warnings.filterwarnings("ignore")
    X_tr, X_te, y_tr, y_te = X[tr], X[te], y[tr], y[te]
    g_tr = groups[tr]

    if config == "tuned":
        inner = StratifiedGroupKFold(N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE + 1000 * rep + fold)
        est = GridSearchCV(clone(spec["pipeline"]), spec["grid"], cv=inner, scoring="f1_macro", n_jobs=1)
        t0 = time.perf_counter()
        est.fit(X_tr, y_tr, groups=g_tr)
        tuning_time = time.perf_counter() - t0
        fit_time = est.refit_time_
        best_params = {k: (type(v).__name__ if hasattr(v, "fit") else v) for k, v in est.best_params_.items()}
    else:
        est = clone(spec["pipeline"])
        t0 = time.perf_counter()
        est.fit(X_tr, y_tr)
        fit_time = time.perf_counter() - t0
        tuning_time = 0.0
        best_params = {}

    t0 = time.perf_counter()
    y_pred = est.predict(X_te)
    proba = est.predict_proba(X_te)
    pred_time = time.perf_counter() - t0

    row = {
        "skenario": scn,
        "repeat": rep,
        "fold": fold,
        "model": code,
        "konfigurasi": config,
        "n_latih": len(tr),
        "n_uji": len(te),
        **compute_metrics(y_te, y_pred, proba, est.classes_),
        "waktu_latih_ms": 1000 * fit_time,
        "waktu_tuning_s": tuning_time,
        "waktu_prediksi_ms_per_1000": 1000 * pred_time / len(te) * 1000,
        "ukuran_model": model_size(est),
        "best_params": json.dumps(best_params, default=str),
    }
    cm = confusion_matrix(y_te, y_pred, labels=CLASS_ORDER)
    row["cm"] = json.dumps(cm.tolist())
    return row


def leak_overlap(X: pd.DataFrame, groups: pd.Series, y: pd.Series, repeats: int) -> dict:
    """Persentase baris uji yang punya salinan identik (atribut sama) di data latih."""
    keys = X.astype(str).agg("|".join, axis=1)
    rates = []
    for rep in range(repeats):
        cv = StratifiedGroupKFold(N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE + rep)
        for tr, te in cv.split(X, y, groups):
            train_keys = set(keys.iloc[tr])
            rates.append(keys.iloc[te].isin(train_keys).mean())
    return {"rata2_persen": 100 * float(np.mean(rates)), "min_persen": 100 * float(np.min(rates)),
            "maks_persen": 100 * float(np.max(rates))}


# ----------------------------------------------------------------------------- statistik
def corrected_ttest(a, b, n_train, n_test):
    """Corrected resampled t-test (Nadeau & Bengio, 2003) untuk skor k-fold berulang."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / n + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return float(t), float(p)


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def pairwise_tests(df: pd.DataFrame, scenario: str, metric: str) -> pd.DataFrame:
    sub = df[(df.skenario == scenario) & ((df.konfigurasi == "tuned") | (df.model == "BASELINE"))]
    sub = sub.sort_values(["repeat", "fold"])
    scores = {m: g[metric].to_numpy() for m, g in sub.groupby("model")}
    n_train = sub.n_latih.mean()
    n_test = sub.n_uji.mean()
    rows = []
    for m1, m2 in combinations([m for m in MODEL_ORDER if m in scores], 2):
        t, p = corrected_ttest(scores[m1], scores[m2], n_train, n_test)
        rows.append({"skenario": scenario, "metrik": metric, "model_1": m1, "model_2": m2,
                     "selisih_rata2": scores[m1].mean() - scores[m2].mean(), "t": t, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out["signifikan_0.05"] = out.p_holm < 0.05
    return out


# ----------------------------------------------------------------------------- grafik
def fig_scenarios(summary: pd.DataFrame, metric: str, fname: str, ylabel: str):
    s = summary[(summary.konfigurasi == "tuned") | (summary.model == "BASELINE")]
    models = [m for m in MODEL_ORDER]
    labels = {"BASELINE": "Baseline", "NB-G": "NB Gaussian", "NB-K": "NB Kategorikal", "DT": "Decision Tree",
              "KNN": "KNN"}
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    x = np.arange(len(models))
    w = 0.26
    for i, scn in enumerate(["A", "B", "C"]):
        vals = [s[(s.skenario == scn) & (s.model == m)][f"{metric}_mean"].iloc[0] for m in models]
        sds = [s[(s.skenario == scn) & (s.model == m)][f"{metric}_std"].iloc[0] for m in models]
        bars = ax.bar(x + (i - 1) * (w + 0.01), vals, width=w, color=viz.SCENARIO_COLORS[scn],
                      label={"A": "A: data mentah", "B": "B: deduplikasi (usulan)", "C": "C: B tanpa label konflik"}[scn],
                      yerr=sds, error_kw={"elinewidth": 0.8, "ecolor": viz.MUTED, "capsize": 0})
        # Label diletakkan di atas ujung error bar agar tidak bertumpuk.
        for b, v, sd in zip(bars, vals, sds):
            ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v + sd), xytext=(0, 2),
                        textcoords="offset points", ha="center", va="bottom", fontsize=6.5, color=viz.INK_2)
    ax.set_xticks(x, [labels[m] for m in models])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", ncol=3, fontsize=7.5, bbox_to_anchor=(0, 1.13))
    viz.clean_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname)
    plt.close(fig)


def fig_confusions(df: pd.DataFrame, scenario: str, fname: str):
    models = ["NB-G", "NB-K", "DT", "KNN"]
    titles = {"NB-G": "NB Gaussian", "NB-K": "NB Kategorikal", "DT": "Decision Tree", "KNN": "KNN"}
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.5))
    short = ["Rendah", "Sedang", "Tinggi"]
    for ax, m in zip(axes, models):
        sub = df[(df.skenario == scenario) & (df.model == m) & (df.konfigurasi == "tuned")]
        cm = np.sum([np.array(json.loads(c)) for c in sub.cm], axis=0)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap=viz.SEQ_CMAP, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if cmn[i, j] > 0.55 else viz.INK)
        ax.set_xticks(range(3), short, fontsize=7.5)
        ax.set_yticks(range(3), short if m == models[0] else [], fontsize=7.5)
        ax.set_title(titles[m], fontsize=9)
        ax.set_xlabel("Prediksi", fontsize=7.5)
        if m == models[0]:
            ax.set_ylabel("Aktual", fontsize=7.5)
        ax.grid(False)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / fname)
    plt.close(fig)


def make_plots(df: pd.DataFrame, summary: pd.DataFrame):
    viz.apply_style()
    fig_scenarios(summary, "f1_macro", "fig_perbandingan_f1.png", "Macro-F1 (rata-rata ± SD)")
    fig_scenarios(summary, "recall_high", "fig_perbandingan_recall_tinggi.png", "Recall kelas Risiko Tinggi")
    fig_confusions(df, "B", "fig_confusion_B.png")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--plots-only", action="store_true", help="gambar ulang grafik dari CSV hasil yang sudah ada")
    args = ap.parse_args()

    if args.plots_only:
        make_plots(pd.read_csv(RESULTS_DIR / "skor_per_fold.csv"), pd.read_csv(RESULTS_DIR / "ringkasan.csv"))
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    scenarios = build_scenarios()
    models = build_models()

    tasks = []
    for scn, sd in scenarios.items():
        X = sd["data"][FEATURES].to_numpy(dtype=float)
        y = sd["data"][TARGET].to_numpy()
        groups = sd["groups"].to_numpy()
        for rep in range(args.repeats):
            cv = StratifiedGroupKFold(N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE + rep)
            for fold, (tr, te) in enumerate(cv.split(X, y, groups)):
                for code, spec in models.items():
                    configs = ["default"] if spec["grid"] is None else ["default", "tuned"]
                    for config in configs:
                        tasks.append((scn, rep, fold, code, config, spec, X, y, groups, tr, te))

    print(f"Total tugas evaluasi: {len(tasks)}")
    t0 = time.time()
    rows = Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_fold)(*t) for t in tasks)
    print(f"Selesai dalam {time.time() - t0:.1f} detik")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "skor_per_fold.csv", index=False)

    metric_cols = [
        "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro", "auc_ovr_macro",
        "recall_low", "recall_mid", "recall_high", "precision_high", "f1_high",
        "waktu_latih_ms", "waktu_prediksi_ms_per_1000", "ukuran_model",
    ]
    summary = df.groupby(["skenario", "model", "konfigurasi"])[metric_cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(RESULTS_DIR / "ringkasan.csv", index=False)

    # Inflasi kinerja akibat duplikasi: selisih skenario A terhadap B (model tuned).
    tuned = summary[(summary.konfigurasi == "tuned") | (summary.model == "BASELINE")]
    infl = tuned.pivot(index="model", columns="skenario", values=["accuracy_mean", "f1_macro_mean", "recall_high_mean"])
    infl.columns = [f"{a}_{b}" for a, b in infl.columns]
    for met in ["accuracy_mean", "f1_macro_mean", "recall_high_mean"]:
        infl[f"inflasi_{met}_A_minus_B"] = infl[f"{met}_A"] - infl[f"{met}_B"]
    infl.reindex(MODEL_ORDER).to_csv(RESULTS_DIR / "inflasi_A_vs_B.csv")

    tests = pd.concat(
        [pairwise_tests(df, s, met) for s in ["A", "B", "C"] for met in ["f1_macro", "recall_high"]],
        ignore_index=True,
    )
    tests.to_csv(RESULTS_DIR / "uji_statistik.csv", index=False)

    params = (
        df[(df.konfigurasi == "tuned")]
        .groupby(["skenario", "model"])
        .best_params.apply(lambda s: Counter(s).most_common(3))
        .reset_index()
    )
    params.to_json(RESULTS_DIR / "parameter_terpilih.json", orient="records", indent=2, force_ascii=False)

    overlap = {
        scn: leak_overlap(sd["data"][FEATURES], sd["groups"], sd["data"][TARGET], args.repeats)
        for scn, sd in scenarios.items()
    }
    meta = {
        "repeats": args.repeats,
        "outer_folds": N_OUTER_FOLDS,
        "inner_folds": N_INNER_FOLDS,
        "n_per_skenario": {s: len(sd["data"]) for s, sd in scenarios.items()},
        "baris_uji_dengan_salinan_di_data_latih": overlap,
    }
    (RESULTS_DIR / "meta_eksperimen.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    make_plots(df, summary)

    pd.set_option("display.width", 200)
    show = summary[["skenario", "model", "konfigurasi", "accuracy_mean", "f1_macro_mean", "f1_macro_std",
                    "recall_high_mean", "auc_ovr_macro_mean", "waktu_latih_ms_mean", "ukuran_model_mean"]]
    print(show.round(3).to_string(index=False))
    print(json.dumps(meta, indent=2))
    print(tests[tests.metrik == "f1_macro"].round(4).to_string(index=False))


plt = viz.plt

if __name__ == "__main__":
    main()
