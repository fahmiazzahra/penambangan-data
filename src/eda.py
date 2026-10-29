"""Eksplorasi data (EDA): profil kualitas data, statistik deskriptif, dan grafik.

Jalankan dari folder proyek:  python -m src.eda
"""
import json

import numpy as np
import pandas as pd

from . import viz
from .config import CLASS_LABEL_ID, CLASS_ORDER, FEATURE_INFO, FEATURES, FIGURES_DIR, RESULTS_DIR, TARGET
from .data import load_raw, profile

plt = viz.plt


def fig_class_distribution(raw: pd.DataFrame, dedup: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    x = np.arange(len(CLASS_ORDER))
    w = 0.34
    series = [
        ("Data mentah (n = %d)" % len(raw), raw, viz.SCENARIO_COLORS["A"], -w / 2 - 0.01),
        ("Setelah deduplikasi (n = %d)" % len(dedup), dedup, viz.SCENARIO_COLORS["B"], w / 2 + 0.01),
    ]
    for label, d, color, off in series:
        counts = d[TARGET].value_counts().reindex(CLASS_ORDER)
        bars = ax.bar(x + off, counts.values, width=w, color=color, label=label)
        for b, n in zip(bars, counts.values):
            ax.annotate(
                f"{n} ({100 * n / len(d):.1f}%)",
                (b.get_x() + b.get_width() / 2, n),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=viz.INK_2,
            )
    ax.set_xticks(x, [CLASS_LABEL_ID[c] for c in CLASS_ORDER])
    ax.set_ylabel("Jumlah baris")
    ax.set_ylim(0, 470)
    ax.legend(loc="upper right", fontsize=8)
    viz.clean_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_distribusi_kelas.png")
    plt.close(fig)


def fig_feature_boxplots(dedup: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    for ax, feat in zip(axes.ravel(), FEATURES):
        data = [dedup.loc[dedup[TARGET] == c, feat].values for c in CLASS_ORDER]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.5,
            medianprops={"color": "#ffffff", "linewidth": 1.4},
            whiskerprops={"color": viz.MUTED},
            capprops={"color": viz.MUTED},
            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": viz.MUTED, "markeredgewidth": 0},
        )
        for patch, c in zip(bp["boxes"], CLASS_ORDER):
            patch.set_facecolor(viz.RISK_COLORS[c])
            patch.set_edgecolor(viz.RISK_COLORS[c])
        info = FEATURE_INFO[feat]
        ax.set_title(f"{info['nama']} ({info['satuan']})", fontsize=8.5)
        ax.set_xticks([1, 2, 3], ["Rendah", "Sedang", "Tinggi"], fontsize=8)
        viz.clean_axes(ax)
    fig.suptitle("Sebaran atribut per tingkat risiko (data unik, n = %d)" % len(dedup), fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_boxplot_atribut.png")
    plt.close(fig)


def fig_correlation(dedup: pd.DataFrame):
    d = dedup.copy()
    d["Risiko (ordinal)"] = d[TARGET].map({c: i for i, c in enumerate(CLASS_ORDER)})
    corr = d[FEATURES + ["Risiko (ordinal)"]].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    im = ax.imshow(corr.values, cmap=viz.DIV_CMAP, vmin=-1, vmax=1)
    labels = [FEATURE_INFO[f]["nama"] for f in FEATURES] + ["Tingkat risiko"]
    ax.set_xticks(range(len(labels)), labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="#ffffff" if abs(v) > 0.6 else viz.INK)
    ax.grid(False)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)
    ax.set_title("Korelasi Spearman (data unik)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig_korelasi.png")
    plt.close(fig)
    return corr


def main():
    viz.apply_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    dedup = raw.drop_duplicates()

    prof = profile(raw)
    (RESULTS_DIR / "data_profile.json").write_text(json.dumps(prof, indent=2, ensure_ascii=False), encoding="utf-8")

    desc = pd.concat(
        {"mentah": raw[FEATURES].describe().T, "unik": dedup[FEATURES].describe().T}, axis=0
    ).round(2)
    desc.to_csv(RESULTS_DIR / "statistik_deskriptif.csv")

    per_class = dedup.groupby(TARGET)[FEATURES].median().reindex(CLASS_ORDER)
    per_class.to_csv(RESULTS_DIR / "median_per_kelas_unik.csv")

    fig_class_distribution(raw, dedup)
    fig_feature_boxplots(dedup)
    corr = fig_correlation(dedup)
    corr.round(3).to_csv(RESULTS_DIR / "korelasi_spearman_unik.csv")

    print(json.dumps(prof, indent=2, ensure_ascii=False))
    print(desc)
    print(per_class)


if __name__ == "__main__":
    main()
