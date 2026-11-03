"""Uji pengaruh tuning: parameter bawaan vs hasil tuning untuk setiap algoritma.

Memakai skor per fold dari src.experiment (tidak melatih ulang model), dengan corrected
resampled t-test dan koreksi Holm yang sama seperti perbandingan antaralgoritma.

Jalankan dari folder proyek setelah src.experiment:  python -m src.uji_tuning
"""
import pandas as pd

from .config import RESULTS_DIR
from .experiment import MODEL_ORDER, corrected_ttest, holm


def tuning_tests(df: pd.DataFrame, scenario: str, metric: str) -> pd.DataFrame:
    sub = df[df.skenario == scenario].sort_values(["repeat", "fold"])
    rows = []
    for m in MODEL_ORDER:
        if m == "BASELINE":
            continue
        g = sub[sub.model == m]
        default = g[g.konfigurasi == "default"][metric].to_numpy()
        tuned = g[g.konfigurasi == "tuned"][metric].to_numpy()
        t, p = corrected_ttest(tuned, default, g.n_latih.mean(), g.n_uji.mean())
        rows.append({"skenario": scenario, "metrik": metric, "model": m, "bawaan": default.mean(), "tuning": tuned.mean(),
                     "selisih_tuning_bawaan": tuned.mean() - default.mean(), "t": t, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out.p.to_numpy())
    out["signifikan_0.05"] = out.p_holm < 0.05
    return out


def main():
    df = pd.read_csv(RESULTS_DIR / "skor_per_fold.csv")
    hasil = pd.concat([tuning_tests(df, s, m) for s in ["A", "B", "C"] for m in ["f1_macro", "recall_high"]])
    hasil.to_csv(RESULTS_DIR / "uji_tuning.csv", index=False)
    print(hasil.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
