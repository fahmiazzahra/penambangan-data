"""Pengujian fungsional (black-box) aplikasi Streamlit memakai streamlit.testing.

Setiap kasus uji mensimulasikan input pengguna lalu memeriksa keluaran yang tampil.
Hasil ditulis ke reports/results/uji_aplikasi.csv sebagai bukti pengujian (kriteria
keberhasilan no. 3 pada proposal).

Jalankan dari folder proyek:  python tests/uji_aplikasi.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT))

results = []


def record(tc, skenario, harapan, ok, catatan=""):
    results.append({"ID": tc, "Skenario": skenario, "Hasil yang diharapkan": harapan,
                    "Status": "LULUS" if ok else "GAGAL", "Catatan": catatan})
    print(f"[{'LULUS' if ok else 'GAGAL'}] {tc} {skenario} {catatan}")


def fresh():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    return at


def ni(at, label):
    return next(w for w in at.number_input if w.label == label)


def alerts(at):
    return [a.value for a in list(at.success) + list(at.warning) + list(at.error)]


def prediction_text(at):
    return next((v for v in alerts(at) if v.startswith("**Prediksi:")), None)


def set_patient(at, age, sbp, dbp, bs, temp_c, hr):
    ni(at, "Usia (tahun)").set_value(age)
    ni(at, "Tekanan darah sistolik (mmHg)").set_value(sbp)
    ni(at, "Tekanan darah diastolik (mmHg)").set_value(dbp)
    ni(at, "Gula darah (mmol/L)").set_value(bs)
    ni(at, "Suhu tubuh (°C)").set_value(temp_c)
    ni(at, "Denyut jantung (bpm)").set_value(hr)
    return at.run()


# ---------------------------------------------------------------------------- TC01
at = fresh()
ok = not at.exception
pages_ok = []
for page in ["views/eksplorasi.py", "views/evaluasi.py", "views/tentang.py", "views/prediksi.py"]:
    at.switch_page(page).run()
    pages_ok.append((page, not at.exception))
ok = ok and all(p[1] for p in pages_ok)
record("TC01", "Membuka keempat halaman aplikasi", "Semua halaman tampil tanpa error", ok,
       "; ".join(f"{p}: {'ok' if s else 'error'}" for p, s in pages_ok))

# ---------------------------------------------------------------------------- TC02
at = fresh()
for code in ["DT", "NB-G", "NB-K", "KNN"]:
    at.selectbox[0].set_value(code).run()
    pred = prediction_text(at)
    exp_ok = len(at.expander) >= 2 and not at.exception
    record("TC02-" + code, f"Input bawaan dengan algoritma {code}",
           "Prediksi, probabilitas, dan penjelasan tampil", bool(pred) and exp_ok, pred or "tidak ada prediksi")

# ---------------------------------------------------------------------------- TC03
at = fresh()
at = set_patient(at, 35, 140, 90, 13.0, 36.7, 76)
pred = prediction_text(at) or ""
flags = [i.value for i in at.info]
record("TC03", "Pasien berisiko tinggi (35 th, 140/90, gula 13 mmol/L)",
       "Prediksi Risiko Tinggi + catatan hipertensi", "Risiko Tinggi" in pred and any("140/90" in f for f in flags), pred)

# ---------------------------------------------------------------------------- TC04
at = fresh()
ni(at, "Tekanan darah sistolik (mmHg)").set_value(80)
ni(at, "Tekanan darah diastolik (mmHg)").set_value(90)
at.run()
err = [e.value for e in at.error]
record("TC04", "Sistolik (80) ≤ diastolik (90)", "Pesan validasi muncul, prediksi tidak dibuat",
       any("lebih besar" in e for e in err) and prediction_text(at) is None, "; ".join(err))

# ---------------------------------------------------------------------------- TC05
at = fresh()
at = set_patient(at, 35, 140, 90, 13.0, 37.0, 76)
p_mmol = prediction_text(at)
at.button_group[0].set_value("mg/dL").run()
at.button_group[1].set_value("°F").run()
ni(at, "Gula darah (mg/dL)").set_value(234.0)
ni(at, "Suhu tubuh (°F)").set_value(98.6)
at.run()
p_mgdl = prediction_text(at)
cap = " ".join(c.value for c in at.caption)
record("TC05", "Konversi satuan: 234 mg/dL = 13 mmol/L dan 98,6 °F = 37 °C",
       "Prediksi sama dan nilai terkirim 13.00 mmol/L", p_mmol == p_mgdl and "13.00 mmol/L" in cap, f"{p_mmol} | {p_mgdl}")

# ---------------------------------------------------------------------------- TC06
at = fresh()
ni(at, "Gula darah (mmol/L)").set_value(25.0)
at.run()
warn = [w.value for w in at.warning]
record("TC06", "Gula darah 25 mmol/L (di atas data latih, maks 19)", "Peringatan di luar rentang data latih",
       any("di luar rentang data latih" in w for w in warn))

# ---------------------------------------------------------------------------- TC07
at = fresh()
csv = (
    "Age,SystolicBP,DiastolicBP,BS,BodyTemp,HeartRate\n"
    "25,120,80,7.5,98,76\n"
    "35,140,90,13,98,70\n"
    "40,80,90,7,98,70\n"
    "abc,120,80,7,98,70\n"
).encode()
at.file_uploader[0].upload("uji.csv", csv, "text/csv")
at.run()
# Tabel hasil batch = tabel yang memuat kolom input (Age) sekaligus kolom Prediksi.
dfs = [d.value for d in at.dataframe if {"Age", "Prediksi"} <= set(getattr(d.value, "columns", []))]
warn = [w.value for w in at.warning]
ok = bool(dfs) and len(dfs[0]) == 2 and any("2 baris dilewati" in w for w in warn)
record("TC07", "Unggah CSV 4 baris (2 valid, 2 tidak valid)", "2 baris diprediksi, 2 baris dilewati dengan peringatan", ok,
       f"baris hasil = {len(dfs[0]) if dfs else 0}")

# ---------------------------------------------------------------------------- TC08
from common import load_models, predict_one  # noqa: E402

models = load_models()
x = np.array([35, 140, 90, 13.0, 98.6, 76], dtype=float)
times = {}
for code, m in models.items():
    t0 = time.perf_counter()
    for _ in range(20):
        predict_one(m, x)
    times[code] = (time.perf_counter() - t0) / 20
record("TC08", "Waktu respons prediksi satu pasien", "< 1 detik untuk setiap algoritma",
       max(times.values()) < 1.0, ", ".join(f"{k}: {1000 * v:.1f} ms" for k, v in times.items()))

out = pd.DataFrame(results)
out.to_csv(ROOT / "reports" / "results" / "uji_aplikasi.csv", index=False)
print(f"\n{(out.Status == 'LULUS').sum()}/{len(out)} kasus uji lulus")
sys.exit(0 if (out.Status == "LULUS").all() else 1)
