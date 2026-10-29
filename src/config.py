"""Konfigurasi bersama: path, nama atribut, label kelas, dan aturan validasi data."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "maternal_health_risk.csv"
MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "reports" / "figures"
RESULTS_DIR = ROOT / "reports" / "results"

RANDOM_STATE = 42

FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
TARGET = "RiskLevel"

# Urutan kelas dari risiko terendah ke tertinggi (dipakai untuk tabel, grafik,
# dan aturan "seri -> pilih risiko lebih tinggi").
CLASS_ORDER = ["low risk", "mid risk", "high risk"]
CLASS_LABEL_ID = {
    "low risk": "Risiko Rendah",
    "mid risk": "Risiko Sedang",
    "high risk": "Risiko Tinggi",
}

FEATURE_INFO = {
    "Age": {"nama": "Usia", "satuan": "tahun"},
    "SystolicBP": {"nama": "Tekanan darah sistolik", "satuan": "mmHg"},
    "DiastolicBP": {"nama": "Tekanan darah diastolik", "satuan": "mmHg"},
    "BS": {"nama": "Kadar gula darah", "satuan": "mmol/L"},
    "BodyTemp": {"nama": "Suhu tubuh", "satuan": "°F"},
    "HeartRate": {"nama": "Denyut jantung", "satuan": "bpm"},
}

# Rentang nilai yang masih masuk akal secara fisiologis. Nilai di luar rentang
# ini dianggap tidak valid (mis. HeartRate = 7 bpm) lalu diubah menjadi NaN dan
# diimputasi median di dalam pipeline, sehingga median hanya dihitung dari data latih.
PLAUSIBLE_RANGE = {
    "Age": (10, 70),
    "SystolicBP": (60, 220),
    "DiastolicBP": (30, 140),
    "BS": (2.0, 30.0),
    "BodyTemp": (93.0, 108.0),
    "HeartRate": (40, 200),
}
