"""Definisi pipeline model dan ruang hyperparameter untuk eksperimen.

Setiap pipeline diawali langkah pembersihan yang sama (nilai tidak wajar -> NaN ->
imputasi median), sehingga semua statistik pra-pemrosesan dihitung hanya dari data latih
pada setiap fold dan tidak membocorkan informasi data uji.

Pipeline menerima array numpy dengan urutan kolom = config.FEATURES.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import FEATURES, PLAUSIBLE_RANGE, RANDOM_STATE


_LO = np.array([PLAUSIBLE_RANGE[c][0] for c in FEATURES], dtype=float)
_HI = np.array([PLAUSIBLE_RANGE[c][1] for c in FEATURES], dtype=float)


class PlausibilityImputer(TransformerMixin, BaseEstimator):
    """Nilai kosong atau di luar rentang fisiologis (config.PLAUSIBLE_RANGE) diganti
    median kolom. Median dihitung saat `fit`, yaitu hanya dari data latih.

    Ditulis dengan numpy (bukan SimpleImputer + pandas) karena dipanggil puluhan ribu
    kali selama nested cross-validation.
    """

    def fit(self, X, y=None):
        X = self._invalid_to_nan(X)
        self.medians_ = np.nanmedian(X, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = self._invalid_to_nan(X)
        rows, cols = np.where(np.isnan(X))
        X[rows, cols] = self.medians_[cols]
        return X

    @staticmethod
    def _invalid_to_nan(X):
        X = np.array(X, dtype=float)  # selalu salinan, data asli tidak diubah
        X[(X < _LO) | (X > _HI)] = np.nan
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURES, dtype=object)


def _base_steps():
    return [("validasi_imputasi", PlausibilityImputer())]


def _catnb_grid():
    # min_categories disamakan dengan n_bins agar smoothing Laplace memakai
    # jumlah kategori yang benar untuk setiap konfigurasi diskretisasi.
    return [
        {
            "diskretisasi__n_bins": [b],
            "diskretisasi__strategy": ["quantile", "uniform", "kmeans"],
            "clf__min_categories": [b],
            "clf__alpha": [0.5, 1.0],
        }
        for b in (3, 4, 5, 6, 8, 10)
    ]


def build_models() -> dict[str, dict]:
    """Mengembalikan {kode: {nama, pipeline, grid}}.

    `pipeline` sudah memakai konfigurasi bawaan (default) sehingga bisa langsung
    dievaluasi sebagai konfigurasi "default"; `grid` dipakai untuk tuning.
    """
    return {
        "BASELINE": {
            "nama": "Baseline (kelas mayoritas)",
            "pipeline": Pipeline(_base_steps() + [("clf", DummyClassifier(strategy="most_frequent"))]),
            "grid": None,
        },
        "NB-G": {
            "nama": "Gaussian Naive Bayes",
            "pipeline": Pipeline(_base_steps() + [("clf", GaussianNB())]),
            "grid": {"clf__var_smoothing": np.logspace(-12, -2, 11)},
        },
        "NB-K": {
            "nama": "Categorical Naive Bayes (diskretisasi + Laplace)",
            "pipeline": Pipeline(
                _base_steps()
                + [
                    (
                        "diskretisasi",
                        KBinsDiscretizer(
                            n_bins=5,
                            encode="ordinal",
                            strategy="quantile",
                            quantile_method="averaged_inverted_cdf",
                        ),
                    ),
                    ("clf", CategoricalNB(alpha=1.0, min_categories=5)),
                ]
            ),
            "grid": _catnb_grid(),
        },
        "DT": {
            "nama": "Decision Tree (CART)",
            "pipeline": Pipeline(
                _base_steps() + [("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]
            ),
            "grid": {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [3, 4, 5, 6, 8, None],
                "clf__min_samples_leaf": [1, 3, 5, 10],
                "clf__class_weight": [None, "balanced"],
            },
        },
        "KNN": {
            "nama": "K-Nearest Neighbor",
            "pipeline": Pipeline(
                _base_steps() + [("skala", "passthrough"), ("clf", KNeighborsClassifier())]
            ),
            "grid": {
                "skala": ["passthrough", StandardScaler(), MinMaxScaler()],
                "clf__n_neighbors": list(range(1, 30, 2)),
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        },
    }


def grid_size(grid) -> int:
    if grid is None:
        return 1
    grids = grid if isinstance(grid, list) else [grid]
    return sum(int(np.prod([len(v) for v in g.values()])) for g in grids)
