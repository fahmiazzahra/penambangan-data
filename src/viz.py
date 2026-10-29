"""Gaya grafik bersama (matplotlib) agar semua gambar konsisten.

Warna mengikuti palet tervalidasi (lolos uji buta warna):
- kelas risiko  : satu hue biru bertingkat (ordinal, terang -> gelap = rendah -> tinggi)
- skenario A/B/C: slot kategorikal 1-3 (biru, oranye, aqua), selalu disertai label nilai
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

RISK_COLORS = {"low risk": "#86b6ef", "mid risk": "#2a78d6", "high risk": "#104281"}
SCENARIO_COLORS = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
SEQ_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "biru", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
)
DIV_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "biru_merah", ["#256abf", "#86b6ef", "#f0efec", "#f19a99", "#c23a3a"]
)


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 200,
            "legend.frameon": False,
        }
    )


def clean_axes(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x" if grid_axis == "y" else "y", visible=False)
    ax.tick_params(length=0)


def label_bars(ax, bars, fmt="{:.0f}", pad=2, fontsize=8):
    for b in bars:
        h = b.get_height()
        ax.annotate(
            fmt.format(h),
            (b.get_x() + b.get_width() / 2, h),
            xytext=(0, pad),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=INK_2,
        )
