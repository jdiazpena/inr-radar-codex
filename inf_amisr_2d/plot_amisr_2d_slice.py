# =========================
# USER SETTINGS
# =========================

CSV_FILE = "slice111748_h330_best_slice.csv"

ANNOTATE_BEAMS = True
SAVE_FIGURES = True

POINT_SIZE = 90
FIGSIZE = (8, 7)

USE_WIDGET_BACKEND = False   # True gives zoom/pan in VSCode Interactive if ipympl is installed


# =========================
# IMPORTS
# =========================

if USE_WIDGET_BACKEND:
    try:
        get_ipython().run_line_magic("matplotlib", "widget")
    except Exception:
        print("Could not activate widget backend. If needed, run: pip install ipympl")

import pandas as pd
import matplotlib.pyplot as plt


# =========================
# LOAD DATA
# =========================

df = pd.read_csv(CSV_FILE)

required = [
    "x_km",
    "y_km",
    "log10_Ne",
    "beamcode",
    "altitude_km",
    "dz_from_h0_km",
]

for col in required:
    if col not in df.columns:
        raise KeyError(f"Missing required column: {col}")

print("Loaded slice:")
print(f"  file: {CSV_FILE}")
print(f"  rows: {len(df)}")
print(f"  unique beams: {df['beamcode'].nunique()}")
print()
print(df[["x_km", "y_km", "altitude_km", "dz_from_h0_km", "Ne", "log10_Ne"]].describe())


# =========================
# PLOT 1: X-Y SCATTER
# =========================

fig, ax = plt.subplots(figsize=FIGSIZE)

sc = ax.scatter(
    df["x_km"],
    df["y_km"],
    c=df["log10_Ne"],
    s=POINT_SIZE,
    edgecolor="k",
    linewidth=0.4,
)

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label(r"$\log_{10}(N_e)$")

ax.set_xlabel("x east [km]")
ax.set_ylabel("y north [km]")
ax.set_title("Selected AMISR 2D slice")
ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)

if ANNOTATE_BEAMS:
    for _, row in df.iterrows():
        ax.text(
            row["x_km"],
            row["y_km"],
            str(int(row["beamcode"])),
            fontsize=7,
            ha="center",
            va="center",
        )

plt.tight_layout()

if SAVE_FIGURES:
    out_png = CSV_FILE.replace(".csv", "_xy_log10Ne.png")
    fig.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")

plt.show()


# =========================
# PLOT 2: ALTITUDE MISMATCH
# =========================

fig, ax = plt.subplots(figsize=(8, 4))

ax.scatter(
    df["beamcode"],
    df["dz_from_h0_km"],
    s=70,
    edgecolor="k",
    linewidth=0.4,
)

ax.axhline(0.0, linewidth=1)
ax.set_xlabel("beamcode")
ax.set_ylabel("selected altitude - target altitude [km]")
ax.set_title("Altitude mismatch of selected gates")
ax.grid(True, alpha=0.3)

plt.tight_layout()

if SAVE_FIGURES:
    out_png = CSV_FILE.replace(".csv", "_altitude_mismatch.png")
    fig.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")

plt.show()

