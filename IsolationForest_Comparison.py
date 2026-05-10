"""
=============================================================
 Model 3: Isolation Forest Anomaly Detection
 + Comparison: Z-Score vs Isolation Forest vs Known Events
 Thailand Crude Oil Imports  |  2008-01 to 2023-06
=============================================================
 Features used in Isolation Forest:
   - crude_oil_ML              (level)
   - crude_logdiff             (month-on-month % change)
   - rolling_z                 (24m Z-Score)
   - brent_price               (external factor)
   - brent_crude_ratio         (relative price-volume signal)
   - crude_12m_change          (year-on-year % change)

 Known crisis windows:
   - GFC 2008     : 2008-07 to 2009-03
   - COVID 2020   : 2020-02 to 2020-08
   - Ukraine 2022 : 2022-02 to 2022-06
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────────────────────────
# 1.  BRENT FALLBACK DICT
# ─────────────────────────────────────────────────────────────────────────────
brent_fallback = {
    "2008-01":72.7,"2008-02":95.4,"2008-03":101.8,"2008-04":109.0,
    "2008-05":122.8,"2008-06":132.7,"2008-07":133.4,"2008-08":113.2,
    "2008-09":97.2,"2008-10":74.0,"2008-11":52.5,"2008-12":40.1,
    "2009-01":46.0,"2009-02":44.3,"2009-03":47.9,"2009-04":50.6,
    "2009-05":58.7,"2009-06":69.3,"2009-07":65.0,"2009-08":73.0,
    "2009-09":67.9,"2009-10":73.6,"2009-11":77.0,"2009-12":74.9,
    "2010-01":76.2,"2010-02":73.7,"2010-03":79.4,"2010-04":84.9,
    "2010-05":77.0,"2010-06":74.8,"2010-07":75.5,"2010-08":76.7,
    "2010-09":78.1,"2010-10":83.4,"2010-11":85.9,"2010-12":91.5,
    "2011-01":96.3,"2011-02":103.5,"2011-03":115.5,"2011-04":122.9,
    "2011-05":113.9,"2011-06":113.4,"2011-07":117.2,"2011-08":107.0,
    "2011-09":103.4,"2011-10":109.6,"2011-11":109.2,"2011-12":107.7,
    "2012-01":111.3,"2012-02":119.2,"2012-03":125.5,"2012-04":120.4,
    "2012-05":108.9,"2012-06":95.6,"2012-07":102.0,"2012-08":112.7,
    "2012-09":113.2,"2012-10":111.2,"2012-11":108.8,"2012-12":109.0,
    "2013-01":112.0,"2013-02":116.1,"2013-03":108.7,"2013-04":102.2,
    "2013-05":103.4,"2013-06":103.4,"2013-07":107.8,"2013-08":111.0,
    "2013-09":111.8,"2013-10":109.4,"2013-11":107.4,"2013-12":110.8,
    "2014-01":107.3,"2014-02":108.9,"2014-03":107.4,"2014-04":107.6,
    "2014-05":109.2,"2014-06":111.9,"2014-07":106.9,"2014-08":101.8,
    "2014-09":97.0,"2014-10":86.5,"2014-11":78.0,"2014-12":61.5,
    "2015-01":47.8,"2015-02":58.0,"2015-03":55.6,"2015-04":59.5,
    "2015-05":64.7,"2015-06":63.1,"2015-07":56.7,"2015-08":46.6,
    "2015-09":47.8,"2015-10":47.8,"2015-11":44.7,"2015-12":37.9,
    "2016-01":30.7,"2016-02":32.2,"2016-03":38.1,"2016-04":41.7,
    "2016-05":47.7,"2016-06":46.4,"2016-07":44.7,"2016-08":45.8,
    "2016-09":46.9,"2016-10":49.9,"2016-11":47.2,"2016-12":54.9,
    "2017-01":55.5,"2017-02":55.9,"2017-03":52.6,"2017-04":52.6,
    "2017-05":50.7,"2017-06":46.4,"2017-07":49.1,"2017-08":51.4,
    "2017-09":55.5,"2017-10":57.3,"2017-11":62.7,"2017-12":64.4,
    "2018-01":69.1,"2018-02":64.3,"2018-03":66.0,"2018-04":71.7,
    "2018-05":76.5,"2018-06":74.9,"2018-07":74.3,"2018-08":73.1,
    "2018-09":79.8,"2018-10":81.3,"2018-11":65.0,"2018-12":57.4,
    "2019-01":59.5,"2019-02":64.0,"2019-03":67.2,"2019-04":71.2,
    "2019-05":70.8,"2019-06":63.6,"2019-07":63.9,"2019-08":59.4,
    "2019-09":62.5,"2019-10":60.2,"2019-11":63.2,"2019-12":67.8,
    "2020-01":63.7,"2020-02":55.7,"2020-03":33.7,"2020-04":26.5,
    "2020-05":30.3,"2020-06":40.3,"2020-07":43.2,"2020-08":44.3,
    "2020-09":41.0,"2020-10":40.5,"2020-11":43.1,"2020-12":50.2,
    "2021-01":54.8,"2021-02":62.0,"2021-03":65.0,"2021-04":65.4,
    "2021-05":68.4,"2021-06":73.5,"2021-07":74.5,"2021-08":70.5,
    "2021-09":74.9,"2021-10":83.7,"2021-11":80.3,"2021-12":74.2,
    "2022-01":83.5,"2022-02":97.2,"2022-03":117.9,"2022-04":104.6,
    "2022-05":112.6,"2022-06":122.7,"2022-07":105.8,"2022-08":99.7,
    "2022-09":91.6,"2022-10":93.5,"2022-11":90.4,"2022-12":80.9,
    "2023-01":82.2,"2023-02":83.0,"2023-03":78.3,"2023-04":84.4,
    "2023-05":75.6,"2023-06":74.9,
}

# ─────────────────────────────────────────────────────────────────────────────
# 2.  LOAD DATA + FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv("C:/Users/Hp/Desktop/my data oil/Monthly_import.csv")
df_raw.columns = df_raw.columns.str.strip()
df_raw["date"] = pd.to_datetime(
    df_raw["year_CE"].astype(str) + "-" + df_raw["month"].astype(str).str.zfill(2))
df_raw = df_raw.sort_values("date").set_index("date").loc["2008-01":"2023-06"]
crude = df_raw["crude_oil_ML"].astype(float)

brent_s = pd.Series(brent_fallback)
brent_s.index = pd.to_datetime([f"{k}-01" for k in brent_s.index])
brent = brent_s.sort_index().reindex(crude.index, method="nearest")

N = len(crude)
print(f"Sample: {crude.index[0].strftime('%Y-%m')} to {crude.index[-1].strftime('%Y-%m')}  N={N}")

# ── Feature engineering ──
WINDOW = 24
roll_mean = crude.rolling(WINDOW, min_periods=WINDOW).mean()
roll_std  = crude.rolling(WINDOW, min_periods=WINDOW).std()
z_score   = (crude - roll_mean) / roll_std

crude_logdiff    = np.log(crude).diff()
crude_12m_change = crude.pct_change(12)           # YoY %
brent_logdiff    = np.log(brent).diff()
brent_crude_ratio = brent / crude * 1000          # price per kML

# ── Z-Score anomaly labels ──
WARN_THR = 2.0
CRIT_THR = 3.0
z_signal = pd.Series("Normal", index=crude.index)
z_signal[z_score.abs() >= WARN_THR] = "Anomaly"
z_signal[z_score.abs() >= CRIT_THR] = "Critical"
z_anomaly = (z_signal != "Normal").astype(int)    # binary: 0/1

# ─────────────────────────────────────────────────────────────────────────────
# 3.  MODEL 3 — ISOLATION FOREST
# ─────────────────────────────────────────────────────────────────────────────
# Build feature matrix — drop early NaNs from rolling/diff
feature_df = pd.DataFrame({
    "crude_level":       crude,
    "crude_logdiff":     crude_logdiff,
    "rolling_zscore":    z_score,
    "brent_price":       brent,
    "brent_logdiff":     brent_logdiff,
    "brent_crude_ratio": brent_crude_ratio,
    "crude_12m_change":  crude_12m_change,
}, index=crude.index).dropna()

print(f"\nIsolation Forest feature matrix: {feature_df.shape}")
print(f"Features: {list(feature_df.columns)}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(feature_df)

# Contamination = expected anomaly fraction
# Z-Score found ~5% anomaly → use 7% for IF to allow some extra detection
CONTAM = 0.07

if_model = IsolationForest(
    n_estimators=300,
    contamination=CONTAM,
    max_samples="auto",
    random_state=42,
    n_jobs=-1,
)
if_model.fit(X_scaled)

# Predictions: -1 = anomaly, 1 = normal
if_pred   = if_model.predict(X_scaled)
if_scores = if_model.score_samples(X_scaled)   # lower = more anomalous

if_anomaly = pd.Series((if_pred == -1).astype(int), index=feature_df.index)
if_score_s = pd.Series(if_scores, index=feature_df.index)

# Normalize scores to 0–1 anomaly probability (invert: higher = more anomalous)
if_score_norm = 1 - (if_score_s - if_score_s.min()) / (if_score_s.max() - if_score_s.min())

n_if_anomaly = if_anomaly.sum()
print(f"\nIsolation Forest: {n_if_anomaly} anomaly months ({n_if_anomaly/len(if_anomaly):.1%})")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  KNOWN CRISIS WINDOWS
# ─────────────────────────────────────────────────────────────────────────────
crisis_windows = {
    "GFC 2008":           ("2008-09", "2009-03"),
    "Oil Crash\n2014–15": ("2014-07", "2015-06"),
    "COVID-19\n2020":     ("2020-02", "2020-08"),
    "Ukraine War\n2022":  ("2022-02", "2022-06"),
}

def months_in_window(start, end, index):
    return index[(index >= start) & (index <= end)]

# Build crisis label series
crisis_label = pd.Series("Normal", index=crude.index)
for name, (s, e) in crisis_windows.items():
    mask = (crude.index >= s) & (crude.index <= e)
    crisis_label[mask] = name.replace("\n", " ")
crisis_binary = (crisis_label != "Normal").astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  AGREEMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
# Align all series on the same index
common_idx = feature_df.index   # IF index (drops early NaNs)
z_a  = z_anomaly.reindex(common_idx, fill_value=0)
if_a = if_anomaly.reindex(common_idx, fill_value=0)
cr_a = crisis_binary.reindex(common_idx, fill_value=0)

# Both agree = flagged by both models
both_agree   = ((z_a == 1) & (if_a == 1)).astype(int)
only_zscore  = ((z_a == 1) & (if_a == 0)).astype(int)
only_if      = ((z_a == 0) & (if_a == 1)).astype(int)

print("\n" + "="*60)
print("  ANOMALY AGREEMENT SUMMARY")
print("="*60)
print(f"Z-Score anomalies    : {z_a.sum()} months")
print(f"Isolation Forest     : {if_a.sum()} months")
print(f"Both agree           : {both_agree.sum()} months")
print(f"Only Z-Score         : {only_zscore.sum()} months")
print(f"Only IF              : {only_if.sum()} months")
print(f"Agreement rate       : {both_agree.sum() / max(1, (z_a | if_a).sum()):.1%}")

# Per-crisis detection
print("\n" + "="*60)
print("  CRISIS WINDOW DETECTION RATE")
print("="*60)
print(f"{'Crisis':<22} {'Window':<20} {'Z-Score':>8} {'IF':>8} {'Both':>6} {'Months':>7}")
print("-"*70)

crisis_summary = []
for name, (s, e) in crisis_windows.items():
    # Use full crude index for IF (starts 2008), common_idx for Z-Score
    cidx_if = crude.index[(crude.index >= s) & (crude.index <= e)]
    cidx_z  = common_idx[(common_idx >= s) & (common_idx <= e)]
    total   = len(cidx_if)
    if total == 0:
        continue
    # Z-Score: may have fewer months due to warm-up
    z_hit  = z_a.reindex(cidx_z, fill_value=0).sum() if len(cidx_z) > 0 else 0
    z_total = len(cidx_z)
    if_hit = if_a.reindex(cidx_if, fill_value=0).sum()
    # Both agree: only on overlapping months
    cidx_both = common_idx[(common_idx >= s) & (common_idx <= e)]
    bo_hit = ((z_a.reindex(cidx_both, fill_value=0)==1) & (if_a.reindex(cidx_both, fill_value=0)==1)).sum() if len(cidx_both) > 0 else 0
    cname  = name.replace("\n", " ")
    z_str  = f"{z_hit}/{z_total}" if z_total > 0 else "N/A*"
    print(f"{cname:<22} {s+' – '+e:<20} {z_str:>8} {if_hit:>6}/{total:<2} {bo_hit:>4}/{total:<2} {total:>6}")
    crisis_summary.append({
        "crisis": cname, "start": s, "end": e,
        "total": total, "z_total": z_total, "z_hit": z_hit,
        "if_hit": if_hit, "both_hit": bo_hit
    })

# ─────────────────────────────────────────────────────────────────────────────
# 6.  VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
C_BG     = "#F7F6F2"
C_TEXT   = "#28251D"
C_MUTED  = "#7A7974"
C_BORDER = "#D4D1CA"
C_TEAL   = "#20808D"
C_DARK   = "#1B474D"
C_WARN   = "#D19900"
C_CRIT   = "#A13544"
C_IF     = "#7A39BB"      # purple for IF
C_BOTH   = "#437A22"      # green for agreement
C_CRISIS_COLORS = {
    "GFC 2008":              "#A13544",
    "Oil Crash 2014–15":     "#964219",
    "COVID-19 2020":         "#A84B2F",
    "Ukraine War 2022":      "#7A39BB",
}

fig = plt.figure(figsize=(16, 15), facecolor=C_BG)
gs  = gridspec.GridSpec(5, 1,
                        height_ratios=[2.2, 1.4, 1.4, 1.4, 2.0],
                        hspace=0.08, left=0.08, right=0.96,
                        top=0.93, bottom=0.06)

ax1 = fig.add_subplot(gs[0])   # Crude + Brent + shaded crises
ax2 = fig.add_subplot(gs[1], sharex=ax1)  # Z-Score
ax3 = fig.add_subplot(gs[2], sharex=ax1)  # IF anomaly score
ax4 = fig.add_subplot(gs[3], sharex=ax1)  # Agreement bar
ax5 = fig.add_subplot(gs[4])               # Heatmap / comparison table

def style_ax(ax):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_BORDER)
    ax.spines["bottom"].set_color(C_BORDER)
    ax.tick_params(colors=C_MUTED, labelsize=8.5)

for ax in [ax1, ax2, ax3, ax4]:
    style_ax(ax)
style_ax(ax5)

# ── Panel 1: Crude + Brent + crisis shading ───────────────────────────────
ax1b = ax1.twinx()
ax1b.set_facecolor("white")
ax1b.spines["top"].set_visible(False)

ax1.fill_between(crude.index, crude.values, alpha=0.10, color=C_TEAL)
lc, = ax1.plot(crude.index, crude.values, color=C_TEAL, lw=1.5, label="Crude Oil Import (ML)")
lb, = ax1b.plot(brent.index, brent.values, color=C_IF, lw=1.1, ls="--", alpha=0.7,
                 label="Brent Price (USD/bbl)")

# Shade known crisis windows
crisis_patch_handles = []
for name, (s, e) in crisis_windows.items():
    cname = name.replace("\n", " ")
    col = C_CRISIS_COLORS.get(cname, C_CRIT)
    ax1.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.13, color=col, zorder=0)
    crisis_patch_handles.append(
        mpatches.Patch(color=col, alpha=0.4, label=cname))

ax1.set_ylabel("Import Volume (ML)", fontsize=9, color=C_TEXT)
ax1b.set_ylabel("Brent (USD/bbl)", fontsize=9, color=C_IF)
ax1b.yaxis.label.set_color(C_IF)
ax1b.tick_params(colors=C_IF, labelsize=8.5)
ax1b.spines["right"].set_color(C_BORDER)
ax1.set_ylim(bottom=0)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

ax1.legend(handles=[lc, lb] + crisis_patch_handles,
           fontsize=7.5, loc="upper left", framealpha=0.85,
           edgecolor=C_BORDER, ncol=3)
ax1.set_title("Anomaly Detection Comparison: Z-Score vs Isolation Forest\n"
              "Thailand Crude Oil Imports | 2008–2023",
              fontsize=12, color=C_TEXT, fontweight="bold", pad=10)

# ── Panel 2: Z-Score with anomaly markers ────────────────────────────────
z_v = z_score.dropna()
ax2.axhline(0,        color=C_BORDER, lw=0.8)
ax2.axhline( WARN_THR, color=C_WARN, lw=1.0, ls="--", alpha=0.7)
ax2.axhline(-WARN_THR, color=C_WARN, lw=1.0, ls="--", alpha=0.7)
ax2.axhline( CRIT_THR, color=C_CRIT, lw=1.0, ls="--", alpha=0.6)
ax2.axhline(-CRIT_THR, color=C_CRIT, lw=1.0, ls="--", alpha=0.6)
ax2.fill_between(z_v.index, -WARN_THR, WARN_THR, alpha=0.05, color=C_TEAL)

for i in range(len(z_v)-1):
    t0, t1 = z_v.index[i], z_v.index[i+1]
    z0, z1 = z_v.iloc[i], z_v.iloc[i+1]
    c = C_CRIT if max(abs(z0),abs(z1)) >= CRIT_THR else \
        (C_WARN if max(abs(z0),abs(z1)) >= WARN_THR else C_TEAL)
    ax2.plot([t0,t1], [z0,z1], color=c, lw=1.4, zorder=2)

z_anom_idx = z_score.index[z_anomaly.reindex(z_score.index, fill_value=0) == 1]
ax2.scatter(z_anom_idx, z_score.loc[z_anom_idx],
            color=C_WARN, s=50, zorder=3, marker="o",
            label=f"Z-Score Anomaly ({z_a.sum()})")
ax2.set_ylabel("Z-Score (24m)", fontsize=9, color=C_TEXT)
ax2.set_ylim(-4, 4)
ax2.legend(fontsize=8, loc="lower left", framealpha=0.85, edgecolor=C_BORDER)
ax2.text(0.005, 0.97, "B  Z-Score Anomaly (Model 1)",
         transform=ax2.transAxes, fontsize=8.5, color=C_DARK,
         fontweight="bold", va="top")

# ── Panel 3: IF Anomaly Score ─────────────────────────────────────────────
ax3.plot(if_score_norm.index, if_score_norm.values,
         color=C_IF, lw=1.2, alpha=0.8, label="IF Anomaly Score")
ax3.fill_between(if_score_norm.index, if_score_norm.values,
                 alpha=0.12, color=C_IF)

# Threshold line (contamination boundary)
score_thresh = if_score_norm[if_anomaly == 1].min()
ax3.axhline(score_thresh, color=C_IF, lw=1.0, ls="--", alpha=0.7,
            label=f"Decision threshold")

ax3_anom = if_anomaly[if_anomaly == 1].index
ax3.scatter(ax3_anom, if_score_norm.loc[ax3_anom],
            color=C_IF, s=40, zorder=3, marker="D", alpha=0.9,
            label=f"IF Anomaly ({if_a.sum()})")
ax3.set_ylabel("Anomaly Score", fontsize=9, color=C_TEXT)
ax3.set_ylim(0, 1.05)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax3.legend(fontsize=8, loc="upper right", framealpha=0.85, edgecolor=C_BORDER, ncol=3)
ax3.text(0.005, 0.97, "C  Isolation Forest Score (Model 3)",
         transform=ax3.transAxes, fontsize=8.5, color=C_DARK,
         fontweight="bold", va="top")

# ── Panel 4: Agreement Bar ────────────────────────────────────────────────
# Stack: 0=Normal, 1=Only Z, 2=Only IF, 3=Both
signal_type = pd.Series(0, index=common_idx)
signal_type[only_zscore == 1] = 1
signal_type[only_if     == 1] = 2
signal_type[both_agree  == 1] = 3

colors_map = {0: C_TEAL, 1: C_WARN, 2: C_IF, 3: C_BOTH}
for t, val in signal_type.items():
    ax4.bar(t, 1, width=25, color=colors_map[val], alpha=0.85)

# Crisis shading in panel 4
for name, (s, e) in crisis_windows.items():
    cname = name.replace("\n", " ")
    col = C_CRISIS_COLORS.get(cname, C_CRIT)
    ax4.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.08, color=col, zorder=0)

ax4.set_ylim(0, 1.2)
ax4.set_yticks([])
normal_p  = mpatches.Patch(color=C_TEAL, alpha=0.85, label="Normal")
zscore_p  = mpatches.Patch(color=C_WARN, alpha=0.85, label=f"Z-Score only")
if_p      = mpatches.Patch(color=C_IF,   alpha=0.85, label=f"IF only")
both_p    = mpatches.Patch(color=C_BOTH, alpha=0.85, label=f"Both agree")
ax4.legend(handles=[normal_p, zscore_p, if_p, both_p],
           fontsize=8, loc="upper right", framealpha=0.85, edgecolor=C_BORDER, ncol=4)
ax4.text(0.005, 0.95, "D  Signal Agreement Timeline",
         transform=ax4.transAxes, fontsize=8.5, color=C_DARK,
         fontweight="bold", va="top")

# X-axis (ax4 is the bottom shared axis)
ax4.xaxis.set_major_locator(mdates.YearLocator())
ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
for ax in [ax1, ax2, ax3]:
    plt.setp(ax.get_xticklabels(), visible=False)
ax4.tick_params(axis="x", labelsize=8.5, colors=C_MUTED)

# Event lines across all 4 shared panels
event_lines = {
    "2008-09": "GFC\n2008",
    "2020-04": "COVID\n2020",
    "2022-03": "Ukraine\n2022",
}
for date_str, label in event_lines.items():
    t = pd.Timestamp(date_str)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(t, color=C_MUTED, lw=0.6, ls=":", alpha=0.5, zorder=0)
    ax1.text(t, ax1.get_ylim()[1]*0.94, label,
             fontsize=7, color=C_MUTED, ha="left", va="top",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_BORDER, alpha=0.7))

ax1.text(0.005, 0.97, "A  Crude Oil Import + Brent Price",
         transform=ax1.transAxes, fontsize=8.5, color=C_DARK,
         fontweight="bold", va="top")

# ── Panel 5: Comparison Heatmap / Table ──────────────────────────────────
ax5.set_facecolor("white")
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis("off")

# Title
ax5.text(0.5, 0.97, "E  Crisis Detection Scorecard",
         ha="center", va="top", fontsize=10, color=C_DARK,
         fontweight="bold")

# Table layout
crises = ["GFC 2008", "Oil Crash 2014–15", "COVID-19 2020", "Ukraine War 2022"]
cols_lbl = ["Crisis Event", "Window", "Months", "Z-Score", "Isolation\nForest", "Both\nAgree", "Agreement\nRate"]
col_x    = [0.01, 0.20, 0.34, 0.43, 0.54, 0.65, 0.76]
col_align= ["left","left","center","center","center","center","center"]

# Header row
header_y = 0.84
for cx, cl, ca in zip(col_x, cols_lbl, col_align):
    ax5.text(cx, header_y, cl, ha=ca, va="center",
             fontsize=8.5, color="white", fontweight="bold",
             transform=ax5.transAxes)
ax5.add_patch(mpatches.FancyBboxPatch(
    (0.0, header_y - 0.07), 1.0, 0.115,
    boxstyle="round,pad=0.005", fc=C_DARK, ec="none",
    transform=ax5.transAxes, zorder=0))

row_y_start = 0.70
row_h = 0.125

for i, row in enumerate(crisis_summary):
    ry = row_y_start - i * row_h
    bg = "#F0F8F8" if i % 2 == 0 else "white"
    ax5.add_patch(mpatches.FancyBboxPatch(
        (0.0, ry - row_h*0.45), 1.0, row_h,
        boxstyle="round,pad=0.005", fc=bg, ec=C_BORDER, lw=0.4,
        transform=ax5.transAxes, zorder=0))

    z_avail = row.get("z_total", row["total"])
    z_rate  = row["z_hit"] / max(1, z_avail)
    if_rate = row["if_hit"] / row["total"]
    bo_rate = row["both_hit"] / row["total"]
    agree_rate = row["both_hit"] / max(1, max(row["z_hit"], row["if_hit"]))
    z_na = (z_avail == 0)

    def rate_color(r):
        return C_BOTH if r >= 0.5 else (C_WARN if r >= 0.25 else C_CRIT)

    ax5.text(col_x[0], ry, row["crisis"],
             ha="left", va="center", fontsize=8.5, color=C_TEXT, fontweight="bold")
    ax5.text(col_x[1], ry, f"{row['start']} – {row['end']}",
             ha="left", va="center", fontsize=8, color=C_MUTED)
    ax5.text(col_x[2], ry, str(row["total"]),
             ha="center", va="center", fontsize=8.5, color=C_TEXT)
    if z_na:
        ax5.text(col_x[3], ry, "N/A*",
                 ha="center", va="center", fontsize=8.5, color=C_MUTED, style="italic")
    else:
        ax5.text(col_x[3], ry, f"{row['z_hit']}/{row.get('z_total',row['total'])} ({z_rate:.0%})",
                 ha="center", va="center", fontsize=8.5, color=rate_color(z_rate))
    ax5.text(col_x[4], ry, f"{row['if_hit']}/{row['total']} ({if_rate:.0%})",
             ha="center", va="center", fontsize=8.5, color=rate_color(if_rate))
    ax5.text(col_x[5], ry, f"{row['both_hit']}/{row['total']} ({bo_rate:.0%})",
             ha="center", va="center", fontsize=8.5, color=rate_color(bo_rate))

    # Agreement rate pill
    pill_col = C_BOTH if agree_rate >= 0.5 else (C_WARN if agree_rate >= 0.25 else C_CRIT)
    ax5.add_patch(mpatches.FancyBboxPatch(
        (col_x[6] - 0.005, ry - 0.04), 0.18, 0.08,
        boxstyle="round,pad=0.01", fc=pill_col, alpha=0.15, ec=pill_col, lw=0.8,
        transform=ax5.transAxes))
    ax5.text(col_x[6] + 0.08, ry, f"{agree_rate:.0%}",
             ha="center", va="center", fontsize=9, color=pill_col, fontweight="bold")

# Summary row
sum_y = row_y_start - len(crisis_summary) * row_h - 0.03
overall_agree = both_agree.sum() / max(1, (z_a | if_a).sum())
ax5.text(0.5, sum_y,
         f"Overall agreement rate: {both_agree.sum()} / {(z_a | if_a).sum()} anomaly-months  "
         f"({overall_agree:.0%})  |  "
         f"Z-Score: {z_a.sum()} months  |  "
         f"Isolation Forest: {if_a.sum()} months",
         ha="center", va="center", fontsize=8.5, color=C_MUTED, style="italic")
ax5.text(0.5, sum_y - 0.10,
         "* N/A = Z-Score requires 24-month warm-up period; data unavailable during GFC 2008 window",
         ha="center", va="center", fontsize=7.5, color=C_MUTED, style="italic")

# ── Footer ────────────────────────────────────────────────────────────────
fig.text(0.5, 0.015,
         f"Model 1: Rolling Z-Score (w=24m, threshold ±2σ)  |  "
         f"Model 3: Isolation Forest (n_estimators=300, contamination={CONTAM:.0%}, features=7)  |  "
         f"N={N} monthly obs",
         ha="center", fontsize=7.5, color=C_MUTED, style="italic")

out = 'C:/Users/Hp/Desktop/my data oil/IsolationForest_Comparison_Result.png'
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"\nPlot saved: {out}")