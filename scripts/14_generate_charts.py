import json
import os
import time
import warnings

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
t0 = time.time()

os.makedirs("work/charts", exist_ok=True)
with open("work/master_metrics.json", "r", encoding="utf-8") as f:
    M = json.load(f)
with open("work/audit_summary.json", "r", encoding="utf-8") as f:
    A = json.load(f)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Segoe UI"],
    "font.size": 10.0,
    "axes.titlesize": 11.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 10.0,
    "axes.labelcolor": "#1E293B",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.frameon": False,
    "legend.fontsize": 8.8,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.14,
})

NAVY = "#0F2942"
BLUE = "#2563EB"
TEAL = "#0D9488"
AMBER = "#D97706"
RED = "#DC2626"
PURPLE = "#7C3AED"
SLATE = "#64748B"
LIGHT_SLATE = "#CBD5E1"
GRID_COLOR = "#E2E8F0"

def add_grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID_COLOR, linestyle="--", linewidth=0.7, alpha=0.85, zorder=0)
    ax.set_axisbelow(True)

def save_fig(fig, name):
    path = f"work/charts/{name}"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

CORE_PARTS = [f"work/core_parquet/part_{i:02d}.parquet" for i in range(40)]

print("Loading fast Parquet slices for duration pair/scatter/LGBM plots...")
pair_sample_durs = []
p3_collect = []
dir_pair_chunks = []
cr_rows = []
norm_pool = []

for idx_p, p in enumerate(CORE_PARTS):
    d = pd.read_parquet(p, columns=[
        "epoch_seconds", "ordering_bic8", "beneficiary_bic8", "bic8_path",
        "total_duration", "continuity_qflag", "fx_count", "participant_count",
        "tx_type", "usd_amount", "ordering_type", "beneficiary_type",
        "ord_det_len", "ben_det_len", "ord_has_iban", "ben_has_iban", "anomaly_type"
    ])
    m_pair = (d["ordering_bic8"] == "0001MQM0") & (d["beneficiary_bic8"] == "0009TNT0") & (d["tx_type"] == "Direct") & (d["anomaly_type"] == "")
    if m_pair.any():
        pair_sample_durs.append(d.loc[m_pair, "total_duration"].to_numpy())
    dir_pair_chunks.append(d.loc[(d["tx_type"] == "Direct") & (d["anomaly_type"] == ""), ["ordering_bic8", "beneficiary_bic8", "total_duration"]])
    if idx_p < 2:
        p3_collect.append(d.loc[(d["participant_count"] == 3) & (d["continuity_qflag"] == "Y") & (d["anomaly_type"] == ""),
                                ["bic8_path", "tx_type", "total_duration", "ordering_bic8", "beneficiary_bic8"]])
    m_cr = (d["anomaly_type"] == "CUSTOMER_RECALL")
    if m_cr.any():
        cr_rows.append(d.loc[m_cr])
    if idx_p % 2 == 0:
        norm_pool.append(d.loc[d["anomaly_type"] == ""].sample(n=10000, random_state=42 + idx_p))

pair_0001_vals = np.concatenate(pair_sample_durs)
all_dir = pd.concat(dir_pair_chunks, ignore_index=True)
pair_agg = all_dir.groupby(["ordering_bic8", "beneficiary_bic8"], observed=True)["total_duration"].agg(["mean", "count"])
pair_mean_map = pair_agg.loc[pair_agg["count"] >= 3, "mean"].to_dict()
del all_dir, dir_pair_chunks, pair_agg

p3_sample_df = pd.concat(p3_collect, ignore_index=True)
del pair_sample_durs, p3_collect

cr_df = pd.concat(cr_rows, ignore_index=True)
no_df = pd.concat(norm_pool, ignore_index=True)
cr_df["y"] = 1
no_df["y"] = 0
ml_df = pd.concat([cr_df, no_df], ignore_index=True)
ml_df["utc_hour"] = (ml_df["epoch_seconds"] % 86400) // 3600
ml_df["is_qflag_n"] = (ml_df["continuity_qflag"] == "N").astype(int)
ml_df["is_ord_B"] = (ml_df["ordering_type"] == "B").astype(int)
ml_df["is_ben_B"] = (ml_df["beneficiary_type"] == "B").astype(int)
ml_df["is_direct"] = (ml_df["tx_type"] == "Direct").astype(int)
ml_df["is_cov"] = (ml_df["tx_type"] == "Covered").astype(int)
ml_df["start_mis"] = (ml_df["bic8_path"].str.slice(0, 8) != ml_df["ordering_bic8"]).astype(int)
ml_df["end_mis"] = (ml_df["bic8_path"].str.slice(-8) != ml_df["beneficiary_bic8"]).astype(int)
feats = ["usd_amount", "total_duration", "fx_count", "participant_count", "utc_hour",
         "is_qflag_n", "is_ord_B", "is_ben_B", "is_direct", "is_cov",
         "ord_det_len", "ben_det_len", "ord_has_iban", "ben_has_iban", "start_mis", "end_mis"]
X_ml = ml_df[feats].to_numpy(dtype=np.float32)
y_ml = ml_df["y"].to_numpy()
oof = np.zeros(len(y_ml))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr, va in skf.split(X_ml, y_ml):
    clf = lgb.LGBMClassifier(n_estimators=80, max_depth=4, num_leaves=15, learning_rate=0.05,
                             class_weight="balanced", random_state=42, n_jobs=8, verbose=-1)
    clf.fit(X_ml[tr], y_ml[tr])
    oof[va] = clf.predict_proba(X_ml[va])[:, 1]
auc_cr = float(roc_auc_score(y_ml, oof))
fpr_cr, tpr_cr, _ = roc_curve(y_ml, oof)
print(f"CUSTOMER_RECALL 5-fold LGBM OOF AUC = {auc_cr:.4f} (prep {time.time() - t0:.1f}s)")
M["anomalies"]["customer_recall_lgbm_auc"] = round(auc_cr, 4)
with open("work/master_metrics.json", "w", encoding="utf-8") as f:
    json.dump(M, f, indent=2)
del cr_df, no_df, ml_df

# ==============================================================================
# C01: DATA PREP SENSITIVITY
# ==============================================================================
sens = pd.DataFrame(M["data_prep_sensitivity"])
labels_s = ["Raw All\n(10.00M)", "Clean\n(9.993M)", "Clean +\nEndpt Match", "Clean +\nQFLAG='Y'", "Clean+Match\n+QFLAG='Y'"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.9))
x = np.arange(len(sens))
ax1.bar(x, sens["usd_mean"] / 1000.0, width=0.56, color=[RED, NAVY, NAVY, SLATE, SLATE], zorder=3)
ax1.set_xticks(x)
ax1.set_xticklabels(labels_s, fontsize=8.4)
ax1.set_ylabel("Mean USD Amount ($ Thousands)")
ax1.set_ylim(0, 118)
add_grid(ax1)
ax1.set_title("1,940 Fat-Finger Errors Inflate Mean +22.1%", loc="left")
for i, v in enumerate(sens["usd_mean"]):
    ax1.text(i, v / 1000.0 + 2.2, f"${v/1000:.1f}k", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=RED if i == 0 else NAVY)
ax1.annotate("Median unchanged:\n$3,662 -> $3,663 (+$0.69)", xy=(1, 80), xytext=(1.75, 102),
             fontsize=8.4, bbox=dict(boxstyle="round,pad=0.3", fc="#F1F5F9", ec=BLUE, lw=0.9), color=NAVY)

ax2.bar(x, sens["direct_pct"], width=0.56, color=[NAVY, NAVY, NAVY, AMBER, AMBER], zorder=3)
ax2.set_xticks(x)
ax2.set_xticklabels(labels_s, fontsize=8.4)
ax2.set_ylabel("Direct (N=2) Share (%)")
ax2.set_ylim(0, 73)
add_grid(ax2)
ax2.set_title("QFLAG='Y' Filter Creates +1.84 pp Collider Bias", loc="left")
for i, v in enumerate(sens["direct_pct"]):
    ax2.text(i, v + 1.2, f"{v:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=AMBER if i >= 3 else NAVY)
ax2.annotate("Longer routes have more '...'\n-> dropping 'N' trims multihop", xy=(3, 64.5), xytext=(0.15, 42),
             arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.2), fontsize=8.3,
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFFBEB", ec=AMBER, lw=0.9), color="#78350F")
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c01_sensitivity_mean_bias.png")

# ==============================================================================
# C02: ANOMALY CATALOG
# ==============================================================================
anom_df = pd.DataFrame([r for r in M["anomalies"]["summary_table"] if r["anomaly"] != "NORMAL"]).sort_values("n", ascending=True)
fig, ax = plt.subplots(figsize=(7.8, 3.8))
y_pos = np.arange(len(anom_df))
colors_a = [PURPLE if a == "CUSTOMER_RECALL" else (BLUE if a in ["FAT_FINGER_MISTAKE", "DUPLICATE_PAYMENT"] else RED) for a in anom_df["anomaly"]]
ax.barh(y_pos, anom_df["n"], height=0.62, color=colors_a, zorder=3)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"{a}{' *' if a == 'CUSTOMER_RECALL' else ''}" for a in anom_df["anomaly"]], fontsize=9.2, fontweight="bold")
ax.set_xlim(0, 3450)
ax.set_xlabel("Labeled Row Count (Total = 7,006 = 0.070% of 10M Rows)")
add_grid(ax, "x")
ax.set_title("Six Anomaly Classes Found (*CUSTOMER_RECALL Undocumented)", loc="left")
for i, (_, r) in enumerate(anom_df.iterrows()):
    ax.text(r["n"] + 45, i, f"{int(r['n']):,}  (med ${r['usd_median']:,.0f}, Covered {r['covered_among_multihop_pct']:.1f}%)",
            va="center", fontsize=8.5, color="#1E293B")
save_fig(fig, "c02_anomaly_counts_bar.png")

# ==============================================================================
# C03: PARTY-TYPE MATRIX
# ==============================================================================
pp = M["entity_graph"]["party_pairs"]
tot = M["row_count"]
segs = ["B->C", "B->B", "C->C", "C->B"]
vals_p = [100.0 * pp[s] / tot for s in segs]
fig, ax = plt.subplots(figsize=(7.0, 3.8))
xp = np.arange(len(segs))
ax.bar(xp, vals_p, width=0.54, color=[BLUE, NAVY, TEAL, AMBER], zorder=3)
ax.set_xticks(xp)
ax.set_xticklabels(["B2C\n(Corp->Retail)", "B2B\n(Wholesale)", "C2C\n(P2P Remit)", "C2B\n(Retail->Corp)"], fontsize=9.2)
ax.set_ylabel("Share of All 10M Transactions (%)")
ax.set_ylim(0, 66)
add_grid(ax)
ax.set_title("78.8% Business Origination vs. Near-Independent 28.8% B-Beneficiary", loc="left")
for i, (s, v) in enumerate(zip(segs, vals_p)):
    ax.text(i, v + 1.5, f"{v:.2f}%\n({pp[s]/1e6:.2f}M)", ha="center", va="bottom", fontsize=8.6, fontweight="bold")
ax.annotate("P(Ben=B | Ord=B) = 28.84%\nP(Ben=B | Ord=C) = 28.73%\n(Near-zero coupling)", xy=(1, 23), xytext=(2.02, 47),
            fontsize=8.6, bbox=dict(boxstyle="round,pad=0.35", fc="#F8FAFC", ec=NAVY, lw=0.9))
save_fig(fig, "c03_party_matrix.png")

# ==============================================================================
# C04: ENTITY CARDINALITY & DUAL-ROLE ASYMMETRY
# ==============================================================================
eg = M["entity_graph"]
multi_ben_accts = sum(v for k, v in eg["ben_bic_per_acct_dist"].items() if int(k) >= 2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.9))
cats = ["Ordering Accts\n(391,132)", "Beneficiary Accts\n(2,590,877)"]
single_bic = [100.0, eg["ben_acct_1_bic_pct"]]
multi_bic = [0.0, 100.0 - eg["ben_acct_1_bic_pct"]]
ax1.bar([0, 1], single_bic, width=0.52, label="Single Home BIC8", color=TEAL, zorder=3)
ax1.bar([0, 1], multi_bic, bottom=single_bic, width=0.52, label="2–8 BIC8s (0% IBANs)", color=RED, zorder=3)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(cats, fontsize=9.2)
ax1.set_ylim(0, 130)
ax1.set_ylabel("Share of Unique Accounts (%)")
add_grid(ax1)
ax1.legend(loc="upper center", ncol=2, fontsize=8.3)
ax1.set_title("6.6x More Beneficiary Accts & 2.38% Multi-Bank", loc="left")
ax1.text(0, 50, "100.0%\nStrict 1:1", ha="center", va="center", color="white", fontweight="bold")
ax1.text(1, 50, f"{eg['ben_acct_1_bic_pct']:.2f}%\n1 Home BIC", ha="center", va="center", color="white", fontweight="bold")
ax1.annotate(f"{multi_ben_accts:,} Accts\nHave 2–8 BICs (2.38%)", xy=(1, 99.0), xytext=(0.45, 104.5),
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.2), fontsize=8.1, fontweight="bold", color=RED, ha="center")

dr_labels = ["Same Home\nCountry", "Same Home\nBIC8", "Same Party\nType (B vs C)"]
dr_vals = [eg["dual_same_country_pct"], eg["dual_same_bic_pct"], eg["dual_same_type_pct"]]
ax2.bar([0, 1, 2], dr_vals, width=0.5, color=[TEAL, TEAL, RED], zorder=3)
ax2.set_xticks([0, 1, 2])
ax2.set_xticklabels(dr_labels, fontsize=9.0)
ax2.set_ylim(0, 118)
ax2.set_ylabel("Agreement Rate (%)")
add_grid(ax2)
ax2.set_title("253,120 Dual-Role Accts Share Bank, Not B/C Type", loc="left")
for i, v in enumerate(dr_vals):
    ax2.text(i, v + 2.5, f"{v:.2f}%", ha="center", va="bottom", fontweight="bold", color=RED if i == 2 else NAVY)
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c04_entity_cardinality_asym.png")

# ==============================================================================
# C05: TIMEZONE ALIGNMENT
# ==============================================================================
tz_list = {d["country"]: d for d in M["temporal"]["tz_countries"]}
show_ccs = [("US", "UTC-4", BLUE), ("GB", "UTC+1", NAVY), ("DE", "UTC+2", TEAL),
            ("AE", "UTC+5", AMBER), ("SG", "UTC+9", PURPLE), ("AU", "UTC+12", RED)]
hrs = np.arange(24)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.0), sharey=True)
for cc, off_lbl, col in show_ccs:
    ax1.plot(hrs, tz_list[cc]["utc_hour_pct"], lw=2.0, label=f"{cc} ({off_lbl})", color=col)
    ax2.plot(hrs, tz_list[cc]["local_hour_pct"], lw=2.0, label=f"{cc}", color=col)
for ax, ttl, xlab in [(ax1, "Raw UTC Profiles Appear Dispersed", "UTC Hour (0–23)"),
                      (ax2, "Integer Timezone Shifts Collapse (r >= 0.997)", "Inferred Local Hour (0–23)")]:
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax.set_xlabel(xlab)
    ax.set_ylim(0, 9.6)
    add_grid(ax)
    ax.set_title(ttl, loc="left")
ax1.set_ylabel("Hourly Transaction Share (%)")
ax1.legend(loc="upper left", ncol=2, fontsize=8.3)
ax2.axvspan(7.5, 19.5, color="#F1F5F9", zorder=0)
ax2.annotate("08:00 Open & 17:00–19:00 Cutoff\nUniversal Twin-Peak Business Curve", xy=(17, 7.6), xytext=(11.5, 8.3),
             fontsize=8.3, ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec=SLATE, lw=0.8))
fig.tight_layout(w_pad=2.4)
save_fig(fig, "c05_timezone_alignment_hours.png")

# ==============================================================================
# C06: DAY-OF-WEEK UTC VS LOCAL + MIDDLE EAST WORKWEEK
# ==============================================================================
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.9))
x_d = np.arange(7)
for idx_c, (cc, col) in enumerate([("US", BLUE), ("AU", RED), ("DE", TEAL)]):
    ax1.bar(x_d + (idx_c - 1) * 0.25, tz_list[cc]["utc_dow_pct"], width=0.25, label=f"{cc} (UTC)", color=col, alpha=0.88, zorder=3)
ax1.set_xticks(x_d)
ax1.set_xticklabels(dow_names)
ax1.set_ylim(0, 22.5)
ax1.set_ylabel("Day-of-Week Share (%)")
add_grid(ax1)
ax1.legend(loc="upper right")
ax1.set_title("UTC Artifact: AU Sun (10.1%) & US Sat (6.6%)", loc="left")

for idx_c, (cc, col) in enumerate([("US", BLUE), ("AU", RED), ("AE", AMBER), ("SA", PURPLE)]):
    ax2.bar(x_d + (idx_c - 1.5) * 0.2, tz_list[cc]["local_dow_pct"], width=0.2, label=f"{cc} (Local)", color=col, alpha=0.9, zorder=3)
ax2.set_xticks(x_d)
ax2.set_xticklabels(dow_names)
ax2.set_ylim(0, 22.5)
add_grid(ax2)
ax2.legend(loc="upper right", ncol=2, fontsize=8.3)
ax2.set_title("Local-Time Collapse (AE/SA Follow Mon–Fri)", loc="left")
ax2.annotate("AE & SA follow Mon–Fri\n(not Sun–Thu real calendar)", xy=(4, 18.8), xytext=(4.35, 11.8),
             arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.1), fontsize=8.2,
             bbox=dict(boxstyle="round,pad=0.3", fc="#FFFBEB", ec=AMBER, lw=0.8))
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c06_dow_utc_vs_local.png")

# ==============================================================================
# C07: 92-DAY STEP-WAVE VOLUME
# ==============================================================================
daily = pd.read_csv("work/chart_data/daily_series.csv")
daily["dt"] = pd.to_datetime(daily["date"])
fig, ax1 = plt.subplots(figsize=(10.4, 3.8))
ax1.plot(daily["dt"], daily["tx_count"] / 1000.0, color=NAVY, lw=1.9, marker="o", ms=2.5, label="Daily Tx Count (Thousands)", zorder=3)
ax1.set_ylabel("Daily Transactions (Thousands)", color=NAVY)
ax1.set_ylim(0, 265)
add_grid(ax1)
ax1.axvspan(pd.Timestamp("2023-06-19"), pd.Timestamp("2023-06-25"), color="#DBEAFE", alpha=0.6, zorder=1)
ax1.axvspan(pd.Timestamp("2023-05-08"), pd.Timestamp("2023-05-21"), color="#F1F5F9", alpha=0.8, zorder=1)
ax1.text(pd.Timestamp("2023-06-22"), 240, "Peak Week\n(~218k/day)", ha="center", fontsize=8.4, fontweight="bold", color=BLUE)
ax1.text(pd.Timestamp("2023-05-14"), 240, "Quiet Weeks\n(~91k/day)", ha="center", fontsize=8.4, fontweight="bold", color=SLATE)
ax2 = ax1.twinx()
ax2.spines["top"].set_visible(False)
ax2.plot(daily["dt"], daily["clean_med_amt"], color=TEAL, lw=1.5, ls="--", alpha=0.9, label="Clean Median USD")
ax2.set_ylabel("Median Clean USD Amount ($)", color=TEAL)
ax2.set_ylim(0, 7500)
ax1.set_title("Weekly Monday–Sunday Volume Step-Waves Leave Median Ticket ($3.66k) Unchanged", loc="left")
fig.autofmt_xdate()
save_fig(fig, "c07_daily_step_waves.png")

# ==============================================================================
# C08: LOG-NORMAL PARALLEL CURVES
# ==============================================================================
lnh = pd.read_csv("work/chart_data/ln_amt_segment_hist.csv")
mids = 0.5 * (lnh["bin_left"] + lnh["bin_right"])
seg_meta = M["payment_value"]["seg_stats"]
fig, ax = plt.subplots(figsize=(8.4, 4.0))
for col_k, seg_k, lbl, col in [
    ("CC", "C->C", "C2C", TEAL),
    ("BC", "B->C", "B2C", BLUE),
    ("CB", "C->B", "C2B", AMBER),
    ("BB", "B->B", "B2B", NAVY),
]:
    st = seg_meta[seg_k]
    ax.plot(mids, lnh[col_k], lw=2.2 if seg_k != "B->C" else 2.5, ls="--" if seg_k == "C->B" else "-",
            color=col, label=f"{lbl}: mu={st['ln_mean']:.2f}, sigma={st['ln_std']:.2f} (med ${st['median']:,.0f})")
ax.set_xlim(-2.5, 19.5)
ax.set_ylim(0, 0.225)
ax.set_xlabel("Natural Log of Payment Value: ln(USD_AMOUNT)")
ax.set_ylabel("Probability Density")
add_grid(ax)
ax.legend(loc="upper right", fontsize=8.4)
ax.set_title("Exact Additive Log-Normal Rule: +2.00 ln (~7.4x) per Business Endpoint", loc="left")
ax.annotate("B2C & C2B overlap\nidentically (mu=7.99 vs 8.03)", xy=(8.0, 0.174), xytext=(13.0, 0.115),
            arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.1), fontsize=8.4,
            bbox=dict(boxstyle="round,pad=0.35", fc="#EFF6FF", ec=BLUE, lw=0.8))
save_fig(fig, "c08_ln_amount_segments.png")

# ==============================================================================
# C09: R2 VARIANCE DECOMPOSITION & COUNTRY MULTIPLIERS
# ==============================================================================
pv = M["payment_value"]
cc_mult = pd.read_csv("work/chart_data/country_amount_multipliers.csv")
top_c = cc_mult.head(7).iloc[::-1]
bot_c = cc_mult.tail(7).iloc[::-1]
show_c = pd.concat([bot_c, top_c], ignore_index=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.1), gridspec_kw={"width_ratios": [1.0, 1.25]})
r2_labels = ["Party Type\n(B/C)", "+ Ordering\nCountry", "+ Currency\n(+0.00001)", "+ Benefic.\nCountry", "+ Ordering\nBIC8"]
r2_vals = [pv["r2_party"], pv["r2_party_ord_cc"], pv["r2_party_ord_cc_curr"], pv["r2_party_ord_cc_ben_cc"], pv["r2_party_ord_bic"]]
ax1.bar(np.arange(5), r2_vals, width=0.56, color=[NAVY, TEAL, LIGHT_SLATE, LIGHT_SLATE, BLUE], zorder=3)
ax1.set_xticks(np.arange(5))
ax1.set_xticklabels(r2_labels, fontsize=8.3)
ax1.set_ylim(0, 0.45)
ax1.set_ylabel("Explained Variance R2 on ln(USD_AMOUNT)")
add_grid(ax1)
ax1.set_title("Currency & Receiver Have Zero Causal Effect", loc="left")
for i, v in enumerate(r2_vals):
    ax1.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=8.2, fontweight="bold")

yc = np.arange(len(show_c))
ax2.barh(yc, show_c["mult_vs_global"], height=0.68, color=[RED if v < 0.1 else TEAL for v in show_c["mult_vs_global"]], zorder=3)
ax2.set_yticks(yc)
ax2.set_yticklabels([f"{r['ord_bic_country']} (med ${r['raw_med']:,.0f})" for _, r in show_c.iterrows()], fontsize=8.4)
ax2.set_xscale("log")
ax2.set_xlim(0.002, 42)
ax2.axvline(1.0, color=NAVY, lw=1.1, ls="--")
add_grid(ax2, "x")
ax2.set_xlabel("Party-Adjusted Geometric Multiplier (Global=1.0x, Log)")
ax2.set_title("1,750x Country Spread (MC 7.7x vs BI 0.004x)", loc="left")
for i, (_, r) in enumerate(show_c.iterrows()):
    ax2.text(r["mult_vs_global"] * 1.15, i, f"{r['mult_vs_global']:.3f}x" if r["mult_vs_global"] < 1 else f"{r['mult_vs_global']:.2f}x",
             va="center", fontsize=8.0)
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c09_amount_r2_and_countries.png")

# ==============================================================================
# C10: USD ROUNDING & DAILY EUR FX
# ==============================================================================
crr = pd.DataFrame(pv["curr_round_rates"]).head(8)
eur_fx = pd.read_csv("work/chart_data/eur_daily_fx_1000.csv")
eur_fx["dt"] = pd.to_datetime(eur_fx["date"])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.9))
xc = np.arange(len(crr))
ax1.bar(xc, crr["exact_int_pct"], width=0.58, color=[BLUE if c == "USD" else SLATE for c in crr["currency"]], label="Exact Integer ($X.00)", zorder=3)
ax1.bar(xc, crr["mult100_pct"], width=0.58, color=[NAVY if c == "USD" else "#475569" for c in crr["currency"]], label="Multiple of $100", zorder=4)
ax1.set_xticks(xc)
ax1.set_xticklabels(crr["currency"])
ax1.set_ylim(0, 46)
ax1.set_ylabel("Share of Clean Transactions (%)")
add_grid(ax1)
ax1.legend(loc="upper right")
ax1.set_title("USD 38.6% Ints vs <1.2% Non-USD", loc="left")
ax1.text(1, 39.5, "38.64%\n(15.55% %100)", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=NAVY)

ax2.plot(eur_fx["dt"], eur_fx["usd_for_1000eur"] / 1000.0, color=TEAL, lw=2.0, marker="o", ms=2.4, zorder=3)
ax2.set_ylabel("Implied EUR/USD Daily Spot Rate")
add_grid(ax2)
ax2.set_ylim(1.08, 1.24)
ax2.set_title("Implied Daily EUR/USD from 1,000 EUR Mode", loc="left")
fig.autofmt_xdate()
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c10_usd_rounding_and_eur_fx.png")

# ==============================================================================
# C11: CURRENCY SELECTION RULES
# ==============================================================================
fc = M["fx_and_currency"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.8))
zones = [f"Same Currency Zone\n({fc['same_zone_count']/1e6:.2f}M Rows)",
         f"Cross Currency Zone\n({fc['cross_zone_count']/1e6:.2f}M Rows)"]
ord_s = [fc["same_zone_shared_curr_pct"], fc["cross_zone_ord_home_pct"]]
ben_s = [0.0, fc["cross_zone_ben_home_pct"]]
trd_s = [100.0 - fc["same_zone_shared_curr_pct"], fc["cross_zone_third_curr_pct"]]
ax1.bar([0, 1], ord_s, width=0.5, label="Ord Home", color=NAVY, zorder=3)
ax1.bar([0, 1], ben_s, bottom=ord_s, width=0.5, label="Ben Home", color=TEAL, zorder=3)
ax1.bar([0, 1], trd_s, bottom=np.array(ord_s) + np.array(ben_s), width=0.5, label="Third Currency", color=AMBER, zorder=3)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(zones)
ax1.set_ylim(0, 114)
ax1.set_ylabel("Share of Zone Transactions (%)")
add_grid(ax1)
ax1.legend(loc="upper center", ncol=3, fontsize=8.3)
ax1.set_title("88.5% Ordering-Home Currency Dominance", loc="left")
ax1.text(0, 48, f"{ord_s[0]:.2f}%\nShared", ha="center", color="white", fontweight="bold")
ax1.text(1, 44, f"{ord_s[1]:.2f}%\nOrd Home", ha="center", color="white", fontweight="bold")
ax1.text(1, 90.8, "5.0%", ha="center", color="white", fontsize=8.0, fontweight="bold")
ax1.text(1, 96.5, "6.5%", ha="center", color="#78350F", fontsize=8.0, fontweight="bold")

t3 = pd.Series(fc["top_third_currencies"]).head(6).iloc[::-1]
ax2.barh(np.arange(len(t3)), t3.values / 1000.0, height=0.6, color=AMBER, zorder=3)
ax2.set_yticks(np.arange(len(t3)))
ax2.set_yticklabels(t3.index)
ax2.set_xlabel("Cross-Zone Vehicle Currency Count (Thousands)")
ax2.set_xlim(0, 225)
add_grid(ax2, "x")
ax2.set_title("EUR & USD Lead Third-Currency Settlements", loc="left")
for i, v in enumerate(t3.values):
    ax2.text(v / 1000.0 + 4, i, f"{v:,}", va="center", fontsize=8.6)
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c11_currency_choice_rules.png")

# ==============================================================================
# C12: EXACT FX TRANSITION VERIFICATION
# ==============================================================================
fx_c = A["fx_counts"]
fx_k = ["0", "1", "2", "3", "4", "5+"]
fx_v = [fx_c["0"], fx_c["1"], fx_c["2"], fx_c["3"], fx_c["4"], sum(v for k, v in fx_c.items() if int(k) >= 5)]
fx_pct = [100.0 * v / M["row_count"] for v in fx_v]
fig, ax = plt.subplots(figsize=(7.6, 3.8))
ax.bar(np.arange(len(fx_k)), fx_pct, width=0.56, color=[TEAL, NAVY, BLUE, SLATE, SLATE, LIGHT_SLATE], zorder=3)
ax.set_xticks(np.arange(len(fx_k)))
ax.set_xticklabels([f"fx = {k}" for k in fx_k])
ax.set_ylabel("Share of 10M Transactions (%)")
ax.set_ylim(0, 69)
add_grid(ax)
ax.set_title("Adjacent-Hop Home-Currency Law Explains 9,999,999 / 10,000,000 Rows", loc="left")
for i, (pct, cnt) in enumerate(zip(fx_pct, fx_v)):
    ax.text(i, pct + 1.3, f"{pct:.2f}%\n({cnt/1e6:.2f}M)" if cnt >= 1e5 else f"{pct:.2f}%\n({cnt:,})",
            ha="center", va="bottom", fontsize=8.3, fontweight="bold")
ax.annotate("Exact Deterministic Formula:\nfx = sum I(HomeCurr(BIC_i) != HomeCurr(BIC_{i+1}))\n100.0% Match Across Direct + N>=3 + QFLAG='N'\n(1 sample diff: Eritrea ER modal mode -> ERN = 10M/10M)",
            xy=(2, 22), xytext=(2.05, 41), fontsize=8.3,
            bbox=dict(boxstyle="round,pad=0.38", fc="#F0FDFA", ec=TEAL, lw=1.0), color=NAVY)
save_fig(fig, "c12_fx_transition_proof.png")

# ==============================================================================
# C13: GEOMETRIC HOP SURVIVAL
# ==============================================================================
pdist = pd.DataFrame(M["routing"]["participant_dist"])
p_show = pdist[pdist["participant_count"] <= 10].copy()
x_n = p_show["participant_count"].to_numpy()
theo_pct = 100.0 * (1.0 - np.exp(-1.0)) * np.exp(-(x_n - 2.0))
fig, ax = plt.subplots(figsize=(8.0, 4.0))
ax.bar(x_n, 100.0 * p_show["direct"] / M["row_count"], width=0.62, label="Direct (N=2 Only)", color=NAVY, zorder=3)
ax.bar(x_n, 100.0 * p_show["indirect"] / M["row_count"], bottom=100.0 * p_show["direct"] / M["row_count"], width=0.62, label="Indirect (N>=3)", color=BLUE, zorder=3)
ax.bar(x_n, 100.0 * p_show["covered"] / M["row_count"], bottom=100.0 * (p_show["direct"] + p_show["indirect"]) / M["row_count"], width=0.62, label="Covered (N>=3)", color=TEAL, zorder=3)
ax.plot(x_n, theo_pct, color=RED, marker="D", ms=5, lw=1.8, ls="--", label="Exact Fit: (N-2) = floor(Exp(1))  [ratio = 1/e = 0.3679]", zorder=5)
ax.set_yscale("log")
ax.set_ylim(0.01, 130)
ax.set_xticks(x_n)
ax.set_xlabel("PARTICIPANT_COUNT (N)")
ax.set_ylabel("Transaction Share (%, Log Scale)")
add_grid(ax)
ax.legend(loc="upper right", fontsize=8.5)
ax.set_title("Exact Memoryless Hop Law: N - 2 = floor(Exp(1)) Across All 10M Rows", loc="left")
for xi, pv_val in zip(x_n[:6], p_show["pct"].iloc[:6]):
    ax.text(xi, pv_val * 1.28, f"{pv_val:.2f}%", ha="center", fontsize=8.3, fontweight="bold")
save_fig(fig, "c13_participant_geometric_law.png")

# ==============================================================================
# C14: COVERED BANK CONTROL & CLEARING HUBS
# ==============================================================================
obc = pd.read_csv("work/chart_data/ord_bank_covered_rates.csv")
mid_cur = M["routing"]["mid_by_curr_pct"]
mid_glob = M["routing"]["mid_global_top10_pct"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.9))
ax1.hist(100.0 * obc["cov_rate"], bins=25, color=TEAL, edgecolor="white", zorder=3)
ax1.set_xlabel("Ordering Bank Covered Share Among N>=3 (%, n>=200)")
ax1.set_ylabel("Number of Ordering Banks")
add_grid(ax1)
ax1.axvline(11.31, color=RED, ls="--", lw=1.4, label="Global Multi-Hop Avg: 11.3%")
ax1.legend(loc="upper right")
ax1.set_title("Covered Policy Set at Ordering Bank (0–95%)", loc="left")
ax1.annotate(f"R2 Explained:\nOrd BIC8 = {M['routing']['cov_r2_ord_bic8']:.3f}\nBen BIC8 = {M['routing']['cov_r2_ben_bic8']:.3f}\nCountry  = {M['routing']['cov_r2_ord_country']:.3f}",
             xy=(35, 350), xytext=(42, 720),
             fontsize=8.4, bbox=dict(boxstyle="round,pad=0.35", fc="#F8FAFC", ec=NAVY, lw=0.9))

hub_pairs = [("US", "USD", "US Hub\n(USD)"), ("CH", "CHF", "CH Hub\n(CHF)"), ("GB", "GBP", "GB Hub\n(GBP)"), ("CN", "CNY", "CN Hub\n(CNY)")]
xh = np.arange(len(hub_pairs))
base_vals = [mid_glob[h[0]] for h in hub_pairs]
curr_vals = [mid_cur[h[1]][h[0]] for h in hub_pairs]
ax2.bar(xh - 0.18, base_vals, width=0.36, label="Overall N=3 Share (%)", color=SLATE, zorder=3)
ax2.bar(xh + 0.18, curr_vals, width=0.36, label="When Tx Currency = Home", color=BLUE, zorder=3)
ax2.set_xticks(xh)
ax2.set_xticklabels([h[2] for h in hub_pairs], fontsize=8.8)
ax2.set_ylabel("Share of N=3 Middle Intermediary (%)")
ax2.set_ylim(0, 26.5)
add_grid(ax2)
ax2.legend(loc="upper right")
ax2.set_title("Home-Currency Clearing Boost at Hubs", loc="left")
for i, (bv, cv) in enumerate(zip(base_vals, curr_vals)):
    ax2.text(i + 0.18, cv + 0.6, f"+{cv - bv:.1f}pp", ha="center", fontsize=8.4, fontweight="bold", color=BLUE)
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c14_covered_bank_bimodality.png")

# ==============================================================================
# C15: NEGBIN GAPS & INDEPENDENT 5% ENDPOINT MISMATCHES
# ==============================================================================
dfit = pd.DataFrame(M["routing"]["dots_fit"])
pc_checks = M["path_checks"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.9), gridspec_kw={"width_ratios": [1.3, 1.0]})
sub_d1 = dfit[dfit["n_dots"].isin([1, 2, 3])]
x_p = np.arange(4)
for k_dot, col, off in [(1, NAVY, -0.22), (2, BLUE, 0.0), (3, TEAL, 0.22)]:
    s_k = sub_d1[sub_d1["n_dots"] == k_dot].sort_values("participant_count")
    ax1.bar(x_p + off, s_k["obs_pct"], width=0.21, color=col, label=f"Obs {k_dot} Gap(s)", zorder=3)
    ax1.scatter(x_p + off, s_k["theo_pct"], color=RED, marker="x", s=45, zorder=5, label="NegBin(N-1, 0.95)" if k_dot == 1 else "")
ax1.set_xticks(x_p)
ax1.set_xticklabels(["N = 2\n(1 Edge)", "N = 3\n(2 Edges)", "N = 4\n(3 Edges)", "N = 5\n(4 Edges)"])
ax1.set_ylabel("Transaction Share (%)")
ax1.set_ylim(0, 19.8)
add_grid(ax1)
ax1.legend(loc="upper left", ncol=2, fontsize=8.2)
ax1.set_title("Edge '...' Gaps Match NegBin(N-1, 0.95)", loc="left")

ep_labels = ["Any Start\nMismatch", "Any End\nMismatch", "Both Mismatched\n(Observed)", "Expected If\nIndependent"]
ep_vals = [
    100.0 * pc_checks["start_mis_rate"],
    100.0 * pc_checks["end_mis_rate"],
    100.0 * pc_checks["both_mis_rate"],
    100.0 * pc_checks["expected_independent_both_rate"],
]
ax2.bar([0, 1, 2, 3], ep_vals, width=0.52, color=[BLUE, AMBER, RED, SLATE], zorder=3)
ax2.set_xticks([0, 1, 2, 3])
ax2.set_xticklabels(ep_labels, fontsize=8.3)
ax2.set_ylabel("Rate Across 10M Rows (%)")
ax2.set_ylim(0, 6.2)
add_grid(ax2)
ax2.set_title("Independent 5.0% Path-Endpoint Noise", loc="left")
for i, v in enumerate(ep_vals):
    ax2.text(i, v + 0.15, f"{v:.3f}%" if v < 1 else f"{v:.2f}%", ha="center", fontsize=8.4, fontweight="bold")
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c15_dots_negbin_and_endpoints.png")

# ==============================================================================
# C16: QFLAG DURATION CONFOUNDER PROOF
# ==============================================================================
qdc = pd.DataFrame(M["qflag_duration_confounder"])
ind_q = qdc[qdc["tx_type"].isin(["Direct", "Indirect"])].copy()
sub_ind = qdc[qdc["tx_type"] == "Indirect"]
raw_y = (sub_ind[sub_ind["qflag"] == "Y"]["mean_dur"] * sub_ind[sub_ind["qflag"] == "Y"]["n"]).sum() / sub_ind[sub_ind["qflag"] == "Y"]["n"].sum()
raw_n = (sub_ind[sub_ind["qflag"] == "N"]["mean_dur"] * sub_ind[sub_ind["qflag"] == "N"]["n"]).sum() / sub_ind[sub_ind["qflag"] == "N"]["n"].sum()
raw_diff = raw_n - raw_y

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.8), gridspec_kw={"width_ratios": [0.85, 1.35]})
ax1.bar([0, 1], [raw_y, raw_n], width=0.46, color=[TEAL, RED], zorder=3)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["QFLAG='Y'\n(Indirect N=3..6)", "QFLAG='N'\n(Indirect N=3..6)"])
ax1.set_ylabel("Pooled Mean Duration (s)")
ax1.set_ylim(0, 1180)
add_grid(ax1)
ax1.set_title(f"Naive Pooled Gap: +{raw_diff:.0f}s Slower", loc="left")
ax1.text(0, raw_y + 20, f"{raw_y:.0f}s", ha="center", fontweight="bold")
ax1.text(1, raw_n + 20, f"{raw_n:.0f}s\n(+{raw_diff:.0f}s)", ha="center", fontweight="bold", color=RED)

ns_show = [2, 3, 4, 5, 6]
qy_vals = [ind_q[(ind_q["participant_count"] == k) & (ind_q["qflag"] == "Y")]["mean_dur"].values[0] for k in ns_show]
qn_vals = [ind_q[(ind_q["participant_count"] == k) & (ind_q["qflag"] == "N")]["mean_dur"].values[0] for k in ns_show]
ax2.plot(ns_show, qy_vals, marker="o", lw=2.3, color=TEAL, label="QFLAG='Y' (Complete Path)", zorder=3)
ax2.plot(ns_show, qn_vals, marker="s", lw=2.0, ls="--", color=RED, label="QFLAG='N' ('...' Gap Present)", zorder=4)
ax2.set_xticks(ns_show)
ax2.set_xticklabels([f"N={k}" for k in ns_show])
ax2.set_ylabel("Mean Settlement Duration (s)")
ax2.set_ylim(0, 1680)
add_grid(ax2)
ax2.legend(loc="upper left")
ax2.set_title("Stratified by N: Zero Causal Effect (<9s Diff)", loc="left")
for k, vy, vn in zip(ns_show, qy_vals, qn_vals):
    ax2.text(k, max(vy, vn) + 45, f"{vy:.0f}s vs {vn:.0f}s", ha="center", fontsize=8.0)
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c16_qflag_confounder_null.png")

# ==============================================================================
# C17: TRIMODAL DURATION & GAMMA(2,5) NOISE
# ==============================================================================
tdh = pd.read_csv("work/chart_data/dur_global_trimodal_hist.csv")
tdm = 0.5 * (tdh["bin_left"] + tdh["bin_right"])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.9))
ax1.plot(tdm, tdh["direct_count"] / 1e5, lw=2.1, color=NAVY, label="Direct (N=2)")
ax1.plot(tdm, tdh["indirect_count"] / 1e5, lw=2.1, color=BLUE, label="Indirect (N>=3)")
ax1.plot(tdm, tdh["covered_count"] / 1e5, lw=2.1, color=TEAL, label="Covered (N>=3)")
ax1.set_xlabel("log10(TOTAL_DURATION in Seconds)")
ax1.set_ylabel("Transactions / Bin (100k Units)")
ax1.set_xlim(0.4, 4.3)
ax1.set_ylim(0, 3.65)
add_grid(ax1)
ax1.legend(loc="upper right")
ax1.set_title("Global Trimodal Mixture (~25s / ~400s / ~2k s)", loc="left")
ax1.annotate("Direct Pair FE R2 = 0.99996\n(RMSE = 6.9–7.0s!)\nBank Add FE R2 = 0.9933",
             xy=(2.3, 1.5), xytext=(1.95, 2.15), fontsize=8.2,
             bbox=dict(boxstyle="round,pad=0.35", fc="#F8FAFC", ec=NAVY, lw=0.9))

exc = pair_0001_vals - pair_0001_vals.min()
bins_g = np.arange(0, 42, 1)
ax2.hist(exc, bins=bins_g, density=True, color=BLUE, alpha=0.65, edgecolor="white", label=f"Observed Excess (n={len(exc):,})", zorder=3)
xg = np.linspace(0.1, 40, 200)
ax2.plot(xg, stats.gamma.pdf(xg, a=2.0, scale=5.0), color=RED, lw=2.2, label="Gamma(k=2, theta=5): mean=10s, var=50", zorder=4)
ax2.set_xlabel("Seconds Above Pair Base (D - 1,586s) on 0001MQM0->0009TNT0")
ax2.set_ylabel("Density")
ax2.set_xlim(0, 40)
ax2.set_ylim(0, 0.082)
add_grid(ax2)
ax2.legend(loc="upper right", fontsize=8.2)
ax2.set_title("Within-Pair Delay Noise Follows Gamma(2, 5)", loc="left")
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c17_duration_trimodal_and_pairs.png")

# ==============================================================================
# C18: N=3 MULTI-HOP COMPOSITION & COVERED 35% SPEEDUP
# ==============================================================================
p3_s = p3_sample_df["bic8_path"].str.split(";", expand=True)
p3_sample_df["A"] = p3_s[0]
p3_sample_df["B"] = p3_s[1]
p3_sample_df["C"] = p3_s[2]
p3_sample_df["d_AB"] = [pair_mean_map.get((a, b), np.nan) for a, b in zip(p3_sample_df["A"], p3_sample_df["B"])]
p3_sample_df["d_BC"] = [pair_mean_map.get((b, c), np.nan) for b, c in zip(p3_sample_df["B"], p3_sample_df["C"])]
p3_sc = p3_sample_df.dropna(subset=["d_AB", "d_BC"]).copy()
p3_sc["sum_AB_BC"] = p3_sc["d_AB"] + p3_sc["d_BC"]

cov_cmp = pd.read_csv("work/chart_data/covered_vs_indirect_bank_dur.csv").head(8).iloc[::-1]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.9))
ind_pts = p3_sc[p3_sc["tx_type"] == "Indirect"].sample(n=min(3500, (p3_sc["tx_type"] == "Indirect").sum()), random_state=42)
cov_pts = p3_sc[p3_sc["tx_type"] == "Covered"].sample(n=min(1500, (p3_sc["tx_type"] == "Covered").sum()), random_state=42)
ax1.scatter(ind_pts["sum_AB_BC"], ind_pts["total_duration"], s=8, alpha=0.35, color=BLUE, label="Indirect (r=0.982)", zorder=3)
ax1.scatter(cov_pts["sum_AB_BC"], cov_pts["total_duration"], s=8, alpha=0.45, color=TEAL, label="Covered (r=0.948)", zorder=4)
ax1.plot([0, 5000], [0, 5000], color=SLATE, ls="--", lw=1.2, label="1:1 Line (D_AB + D_BC)")
ax1.set_xlim(0, 5000)
ax1.set_ylim(0, 5000)
ax1.set_xlabel("Sum of Direct Edge Means: D(A,B) + D(B,C) (s)")
ax1.set_ylabel("Observed N=3 Duration (s)")
add_grid(ax1)
ax1.legend(loc="upper left", fontsize=8.3)
ax1.set_title("N=3 Latency Scales with Edges (R2=0.992)", loc="left")

yb = np.arange(len(cov_cmp))
ax2.barh(yb + 0.18, cov_cmp["mean_ind"], height=0.34, color=BLUE, label="Indirect N=3 Mean (s)", zorder=3)
ax2.barh(yb - 0.18, cov_cmp["mean_cov"], height=0.34, color=TEAL, label="Covered N=3 Mean (s)", zorder=4)
ax2.set_yticks(yb)
ax2.set_yticklabels(cov_cmp["bic8"], fontsize=8.5)
ax2.set_xlim(0, 1180)
ax2.set_ylim(-0.6, 8.8)
ax2.set_xlabel("Mean Duration at Same Ordering Bank & N=3 (s)")
add_grid(ax2, "x")
ax2.legend(loc="upper center", ncol=2, fontsize=8.2)
ax2.set_title("Covered Cuts Latency by ~35% (0.62–0.71x)", loc="left")
for i, (_, r) in enumerate(cov_cmp.iterrows()):
    ax2.text(r["mean_ind"] + 15, i, f"{r['ratio']:.2f}x", va="center", fontsize=8.1, fontweight="bold", color=TEAL)
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c18_duration_n3_and_covered_speedup.png")
del p3_sample_df, p3_sc

# ==============================================================================
# C19: ORDERING VS BENEFICIARY DETAIL COUNTRY MATCH
# ==============================================================================
pd_m = M["party_details"]
ben_inf = pd.Series(pd_m["top15_ben_inferred_cc_pct"]).head(10).iloc[::-1]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.8), gridspec_kw={"width_ratios": [0.85, 1.3]})
ax1.bar([0, 1], [pd_m["ord_cc_match_among_nonempty_pct"], pd_m["ben_cc_match_among_nonempty_pct"]],
        width=0.5, color=[TEAL, RED], zorder=3)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(["Ordering Party\n(8.91M w/ CC)", "Beneficiary Party\n(8.96M w/ CC)"])
ax1.set_ylim(0, 116)
ax1.set_ylabel("Text Country == BIC8 Country (%)")
add_grid(ax1)
ax1.set_title("Ord 100% vs Ben 0.98% Geo-Match", loc="left")
ax1.text(0, 102.5, "100.00%\nExact Match", ha="center", fontweight="bold", color=TEAL)
ax1.text(1, 4.0, "0.98%\nRandom Locale!", ha="center", fontweight="bold", color=RED)

yb2 = np.arange(len(ben_inf))
ax2.barh(yb2, ben_inf.values, height=0.62, color=AMBER, zorder=3)
ax2.set_yticks(yb2)
ax2.set_yticklabels(ben_inf.index, fontsize=9.0, fontweight="bold")
ax2.set_xlabel("Share of Inferred Beneficiary Detail Addresses (%)")
ax2.set_xlim(0, 17.5)
add_grid(ax2, "x")
ax2.set_title("Beneficiary Faker Skews to Populous Locales", loc="left")
for i, v in enumerate(ben_inf.values):
    ax2.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=8.4)
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c19_details_country_mismatch.png")

# ==============================================================================
# C20: IBAN BOUNDARY & FINANCE-WORD GRAMMAR
# ==============================================================================
ib_df = pd.DataFrame(pd_m["iban_by_country"])
show_ib = pd.concat([ib_df.head(6), ib_df.tail(6)], ignore_index=True)
fw_tot = {k: pd_m["ord_fin_words"][k] + pd_m["ben_fin_words"][k] for k in pd_m["ord_fin_words"]}
fw_s = pd.Series(fw_tot).sort_values()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.9))
x_i = np.arange(len(show_ib))
ax1.bar(x_i, show_ib["ord_iban_pct"], width=0.58, color=[TEAL if v > 50 else LIGHT_SLATE for v in show_ib["ord_iban_pct"]], zorder=3)
ax1.set_xticks(x_i)
ax1.set_xticklabels(show_ib["country"], fontsize=8.5)
ax1.set_ylim(0, 112)
ax1.set_ylabel("Ordering Account Has IBAN (%)")
add_grid(ax1)
ax1.set_title("IBAN Boundary: EU/CH ~92% vs US/APAC 0%", loc="left")
ax1.axvline(5.5, color=SLATE, ls="--", lw=1.0)
ax1.text(2.5, 99, "IBAN Countries (90–96%)", ha="center", fontsize=8.4, fontweight="bold", color=TEAL)
ax1.text(8.5, 14, "Non-IBAN (Exact 0.00%)", ha="center", fontsize=8.4, fontweight="bold", color=SLATE)

ax2.barh(np.arange(len(fw_s)), fw_s.values / 1000.0, height=0.62, color=NAVY, zorder=3)
ax2.set_yticks(np.arange(len(fw_s)))
ax2.set_yticklabels(fw_s.index, fontsize=8.8)
ax2.set_xlim(0, 275)
ax2.set_xlabel("Total Occurrences Across Ord + Ben Details (k)")
add_grid(ax2, "x")
ax2.set_title("10 Uniform <finance_word> Tokens", loc="left")
ax2.annotate("Unstructured Text: 9.42%\n1/2/3/ Structured: 0.002%\nUniform ~159k each (1/10)",
             xy=(160, 4.5), xytext=(170, 3.2), fontsize=8.1,
             bbox=dict(boxstyle="round,pad=0.35", fc="#F1F5F9", ec=BLUE, lw=0.9), color=NAVY)
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c20_iban_structure_finwords.png")

# ==============================================================================
# C21: ROMANCE SCAM ESCALATION CURVE
# ==============================================================================
rom_s = pd.read_csv("work/chart_data/romance_scam_steps.csv")
steps = rom_s["step"].to_numpy()
slope_ln, intercept_ln = np.polyfit(steps - 1, np.log(rom_s["median_usd"].to_numpy()), 1)
step_factor = float(np.exp(slope_ln))
fitted_rom = np.exp(intercept_ln) * (step_factor ** (steps - 1))
base150 = rom_s["median_usd"].iloc[0] * (1.50 ** (steps - 1))

fig, ax = plt.subplots(figsize=(7.9, 3.9))
ax.plot(steps, rom_s["median_usd"], marker="o", ms=6.2, lw=2.3, color=RED, label="Observed Step Median ($)", zorder=4)
ax.fill_between(steps, rom_s["p25_usd"], rom_s["p75_usd"], color=RED, alpha=0.14, label="Interquartile Range (p25–p75)")
ax.plot(steps, base150, ls="--", lw=1.9, color=NAVY, label="Within-Chain Rule: 1.50x/step from Step 1 ($33)", zorder=5)
ax.plot(steps, fitted_rom, ls=":", lw=1.8, color=BLUE, label=f"Cross-Sectional Log-Linear Fit ({step_factor:.2f}x/step)", zorder=5)
ax.set_yscale("log")
ax.set_xticks(steps)
ax.set_ylim(2, 22000)
ax.set_xlabel("Sequential Transfer Index Within Victim Chain (n = 500 Victims)")
ax.set_ylabel("Payment Amount (USD, Log Scale)")
add_grid(ax)
ax.legend(loc="upper left", fontsize=8.3)
ax.set_title("Romance Scam Grooming Escalation (1.50x Within-Chain, Survivor Effect at Step 8)", loc="left")
for s_i, med_v, n_v in zip(steps, rom_s["median_usd"], rom_s["n"]):
    ax.text(s_i, med_v * 1.4, f"${med_v:,.0f}\n(n={int(n_v)})", ha="center", fontsize=7.6, fontweight="bold")
save_fig(fig, "c21_romance_scam_steps.png")

# ==============================================================================
# C22: SHARED MULE POOL & DESTINATION LIFT
# ==============================================================================
mv = M["anomalies"]["mule_venn"]
mcl = pd.DataFrame(M["anomalies"]["mule_country_lift"]).head(8).iloc[::-1]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.9))
ov_cats = ["CEO\nOnly\n(327)", "ROM\nOnly\n(197)", "ARS\nOnly\n(192)", "CEO &\nARS\n(171)", "CEO &\nROM\n(82)", "All 3\nScams\n(54)", "ARS &\nROM\n(45)"]
ov_vals = [mv["ceo_only"], mv["rom_only"], mv["ars_only"], mv["ceo_ars_only"], mv["ceo_rom_only"], mv["all_three"], mv["ars_rom_only"]]
ax1.bar(np.arange(7), ov_vals, width=0.58, color=[BLUE, AMBER, TEAL, RED, RED, PURPLE, RED], zorder=3)
ax1.set_xticks(np.arange(7))
ax1.set_xticklabels(ov_cats, fontsize=8.1)
ax1.set_ylim(0, 385)
ax1.set_ylabel("Unique Beneficiary Accounts (Union=1,068)")
add_grid(ax1)
ax1.set_title("1,068 Shared Mules Link Three Scams", loc="left")
ax1.annotate("441 of 1,068 mule accounts\nalso receive 7,828 'Normal' txs", xy=(5, 65), xytext=(2.8, 250),
             fontsize=8.3, bbox=dict(boxstyle="round,pad=0.35", fc="#FEF2F2", ec=RED, lw=0.9), color=NAVY)

ax2.barh(np.arange(len(mcl)), mcl["lift"], height=0.62, color=[RED if l > 20 else AMBER for l in mcl["lift"]], zorder=3)
ax2.set_yticks(np.arange(len(mcl)))
ax2.set_yticklabels([f"{r['country']} ({r['mule_pct']:.1f}% vs {r['normal_pct']:.2f}%)" for _, r in mcl.iterrows()], fontsize=8.3)
ax2.set_xlabel("Concentration Lift vs. Normal Baseline (x)")
ax2.set_xlim(0, 265)
add_grid(ax2, "x")
ax2.set_title("CEO & ARS Mule Destination Lift vs Normal", loc="left")
for i, v in enumerate(mcl["lift"]):
    ax2.text(v + 5, i, f"{v:.1f}x", va="center", fontsize=8.3, fontweight="bold")
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c22_mule_pool_venn_countries.png")

# ==============================================================================
# C23: FAT FINGER SHIFTS & COVERED SUPPRESSION
# ==============================================================================
ffh = pd.read_csv("work/chart_data/fat_finger_log10_hist.csv")
ff_mid = 0.5 * (ffh["bin_left"] + ffh["bin_right"])
asum = pd.DataFrame(M["anomalies"]["summary_table"])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.9))
ax1.plot(ff_mid, ffh["normal_density"], lw=2.0, color=NAVY, label="Normal Clean Baseline (max $1.40B)")
ax1.plot(ff_mid, ffh["fat_finger_density"], lw=2.1, color=RED, label="FAT_FINGER (max $15.66B, +1.57 log10)")
ax1.fill_between(ff_mid, ffh["normal_density"], ffh["fat_finger_density"],
                 where=(ffh["fat_finger_density"] > ffh["normal_density"]), color=RED, alpha=0.18)
ax1.set_xlabel("log10(USD_AMOUNT)")
ax1.set_ylabel("Density")
ax1.set_xlim(-2, 11)
ax1.set_ylim(0, 0.39)
add_grid(ax1)
ax1.legend(loc="upper right", fontsize=8.1)
ax1.set_title("Fat-Finger 10^k Pre-FX Shifts (USD Int 77.5%)", loc="left")

scam_order = ["NORMAL", "FAT_FINGER_MISTAKE", "DUPLICATE_PAYMENT", "CUSTOMER_RECALL", "CEO_SCAM", "ACCOUNT_REPLACEMENT_SCAM", "ROMANCE_SCAM"]
sub_s = asum.set_index("anomaly").loc[scam_order].reset_index()
short_lbl = ["Normal", "Fat\nFinger", "Duplicate", "Recall*", "CEO\nScam", "Acct\nRepl.", "Romance\nScam"]
ax2.bar(np.arange(7), sub_s["covered_among_multihop_pct"], width=0.58, color=[NAVY, SLATE, SLATE, PURPLE, RED, RED, RED], zorder=3)
ax2.set_xticks(np.arange(7))
ax2.set_xticklabels(short_lbl, fontsize=8.2)
ax2.set_ylim(0, 16)
ax2.set_ylabel("Covered Share Among N>=3 Multi-Hop (%)")
add_grid(ax2)
ax2.set_title("Social Engineering Suppresses Covered 6–37x", loc="left")
for i, v in enumerate(sub_s["covered_among_multihop_pct"]):
    ax2.text(i, v + 0.35, f"{v:.1f}%", ha="center", fontsize=8.2, fontweight="bold")
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c23_fat_finger_and_scam_routing.png")

# ==============================================================================
# C24: UNDOCUMENTED CUSTOMER_RECALL NULL SIGNATURE
# ==============================================================================
ks_d = M["anomalies"]["customer_recall_ks"]
feat_lbls = {"usd_amount": "USD Amount", "total_duration": "Duration", "epoch_seconds": "Timestamp",
             "fx_count": "FX Count", "participant_count": "Participants", "ord_det_len": "Ord Text Len", "ben_det_len": "Ben Text Len"}
ks_keys = list(feat_lbls.keys())
p_vals = [ks_d[k]["ks_p"] for k in ks_keys]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 3.8))
ax1.barh(np.arange(len(ks_keys)), p_vals, height=0.58, color=PURPLE, alpha=0.85, zorder=3)
ax1.axvline(0.05, color=RED, ls="--", lw=1.4, label="p = 0.05 Cutoff")
ax1.set_yticks(np.arange(len(ks_keys)))
ax1.set_yticklabels([feat_lbls[k] for k in ks_keys], fontsize=8.8)
ax1.set_xlim(0, 1.08)
ax1.set_xlabel("Two-Sample KS p-value (n=317 vs 200k Normal)")
add_grid(ax1, "x")
ax1.legend(loc="upper right", fontsize=8.4)
ax1.set_title("Core Features Fail KS Test (p = 0.23–0.90)", loc="left")
for i, pv_val in enumerate(p_vals):
    ax1.text(pv_val + 0.02, i, f"p={pv_val:.2f}", va="center", fontsize=8.2)

ax2.plot(fpr_cr, tpr_cr, color=PURPLE, lw=2.3, label=f"5-Fold LightGBM (OOF AUC = {auc_cr:.3f})")
ax2.plot([0, 1], [0, 1], color=SLATE, ls="--", lw=1.3, label="Random Chance (AUC = 0.500)")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
add_grid(ax2)
ax2.legend(loc="lower right", fontsize=8.4)
ax2.set_title(f"16-Feature GBDT Yields Chance AUC = {auc_cr:.3f}", loc="left")
ax2.annotate("Sole marginal flag:\nQFLAG='N' = 11.04% vs 7.64%\n(Chi2 p=0.023, insufficient\nfor operational detection)",
             xy=(0.28, 0.25), xytext=(0.43, 0.23), fontsize=8.1,
             bbox=dict(boxstyle="round,pad=0.35", fc="#FAF5FF", ec=PURPLE, lw=0.9), color=NAVY)
fig.tight_layout(w_pad=2.8)
save_fig(fig, "c24_customer_recall_null.png")

print(f"Regenerated all 24 polished charts cleanly in {time.time() - t0:.1f}s")
