import time
import numpy as np
import pandas as pd
from scipy import stats

t0 = time.time()
cols = [
    "row_id", "epoch_seconds", "currency_code", "usd_amount",
    "ordering_bic8", "beneficiary_bic8", "bic8_path",
    "ord_bic_country", "ben_bic_country",
    "total_duration", "continuity_qflag", "fx_count", "participant_count", "tx_type",
    "ordering_account", "beneficiary_account",
    "ordering_type", "beneficiary_type", "anomaly_type",
    "ord_inferred_cc", "ben_inferred_cc",
    "ord_fin_word", "ben_fin_word",
    "ord_has_iban", "ben_has_iban",
    "ord_is_struct_123", "ben_is_struct_123",
    "ord_has_lb35", "ben_has_lb35",
    "ord_has_trunc", "ben_has_trunc",
    "ord_det_len", "ben_det_len", "ord_n_lines", "ben_n_lines",
]
df = pd.read_parquet("work/core_parquet", columns=cols)
clean = df[df["anomaly_type"] == ""].copy()
print(f"Loaded in {time.time()-t0:.1f}s")

# === 1. TOTAL_DURATION EXACT ADDITIVE BANK MODEL ===
print("\n=== 1. TOTAL_DURATION BANK-LEVEL MODEL ===")
d2_clean = clean[(clean["participant_count"] == 2) & (clean["bic8_path"].str.slice(0, 8) == clean["ordering_bic8"]) & (clean["bic8_path"].str.slice(-8) == clean["beneficiary_bic8"])].copy()
top_pairs = d2_clean.groupby(["ordering_bic8", "beneficiary_bic8"]).size().sort_values(ascending=False).head(5)
for (ob, bb), n in top_pairs.items():
    s = d2_clean[(d2_clean["ordering_bic8"] == ob) & (d2_clean["beneficiary_bic8"] == bb)]["total_duration"]
    print(f"  Direct pair {ob}->{bb} (n={n}): min={s.min()}, p5={s.quantile(0.05):.1f}, med={s.median():.1f}, mean={s.mean():.1f}, p95={s.quantile(0.95):.1f}, max={s.max()}, std={s.std():.1f}")

# Fit iterative additive node model on Direct: total_duration ~ a[ord_bic] + b[ben_bic] (+ fx?)
# First check if fx_count (0 vs 1) changes duration WITHIN same (ord_bic, ben_bic)!
pair_fx = d2_clean.groupby(["ordering_bic8", "beneficiary_bic8"])["fx_count"].nunique()
mixed_fx_pairs = pair_fx[pair_fx > 1].index[:5]
print("Within same Direct (ord_bic, ben_bic) pair, fx=0 vs fx=1 duration:")
sub_m = d2_clean.set_index(["ordering_bic8", "beneficiary_bic8"]).loc[mixed_fx_pairs].reset_index()
print(sub_m.groupby(["ordering_bic8", "beneficiary_bic8", "fx_count"])["total_duration"].agg(["count", "mean", "median", "std"]).head(12))

# Compute R2 of additive (ord_bic8 + ben_bic8) on Direct total_duration & log(total_duration)!
s500k = d2_clean.sample(500_000, random_state=42).copy()
# Simple alternating projections for 10 iterations to get exact OLS additive fixed effects!
y = s500k["total_duration"].to_numpy(dtype=np.float64)
y_log = np.log(y)
ob_codes, _ = pd.factorize(s500k["ordering_bic8"])
bb_codes, _ = pd.factorize(s500k["beneficiary_bic8"])
for label, target in [("TOTAL_DURATION (linear)", y), ("ln(TOTAL_DURATION)", y_log)]:
    a = np.zeros(ob_codes.max() + 1)
    b = np.zeros(bb_codes.max() + 1)
    mu = target.mean()
    res = target - mu
    cnt_a = np.bincount(ob_codes)
    cnt_b = np.bincount(bb_codes)
    for _ in range(15):
        a = np.bincount(ob_codes, weights=target - mu - b[bb_codes]) / np.maximum(cnt_a, 1)
        b = np.bincount(bb_codes, weights=target - mu - a[ob_codes]) / np.maximum(cnt_b, 1)
    pred = mu + a[ob_codes] + b[bb_codes]
    r2 = 1.0 - np.var(target - pred) / np.var(target)
    rmse = np.sqrt(np.mean((target - pred) ** 2))
    print(f"  Direct {label} ~ a[ord_bic8] + b[ben_bic8]: R2={r2:.5f}, RMSE={rmse:.3f}, std(a)={a.std():.2f}, std(b)={b.std():.2f}")

# Also check if a[bic] == b[bic] (symmetric node latency) or asymmetric!
# Compare bank mean as sender vs bank mean as receiver (after subtracting grand mean)
s500k["a_est"] = a[ob_codes]
s500k["b_est"] = b[bb_codes]
ob_fe = s500k.groupby("ordering_bic8")["a_est"].first()
bb_fe = s500k.groupby("beneficiary_bic8")["b_est"].first()
both_fe = pd.concat([ob_fe, bb_fe], axis=1).dropna()
print("  Correlation between sender FE a[bic] and receiver FE b[bic] (in log space):", both_fe["a_est"].corr(both_fe["b_est"]))
print("  Top 5 fastest and slowest bank FEs (exp(mu+fe)):")
# Linear FE check too:
a_lin = np.zeros(ob_codes.max() + 1)
b_lin = np.zeros(bb_codes.max() + 1)
mu_lin = y.mean()
for _ in range(15):
    a_lin = np.bincount(ob_codes, weights=y - mu_lin - b_lin[bb_codes]) / np.maximum(cnt_a, 1)
    b_lin = np.bincount(bb_codes, weights=y - mu_lin - a_lin[ob_codes]) / np.maximum(cnt_b, 1)
s500k["a_lin"] = a_lin[ob_codes]
s500k["b_lin"] = b_lin[bb_codes]
both_lin = pd.concat([s500k.groupby("ordering_bic8")["a_lin"].first(), s500k.groupby("beneficiary_bic8")["b_lin"].first()], axis=1).dropna()
print("  Linear FE corr(a_lin, b_lin):", both_lin["a_lin"].corr(both_lin["b_lin"]))
print("  Sample 10 big banks linear FE (a_lin vs b_lin):")
big_bics = d2_clean["ordering_bic8"].value_counts().head(10).index
print(both_lin.loc[both_lin.index.isin(big_bics)].round(1))

# === 2. FX_COUNT: ALL PARTICIPANT_COUNTS & DIRECT MISMATCHES ===
print("\n=== 2. FX_COUNT FULL CHECK ===")
# Build country -> currency map from d2_clean same-country fx==0 + multi-edge fx==0
fx0_d2 = d2_clean[d2_clean["fx_count"] == 0]
fx0_both = pd.concat([
    fx0_d2[["ord_bic_country", "currency_code"]].rename(columns={"ord_bic_country": "cc"}),
    fx0_d2[["ben_bic_country", "currency_code"]].rename(columns={"ben_bic_country": "cc"}),
])
cc_cur = {}
for c, g in fx0_both.groupby("cc"):
    cc_cur[c] = g["currency_code"].value_counts().index[0]
for c in set(clean["ord_bic_country"].unique()) | set(clean["ben_bic_country"].unique()):
    if c not in cc_cur:
        sub = clean[clean["ord_bic_country"] == c]
        if len(sub) > 0:
            cc_cur[c] = sub["currency_code"].value_counts().index[0]

# Inspect those 12,429 Direct mismatches (US->US EUR with fx=0 etc.)!
mis_d2 = d2_clean[(d2_clean["ord_bic_country"].map(cc_cur) != d2_clean["currency_code"]) | (d2_clean["ben_bic_country"].map(cc_cur) != d2_clean["currency_code"])]
mis_fx0 = mis_d2[mis_d2["fx_count"] == 0]
print("Direct mismatches where fx==0 (n =", len(mis_fx0), "): are specific BIC8s responsible?")
print("  Top ordering_bic8 in mis_fx0:", mis_fx0["ordering_bic8"].value_counts().head(10).to_dict())
print("  Top beneficiary_bic8 in mis_fx0:", mis_fx0["beneficiary_bic8"].value_counts().head(10).to_dict())
print("  Sample 10 mis_fx0 rows:")
print(mis_fx0[["ordering_bic8", "beneficiary_bic8", "currency_code", "fx_count"]].head(10).to_string())
# Check if US->US in EUR is 100% fx==0 or tiny %!
us_us_eur = d2_clean[(d2_clean["ord_bic_country"] == "US") & (d2_clean["ben_bic_country"] == "US") & (d2_clean["currency_code"] == "EUR")]
print("  US->US in EUR fx_count dist:", us_us_eur["fx_count"].value_counts().to_dict())

# Test adjacent currency change sum(c[i] != c[i+1]) on ALL continuity_qflag=='Y' across participant_count 3..10!
y_multi = clean[(clean["continuity_qflag"] == "Y") & (clean["participant_count"] >= 3)].sample(500_000, random_state=42)
def calc_adj_diff(path):
    parts = path.split(";")
    curs = [cc_cur.get(p[4:6], p[4:6]) for p in parts]
    return sum(curs[i] != curs[i + 1] for i in range(len(curs) - 1))
adj_pred = y_multi["bic8_path"].map(calc_adj_diff)
print("Multi-hop (part>=3, QFLAG=Y, n=500k) exact match of sum(c[i] != c[i+1]):", (adj_pred == y_multi["fx_count"]).mean())
print("  By participant_count:")
y_multi = y_multi.copy()
y_multi["adj_pred"] = adj_pred
print(y_multi.groupby(y_multi["participant_count"].clip(upper=8)).apply(lambda g: (g["adj_pred"] == g["fx_count"]).mean(), include_groups=False))

# Wait: what about QFLAG=='N' (paths with '...')? Does '...' count as a gap in fx_count too?!
n_multi = clean[(clean["continuity_qflag"] == "N") & (clean["participant_count"] >= 2)].sample(200_000, random_state=42).copy()
def calc_vis_adj(path):
    parts = [p for p in path.split(";") if p != "..."]
    curs = [cc_cur.get(p[4:6], p[4:6]) for p in parts]
    return sum(curs[i] != curs[i + 1] for i in range(len(curs) - 1))
n_multi["vis_adj"] = n_multi["bic8_path"].map(calc_vis_adj)
print("QFLAG=='N' (with '...') exact match of visible adjacent diff:", (n_multi["vis_adj"] == n_multi["fx_count"]).mean())
print("QFLAG=='N' (fx_count - vis_adj) dist:", (n_multi["fx_count"] - n_multi["vis_adj"]).value_counts().head(6).to_dict())

# === 3. FAT_FINGER_MISTAKE: USD vs NON-USD & SHIFT MECHANISM ===
print("\n=== 3. FAT_FINGER_MISTAKE EXACT SHIFT ===")
ff = df[df["anomaly_type"] == "FAT_FINGER_MISTAKE"].copy()
ff["is_usd"] = ff["currency_code"] == "USD"
ff["is_int"] = (ff["usd_amount"] % 1) == 0
ff["is_mult10"] = (ff["usd_amount"] % 10) == 0
ff["is_mult100"] = (ff["usd_amount"] % 100) == 0
ff["is_mult1000"] = (ff["usd_amount"] % 1000) == 0
print("FAT_FINGER USD vs non-USD integer/multiple rates:")
print(ff.groupby("is_usd")[["is_int", "is_mult10", "is_mult100", "is_mult1000"]].agg(["count", "mean"]))

# Compute expected baseline mu(party_pair, ord_bic_country) on clean
clean_mu = clean.assign(la=np.log(clean["usd_amount"])).groupby(["ordering_type", "beneficiary_type", "ord_bic_country"])["la"].mean()
ff = ff.merge(clean_mu.rename("base_mu"), on=["ordering_type", "beneficiary_type", "ord_bic_country"], how="left")
ff["log10_shift"] = (np.log(ff["usd_amount"]) - ff["base_mu"]) / np.log(10)
print("FAT_FINGER log10_shift quantiles:", ff["log10_shift"].quantile([0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]).round(2).to_dict())
print("FAT_FINGER log10_shift < 0 rate:", (ff["log10_shift"] < 0).mean(), "mean shift:", ff["log10_shift"].mean())

# === 4. CUSTOMER_RECALL EXHAUSTIVE TEST ===
print("\n=== 4. CUSTOMER_RECALL EXHAUSTIVE COMPARISON vs NORMAL ===")
cr = df[df["anomaly_type"] == "CUSTOMER_RECALL"].copy()
bg = clean.sample(100_000, random_state=7).copy()
for col in ["usd_amount", "total_duration", "epoch_seconds", "fx_count", "participant_count", "ord_det_len", "ben_det_len"]:
    stat, p = stats.ks_2samp(cr[col], bg[col])
    print(f"  KS test {col}: stat={stat:.4f}, p={p:.4f} (cr_med={cr[col].median():.2f}, bg_med={bg[col].median():.2f})")
for col in ["tx_type", "continuity_qflag", "ordering_type", "beneficiary_type", "ord_has_iban", "ben_has_iban", "ord_is_struct_123", "ben_is_struct_123", "ord_has_trunc", "ben_has_trunc"]:
    cr_d = cr[col].value_counts(normalize=True)
    bg_d = bg[col].value_counts(normalize=True)
    # Chi-square test on counts
    cats = sorted(set(cr_d.index) | set(bg_d.index))
    obs = [int((cr[col] == c).sum()) for c in cats]
    exp = [float((bg[col] == c).mean()) * len(cr) for c in cats]
    chi2, p = stats.chisquare(obs, exp)
    print(f"  Chi2 {col}: p={p:.4f} | cr={cr_d.round(4).to_dict()} vs bg={bg_d.round(4).to_dict()}")

# Check if CUSTOMER_RECALL ordering_account / beneficiary_account overlap with ANY other anomaly!
cr_ord = set(cr["ordering_account"])
cr_ben = set(cr["beneficiary_account"])
print("  CR ord_acct in other anomalies:", df[(df["anomaly_type"] != "") & (df["anomaly_type"] != "CUSTOMER_RECALL") & (df["ordering_account"].isin(cr_ord))]["anomaly_type"].value_counts().to_dict())
print("  CR ben_acct in other anomalies:", df[(df["anomaly_type"] != "") & (df["anomaly_type"] != "CUSTOMER_RECALL") & (df["beneficiary_account"].isin(cr_ben))]["anomaly_type"].value_counts().to_dict())
# Check if CR has weird row_id distribution (uniform across 0..10M?)
print("  CR row_id min/med/max:", cr["row_id"].min(), cr["row_id"].median(), cr["row_id"].max())

# === 5. CEO_SCAM & ACCOUNT_REPLACEMENT_SCAM: AMOUNT SHIFT & MULE POOL ===
print("\n=== 5. CEO_SCAM & ACCOUNT_REPLACEMENT_SCAM: AMOUNT SHIFT & MULE POOL ===")
for atype in ["CEO_SCAM", "ACCOUNT_REPLACEMENT_SCAM", "ROMANCE_SCAM"]:
    sub = df[df["anomaly_type"] == atype].merge(clean_mu.rename("base_mu"), on=["ordering_type", "beneficiary_type", "ord_bic_country"], how="left")
    delta = np.log(sub["usd_amount"]) - sub["base_mu"]
    print(f"  {atype}: n={len(sub)}, exp(mean log ratio vs baseline)={np.exp(delta.mean()):.3f}x, median ratio={np.exp(delta.median()):.3f}x, std={delta.std():.3f}")

# Total mule pool size
all_scam_bens = set(df[df["anomaly_type"].isin(["CEO_SCAM", "ACCOUNT_REPLACEMENT_SCAM", "ROMANCE_SCAM"])]["beneficiary_account"])
print("  Total unique ben_accts across (CEO, ARS, RS):", len(all_scam_bens))
# How many of these mule accounts appear in normal tx, and in what volume?
mule_in_norm = clean[clean["beneficiary_account"].isin(all_scam_bens)]
print("  Mule ben_accts appearing in normal tx:", mule_in_norm["beneficiary_account"].nunique(), "accounts,", len(mule_in_norm), "normal rows")
# Check top countries of the mule pool!
mule_rows = df[df["anomaly_type"].isin(["CEO_SCAM", "ACCOUNT_REPLACEMENT_SCAM", "ROMANCE_SCAM"])].drop_duplicates("beneficiary_account")
print("  Mule pool top 15 ben_bic_country:", mule_rows["ben_bic_country"].value_counts().head(15).to_dict())

# === 6. FX RATE RECOVERY FROM ROUND AMOUNTS & WEEKEND BY COUNTRY ===
print("\n=== 6. FX RATE RECOVERY & WEEKEND BY COUNTRY ===")
for cur_c in ["EUR", "GBP", "CHF", "CNY", "JPY", "SGD", "AUD", "CAD"]:
    sub = clean[(clean["currency_code"] == cur_c) & (clean["usd_amount"] >= 100)]
    top_vals = sub["usd_amount"].value_counts().head(5)
    print(f"  {cur_c} top exact USD_AMOUNTs:", top_vals.to_dict())

# Weekend volume ratio (Sat/Sun vs Mon) by country!
dt = pd.to_datetime(clean["epoch_seconds"], unit="s", utc=True)
clean["dow"] = dt.dt.dayofweek
top_cc = ["US", "CH", "GB", "DE", "SG", "CN", "AE", "SA", "IL", "EG", "JP", "AU"]
dow_cc = pd.crosstab(clean.loc[clean["ord_bic_country"].isin(top_cc), "ord_bic_country"], clean.loc[clean["ord_bic_country"].isin(top_cc), "dow"], normalize="index") * 100
print("\nDay-of-week % by country (0=Mon..4=Fri, 5=Sat, 6=Sun):")
print(dow_cc.round(2))

# === 7. DUAL-ROLE ACCOUNTS (BOTH ORDERING AND BENEFICIARY) ===
print("\n=== 7. DUAL-ROLE ACCOUNTS CONSISTENCY ===")
ord_set = set(df["ordering_account"].unique())
ben_set = set(df["beneficiary_account"].unique())
both_accts = list(ord_set & ben_set)[:5000]
sub_o = df[df["ordering_account"].isin(both_accts)].drop_duplicates("ordering_account")[["ordering_account", "ordering_bic8", "ordering_type", "ord_inferred_cc"]].rename(
    columns={"ordering_account": "acct", "ordering_bic8": "o_bic", "ordering_type": "o_type", "ord_inferred_cc": "o_cc"}
)
sub_b = df[df["beneficiary_account"].isin(both_accts)].drop_duplicates("beneficiary_account")[["beneficiary_account", "beneficiary_bic8", "beneficiary_type", "ben_inferred_cc"]].rename(
    columns={"beneficiary_account": "acct", "beneficiary_bic8": "b_bic", "beneficiary_type": "b_type", "ben_inferred_cc": "b_cc"}
)
m_both = sub_o.merge(sub_b, on="acct")
print("Dual-role accounts sample (n=", len(m_both), "):")
print("  o_type == b_type rate:", (m_both["o_type"] == m_both["b_type"]).mean())
print("  o_bic == b_bic rate:", (m_both["o_bic"] == m_both["b_bic"]).mean())
has_both_cc = (m_both["o_cc"] != "") & (m_both["b_cc"] != "")
print("  o_cc == b_cc rate (when both non-empty):", (m_both.loc[has_both_cc, "o_cc"] == m_both.loc[has_both_cc, "b_cc"]).mean())
print("  Sample 5 dual-role accounts:")
print(m_both.head(5).to_string())

print(f"\nElapsed: {time.time()-t0:.1f}s")
