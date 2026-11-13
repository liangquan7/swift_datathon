import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

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
    "ord_has_biz", "ord_has_person", "ben_has_biz", "ben_has_person",
    "ord_has_acct_line", "ben_has_acct_line",
    "ord_has_double_slash", "ben_has_double_slash",
    "ord_det_len", "ben_det_len", "ord_n_lines", "ben_n_lines",
]
df = pd.read_parquet("work/core_parquet", columns=cols)
clean = df[df["anomaly_type"] == ""].copy()
print(f"Loaded in {time.time()-t0:.1f}s")

# === 1. TOTAL_DURATION RESIDUAL DIAGNOSTIC & MULTI-HOP FORMULA ===
print("\n=== 1. TOTAL_DURATION EXACT DIAGNOSTIC ===")
d2 = clean[(clean["participant_count"] == 2) & (clean["bic8_path"].str.slice(0, 8) == clean["ordering_bic8"]) & (clean["bic8_path"].str.slice(-8) == clean["beneficiary_bic8"])].copy()
# Only keep banks with >= 50 obs on both sides for stable estimation
ob_cnt = d2["ordering_bic8"].value_counts()
bb_cnt = d2["beneficiary_bic8"].value_counts()
d2_freq = d2[d2["ordering_bic8"].isin(ob_cnt[ob_cnt >= 50].index) & d2["beneficiary_bic8"].isin(bb_cnt[bb_cnt >= 50].index)].copy()
y = d2_freq["total_duration"].to_numpy(dtype=np.float64)
ob_c, ob_u = pd.factorize(d2_freq["ordering_bic8"])
bb_c, bb_u = pd.factorize(d2_freq["beneficiary_bic8"])
a = np.zeros(len(ob_u))
b = np.zeros(len(bb_u))
mu = y.mean()
ca = np.bincount(ob_c)
cb = np.bincount(bb_c)
for _ in range(25):
    a = np.bincount(ob_c, weights=y - mu - b[bb_c]) / np.maximum(ca, 1)
    b = np.bincount(bb_c, weights=y - mu - a[ob_c]) / np.maximum(cb, 1)
pred = mu + a[ob_c] + b[bb_c]
res = y - pred
print(f"d2_freq (n={len(d2_freq):,}): R2={1 - np.var(res)/np.var(y):.6f}, RMSE={np.sqrt(np.mean(res**2)):.2f}, MAE={np.mean(np.abs(res)):.2f}")
print("Residual quantiles:", np.quantile(res, [0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]).round(2))
# Inspect largest |res| > 100!
d2_freq["pred"] = pred
d2_freq["res"] = res
big_res = d2_freq[np.abs(res) > 100]
print("Rows with |res| > 100s:", len(big_res), f"({len(big_res)/len(d2_freq)*100:.3f}%)")
if len(big_res) > 0:
    print("Top ordering_bic8 in big_res:", big_res["ordering_bic8"].value_counts().head(10).to_dict())
    print("Top beneficiary_bic8 in big_res:", big_res["beneficiary_bic8"].value_counts().head(10).to_dict())
    # Check one pair inside big_res!
    p0 = big_res.iloc[0]
    sub_p = d2_freq[(d2_freq["ordering_bic8"] == p0["ordering_bic8"]) & (d2_freq["beneficiary_bic8"] == p0["beneficiary_bic8"])]
    print(f"Sample pair {p0['ordering_bic8']}->{p0['beneficiary_bic8']} (n={len(sub_p)}) durations quantiles:", sub_p["total_duration"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).to_dict())

# Also check distribution of bank latency parameters a (shifted so min ~ 0)!
# Notice a and b can be shifted by constant c so a_shift = a - a.min(), b_shift = b - b.min()
a_s = pd.Series(mu + a, index=ob_u)
b_s = pd.Series(b, index=bb_u)
# Center using a median fast bank: find bank where a+b+mu is ~20
fast_min = (a_s.min() + b_s.min())
print("Distribution of estimated bank delay L_ord (shifted by min):", (a_s - a_s.min()).quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]).round(1).to_dict())
print("Distribution of estimated bank delay L_ben (shifted by min):", (b_s - b_s.min()).quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]).round(1).to_dict())

# Check multi-hop (part==3) Indirect vs Covered using bank delays!
# Map bank delay L[bic] = (a_s - a_s.min())
bank_delay = (a_s - a_s.min()).to_dict()
p3 = clean[(clean["participant_count"] == 3) & (clean["continuity_qflag"] == "Y")].head(300_000).copy()
sp3 = p3["bic8_path"].str.split(";", expand=True)
p3["d0"] = sp3[0].map(bank_delay)
p3["d1"] = sp3[1].map(bank_delay)
p3["d2"] = sp3[2].map(bank_delay)
p3v = p3.dropna(subset=["d0", "d1", "d2"]).copy()
p3v["sum3"] = p3v["d0"] + p3v["d1"] + p3v["d2"]
p3v["sum_end2"] = p3v["d0"] + p3v["d2"]
for tt in ["Indirect", "Covered"]:
    sub = p3v[p3v["tx_type"] == tt]
    print(f"part=3 {tt} (n={len(sub):,}): corr(dur, sum3)={sub['total_duration'].corr(sub['sum3']):.5f}, corr(dur, sum_end2)={sub['total_duration'].corr(sub['sum_end2']):.5f}")
    print(f"  median(dur - sum3)={ (sub['total_duration'] - sub['sum3']).median():.1f}, median(dur - sum_end2)={(sub['total_duration'] - sub['sum_end2']).median():.1f}, median(dur / sum3)={(sub['total_duration'] / np.maximum(sub['sum3'], 1)).median():.3f}")
    # Regress dur on d0, d1, d2!
    X = np.column_stack([np.ones(len(sub)), sub["d0"], sub["d1"], sub["d2"]])
    coef, _, _, _ = np.linalg.lstsq(X, sub["total_duration"].to_numpy(), rcond=None)
    pred3 = X @ coef
    r2_3 = 1 - np.var(sub["total_duration"] - pred3) / np.var(sub["total_duration"])
    print(f"  OLS dur ~ const + c0*d0 + c1*d1 + c2*d2: coef={coef.round(3)}, R2={r2_3:.5f}")

# === 2. FX_COUNT ON ALL 6.32M DIRECT & 3.68M MULTI-HOP ===
print("\n=== 2. FX_COUNT ALL-DATA CHECK ===")
fx0_d2 = d2[d2["fx_count"] == 0]
fx0_both = pd.concat([
    fx0_d2[["ord_bic_country", "currency_code"]].rename(columns={"ord_bic_country": "cc"}),
    fx0_d2[["ben_bic_country", "currency_code"]].rename(columns={"ben_bic_country": "cc"}),
])
cc_cur = {c: g["currency_code"].value_counts().index[0] for c, g in fx0_both.groupby("cc")}
for c in set(clean["ord_bic_country"].unique()) | set(clean["ben_bic_country"].unique()):
    if c not in cc_cur:
        sub = clean[clean["ord_bic_country"] == c]
        if len(sub) > 0:
            cc_cur[c] = sub["currency_code"].value_counts().index[0]

d2_all = clean[clean["participant_count"] == 2].copy()
d2_all["pf_cur"] = d2_all["bic8_path"].str.slice(4, 6).map(cc_cur)
d2_all["pl_cur"] = d2_all["bic8_path"].str.slice(-4, -2).map(cc_cur)
d2_pred = (d2_all["pf_cur"] != d2_all["pl_cur"]).astype(int)
print("Direct ALL (n=6.32M) exact match (pf_cur != pl_cur) == fx_count:", (d2_pred == d2_all["fx_count"]).mean(), "mismatches:", (d2_pred != d2_all["fx_count"]).sum())
mis = d2_all[d2_pred != d2_all["fx_count"]]
if len(mis) > 0:
    print("  Top mismatch pairs:", mis.groupby([mis["bic8_path"].str.slice(4, 6), mis["bic8_path"].str.slice(-4, -2), "fx_count"]).size().sort_values(ascending=False).head(15))

# === 3. FX RATE TIMESTAMPS FOR ROUND AMOUNTS ===
print("\n=== 3. FX RATE TIMESTAMPS FOR EUR 1000 SPIKES ===")
dt = pd.to_datetime(clean["epoch_seconds"], unit="s", utc=True)
clean["date"] = dt.dt.date
for val in [1149.08, 1150.90, 1136.73, 1148.13]:
    sub = clean[(clean["currency_code"] == "EUR") & (clean["usd_amount"] == val)]
    print(f"  EUR == {val}: n={len(sub)}, n_dates={sub['date'].nunique()}, min_date={sub['date'].min()}, max_date={sub['date'].max()}, top_dates={sub['date'].value_counts().head(5).to_dict()}")

# === 4. LOCAL TIME ALIGNMENT & MON-SUN IN LOCAL TIME ===
print("\n=== 4. LOCAL TIME ALIGNMENT BY COUNTRY ===")
# Find optimal integer UTC offset [-11..+12] matching US (offset -4) profile!
hr_by_cc = pd.crosstab(clean["ord_bic_country"], dt.dt.hour, normalize="index")
us_prof = hr_by_cc.loc["US"].to_numpy() # US EDT = UTC-4
# local_hour = (utc_hour + offset) % 24 => local[0] = utc[-offset]; roll(p, +offset) puts p[i] at p[(i+offset)%24] -> WAIT:
# roll([a0,a1,a2,a3,a4,...], -4) puts a4 at index 0! So local_ref[0] = us_prof[4] (04:00 UTC = 00:00 EDT)!
local_ref = np.roll(us_prof, -4)
print("US local-hour profile (00..23 EDT):", (local_ref * 100).round(2).tolist())
for cc in ["US", "CA", "GB", "CH", "DE", "FR", "AE", "SA", "IL", "EG", "IN", "SG", "CN", "JP", "AU", "NZ"]:
    if cc in hr_by_cc.index:
        p = hr_by_cc.loc[cc].to_numpy()
        best_off = max(range(-11, 13), key=lambda k: np.corrcoef(np.roll(p, k), local_ref)[0, 1])
        best_corr = np.corrcoef(np.roll(p, best_off), local_ref)[0, 1]
        # With np.roll(p, k), new[i+k] = p[i], so local_hour = (utc_hour + k) % 24!
        sub_mask = clean["ord_bic_country"] == cc
        loc_dt = pd.to_datetime(clean.loc[sub_mask, "epoch_seconds"] + best_off * 3600, unit="s", utc=True)
        loc_dow = loc_dt.dt.dayofweek.value_counts(normalize=True).sort_index() * 100
        print(f"  {cc}: best_utc_offset={best_off:+3d} (r={best_corr:.4f}) | local Mon-Fri={loc_dow.loc[0:4].round(1).to_dict()} Sat={loc_dow.get(5,0):.2f}% Sun={loc_dow.get(6,0):.2f}%")

# === 5. FAT_FINGER_MISTAKE ON USD & CUSTOMER_RECALL AUC ===
print("\n=== 5. FAT_FINGER USD DIGITS & CUSTOMER_RECALL AUC ===")
ff_usd = df[(df["anomaly_type"] == "FAT_FINGER_MISTAKE") & (df["currency_code"] == "USD")].copy()
bg_usd = clean[clean["currency_code"] == "USD"].sample(50_000, random_state=1).copy()
for name, s in [("FF_USD", ff_usd["usd_amount"]), ("CLEAN_USD", bg_usd["usd_amount"])]:
    is_int = (s % 1 == 0).mean()
    m10 = (s % 10 == 0).mean()
    m100 = (s % 100 == 0).mean()
    m1000 = (s % 1000 == 0).mean()
    m10000 = (s % 10000 == 0).mean()
    has1dec = (((s * 10) % 1 == 0) & (s % 1 != 0)).mean()
    print(f"  {name} (n={len(s):,}): int={is_int:.3f}, 1dec={has1dec:.3f}, %10={m10:.3f}, %100={m100:.3f}, %1000={m1000:.3f}, %10000={m10000:.3f}")

# LightGBM AUC check for CUSTOMER_RECALL vs NORMAL (5-fold CV)
cr = df[df["anomaly_type"] == "CUSTOMER_RECALL"].copy()
bg_cr = clean.sample(30_000, random_state=42).copy()
cr["y"] = 1
bg_cr["y"] = 0
cr_all = pd.concat([cr, bg_cr], ignore_index=True)
dt_cr = pd.to_datetime(cr_all["epoch_seconds"], unit="s", utc=True)
cr_all["utc_hour"] = dt_cr.dt.hour
cr_all["dow"] = dt_cr.dt.dayofweek
# Account degree features!
ord_deg = df["ordering_account"].value_counts()
ben_deg = df["beneficiary_account"].value_counts()
cr_all["ord_deg"] = cr_all["ordering_account"].map(ord_deg)
cr_all["ben_deg"] = cr_all["beneficiary_account"].map(ben_deg)
cr_all["same_cc"] = (cr_all["ord_bic_country"] == cr_all["ben_bic_country"]).astype(int)
cr_all["cc_match_ben"] = (cr_all["ben_inferred_cc"] == cr_all["ben_bic_country"]).astype(int)
feat_cols = ["usd_amount", "total_duration", "fx_count", "participant_count", "utc_hour", "dow", "ord_deg", "ben_deg",
             "same_cc", "cc_match_ben", "ord_det_len", "ben_det_len", "ord_n_lines", "ben_n_lines"]
for cat in ["tx_type", "continuity_qflag", "ordering_type", "beneficiary_type", "ord_has_iban", "ben_has_iban", "ord_is_struct_123", "ben_is_struct_123"]:
    cr_all[cat] = cr_all[cat].astype("category")
    feat_cols.append(cat)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(cr_all))
for tr, va in skf.split(cr_all, cr_all["y"]):
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=15, random_state=42, verbose=-1)
    clf.fit(cr_all.iloc[tr][feat_cols], cr_all.iloc[tr]["y"])
    oof[va] = clf.predict_proba(cr_all.iloc[va][feat_cols])[:, 1]
print(f"CUSTOMER_RECALL 5-fold CV LightGBM ROC-AUC: {roc_auc_score(cr_all['y'], oof):.4f}")

# === 6. DETAIL TEMPLATE & ACCOUNT_REPLACEMENT_SCAM vs CEO_SCAM ===
print("\n=== 6. DETAILS TEMPLATES & ARS vs CEO ===")
clean["ord_no_name"] = (~clean["ord_has_biz"]) & (~clean["ord_has_person"])
clean["ben_no_name"] = (~clean["ben_has_biz"]) & (~clean["ben_has_person"])
print("Clean ord_no_name rate:", clean["ord_no_name"].mean(), "ben_no_name rate:", clean["ben_no_name"].mean())
print("When ord_no_name==True, ord_n_lines value counts:", clean.loc[clean["ord_no_name"], "ord_n_lines"].value_counts().head(5).to_dict())
print("When ord_no_name==True, ord_has_acct_line rate:", clean.loc[clean["ord_no_name"], "ord_has_acct_line"].mean())

# Check CEO vs ARS on ALL distinguishing fields!
for at in ["CEO_SCAM", "ACCOUNT_REPLACEMENT_SCAM"]:
    sub = df[df["anomaly_type"] == at]
    print(f"  {at}: n={len(sub)}, pct_B_ord={(sub['ordering_type']=='B').mean():.3f}, pct_B_ben={(sub['beneficiary_type']=='B').mean():.3f}, "
          f"ben_has_iban={sub['ben_has_iban'].mean():.3f}, ben_no_name={(~sub['ben_has_biz'] & ~sub['ben_has_person']).mean():.3f}, "
          f"covered={(sub['tx_type']=='Covered').mean():.4f}")

print(f"\nElapsed: {time.time()-t0:.1f}s")
