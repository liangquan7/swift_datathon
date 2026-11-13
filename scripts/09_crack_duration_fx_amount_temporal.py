import time
import numpy as np
import pandas as pd

t0 = time.time()
cols = [
    "row_id", "epoch_seconds", "currency_code", "usd_amount",
    "ordering_bic8", "beneficiary_bic8", "bic8_path",
    "ord_bic_country", "ben_bic_country",
    "total_duration", "continuity_qflag", "fx_count", "participant_count", "tx_type",
    "ordering_type", "beneficiary_type", "anomaly_type",
]
df = pd.read_parquet("work/core_parquet", columns=cols)
clean = df[df["anomaly_type"] == ""].copy()
print(f"Loaded in {time.time()-t0:.1f}s")

# === 1. COMPLETE COUNTRY -> HOME CURRENCY MAP & MULTI-HOP FX RULE ===
print("\n=== 1. COMPLETE COUNTRY -> CURRENCY & MULTI-HOP FX RULE ===")
d2 = clean[clean["participant_count"] == 2].copy()
d2["pf_cc"] = d2["bic8_path"].str.slice(4, 6)
d2["pl_cc"] = d2["bic8_path"].str.slice(-4, -2)

# Find currency with fx_count == 0 when pf_cc == pl_cc, or when pf_cc appears with fx_count == 0!
# Better: for any country X in d2 where fx_count == 0, what currencies occur?
fx0 = d2[d2["fx_count"] == 0]
# Countries that appear as BOTH pf_cc and pl_cc in fx0:
fx0_same = fx0[fx0["pf_cc"] == fx0["pl_cc"]]
cc_cur = fx0_same.groupby("pf_cc")["currency_code"].agg(lambda s: s.value_counts().index[0]).to_dict()
# For any remaining country in clean not in cc_cur, check fx0 where pf_cc==c or pl_cc==c
all_cc = set(clean["ord_bic_country"].unique()) | set(clean["ben_bic_country"].unique())
missing_cc = sorted(all_cc - set(cc_cur.keys()))
print("Countries missing from d2 same-country fx==0:", len(missing_cc), missing_cc[:30])
for c in missing_cc:
    # Look at d2 fx==0 involving c
    sub = fx0[(fx0["pf_cc"] == c) | (fx0["pl_cc"] == c)]
    if len(sub) > 0:
        cc_cur[c] = sub["currency_code"].value_counts().index[0]
    else:
        # Fallback: most common currency when ord_bic_country==c
        sub2 = d2[d2["pf_cc"] == c]
        if len(sub2) > 0:
            cc_cur[c] = sub2["currency_code"].value_counts().index[0]

# Wait! Earlier AD, SM, VA, EE, MQ, MC had fx==0 with EUR, and BQ, GU, VI, TL, SV had USD! Why?
# Check countries where multiple currencies have fx_count == 0 in d2_same or overall!
print("Check AD, SM, BQ, EE, VA, GU, TL, SV, VI, KI in d2 fx==0:")
for c in ["AD", "SM", "BQ", "EE", "VA", "GU", "TL", "SV", "VI", "KI", "EC", "PA", "ZW", "ME", "MC"]:
    sub_same = fx0_same[fx0_same["pf_cc"] == c]["currency_code"].value_counts().head(3).to_dict()
    sub_any = fx0[(fx0["pf_cc"] == c) | (fx0["pl_cc"] == c)]["currency_code"].value_counts().head(3).to_dict()
    print(f"  {c}: cc_cur={cc_cur.get(c)} | fx0_same={sub_same} | fx0_any={sub_any}")

# Also build mapping directly from d2 fx0 (any edge where pf_cc or pl_cc has fx==0 AND currency_code is consistent)
# Check how many d2 rows still violate: fx==0 <=> pf_cur == pl_cur == cur!
# Wait: what if a country maps to EUR/USD? Let's define each country's currency by the most frequent currency among fx0 rows touching it!
cc_cur2 = {}
fx0_both = pd.concat([fx0[["pf_cc", "currency_code"]].rename(columns={"pf_cc": "cc"}),
                      fx0[["pl_cc", "currency_code"]].rename(columns={"pl_cc": "cc"})])
for c, grp in fx0_both.groupby("cc"):
    cc_cur2[c] = grp["currency_code"].value_counts().index[0]
for c in missing_cc:
    if c not in cc_cur2:
        sub2 = clean[clean["ord_bic_country"] == c]
        if len(sub2) > 0:
            cc_cur2[c] = sub2["currency_code"].value_counts().index[0]

d2["pf_cur2"] = d2["pf_cc"].map(cc_cur2)
d2["pl_cur2"] = d2["pl_cc"].map(cc_cur2)
d2_pred_fx0 = (d2["pf_cur2"] == d2["currency_code"]) & (d2["pl_cur2"] == d2["currency_code"])
print("Direct fx==0 prediction accuracy with cc_cur2:", (d2_pred_fx0 == (d2["fx_count"] == 0)).mean())
# Inspect remaining mismatches!
mis_d2 = d2[d2_pred_fx0 != (d2["fx_count"] == 0)]
print("Direct remaining mismatches:", len(mis_d2), "/", len(d2))
print("  False positives (pred fx=0, actual fx=1):", len(mis_d2[d2_pred_fx0[mis_d2.index]]), mis_d2[d2_pred_fx0[mis_d2.index]][["pf_cc", "pl_cc", "currency_code", "pf_cur2"]].head(10).to_dict("records"))
print("  False negatives (pred fx=1, actual fx=0):", len(mis_d2[~d2_pred_fx0[mis_d2.index]]), mis_d2[~d2_pred_fx0[mis_d2.index]][["pf_cc", "pl_cc", "currency_code", "pf_cur2", "pl_cur2"]].value_counts(["pf_cc", "pl_cc", "currency_code"]).head(15))

# Now test multi-hop (part == 3 and part == 4, continuity_qflag == 'Y')!
p3 = clean[(clean["participant_count"] == 3) & (clean["continuity_qflag"] == "Y")].head(500_000).copy()
p3_parts = p3["bic8_path"].str.split(";", expand=True)
p3["c0"] = p3_parts[0].str.slice(4, 6).map(cc_cur2)
p3["c1"] = p3_parts[1].str.slice(4, 6).map(cc_cur2)
p3["c2"] = p3_parts[2].str.slice(4, 6).map(cc_cur2)
cur = p3["currency_code"]
# Candidate rules for fx_count in 3-node path (c0 -> c1 -> c2):
# Rule A (edge currency mismatch): int(c0!=c1) + int(c1!=c2)  [ignores payment currency!]
# Rule B (edge against payment cur): int((c0!=cur) | (c1!=cur)) + int((c1!=cur) | (c2!=cur))
# Rule C (count of hops where receiver != sender or != cur):
# Rule D (number of distinct currencies among {c0, c1, c2, cur} - 1, capped at 2?!)
# Rule E (count nodes with c_i != cur, capped at 2?!)
# Rule F (count of nodes with c_i != c_{i-1} PLUS 1 if c0 != cur? capped at 2?)
p3["ruleA_adj_diff"] = (p3["c0"] != p3["c1"]).astype(int) + (p3["c1"] != p3["c2"]).astype(int)
p3["ruleB_edge_not_cur"] = ((p3["c0"] != cur) | (p3["c1"] != cur)).astype(int) + ((p3["c1"] != cur) | (p3["c2"] != cur)).astype(int)
p3["ruleC_nodes_not_cur_cap2"] = ((p3["c0"] != cur).astype(int) + (p3["c1"] != cur).astype(int) + (p3["c2"] != cur).astype(int)).clip(upper=2)
p3["ruleD_distinct_minus1_cap2"] = p3.apply(lambda r: min(2, len({r["c0"], r["c1"], r["c2"], r["currency_code"]}) - 1), axis=1)
p3["ruleE_distinct_nodes_minus1"] = p3.apply(lambda r: len({r["c0"], r["c1"], r["c2"]}) - 1, axis=1)

for rname in ["ruleA_adj_diff", "ruleB_edge_not_cur", "ruleC_nodes_not_cur_cap2", "ruleD_distinct_minus1_cap2", "ruleE_distinct_nodes_minus1"]:
    print(f"p3 exact match {rname}: {(p3[rname] == p3['fx_count']).mean():.5f}")

# Print cross-tab of (c0==cur, c1==cur, c2==cur, c0==c1, c1==c2, c0==c2) vs fx_count!
p3["pat"] = (
    (p3["c0"] == cur).astype(int).astype(str)
    + (p3["c1"] == cur).astype(int).astype(str)
    + (p3["c2"] == cur).astype(int).astype(str)
    + "_"
    + (p3["c0"] == p3["c1"]).astype(int).astype(str)
    + (p3["c1"] == p3["c2"]).astype(int).astype(str)
    + (p3["c0"] == p3["c2"]).astype(int).astype(str)
)
print("\np3 fx_count distribution by pattern (c0==cur,c1==cur,c2==cur _ c0==c1,c1==c2,c0==c2):")
print(pd.crosstab(p3["pat"], p3["fx_count"]))

# === 2. TOTAL_DURATION BANK-LEVEL & PATH-LEVEL MODEL ===
print("\n=== 2. TOTAL_DURATION BANK-LEVEL & PATH-LEVEL MODEL ===")
# Inspect log10(total_duration) within top 3 specific (ordering_bic8, beneficiary_bic8) pairs in Direct!
top_pairs = d2.groupby(["ordering_bic8", "beneficiary_bic8"]).size().sort_values(ascending=False).head(5)
print("Top 5 Direct bank pairs:")
print(top_pairs)
for (ob, bb), cnt in top_pairs.head(3).items():
    sub = d2[(d2["ordering_bic8"] == ob) & (d2["beneficiary_bic8"] == bb)]
    bins = np.arange(0, 4.5, 0.25)
    h, e = np.histogram(np.log10(sub["total_duration"]), bins=bins)
    print(f"\nPair {ob} -> {bb} (n={cnt}) log10_dur:")
    for b0, b1, c in zip(e[:-1], e[1:], h):
        if c > 0:
            print(f"  [{b0:.2f},{b1:.2f}): {c:4d} ({c/cnt*100:5.1f}%)")

# Compute per-BIC8 mean & median duration in Direct! Check R^2 of log10(dur) on ord_bic8 + ben_bic8!
d2["log_dur"] = np.log(d2["total_duration"])
# Check ordering_bic8 vs beneficiary_bic8 fast rate correlation (is a bank fast BOTH as sender and receiver?)
ord_fast = d2.groupby("ordering_bic8").agg(n_ord=("total_duration", "size"), ord_fast=("total_duration", lambda s: (s < 100).mean()), ord_med=("total_duration", "median"))
ben_fast = d2.groupby("beneficiary_bic8").agg(n_ben=("total_duration", "size"), ben_fast=("total_duration", lambda s: (s < 100).mean()), ben_med=("total_duration", "median"))
bic_perf = ord_fast.join(ben_fast, how="inner")
bic_perf_big = bic_perf[(bic_perf["n_ord"] >= 200) & (bic_perf["n_ben"] >= 200)]
print("\nCorrelation across banks between ord_fast and ben_fast:", bic_perf_big["ord_fast"].corr(bic_perf_big["ben_fast"]))
print("Correlation across banks between ord_med and ben_med:", bic_perf_big["ord_med"].corr(bic_perf_big["ben_med"]))
print("Sample 15 banks showing ord_fast vs ben_fast:")
print(bic_perf_big.sort_values("n_ord", ascending=False).head(15).round(3))

# Check duration for Covered vs Indirect controlling for participant_count AND ordering_bic8!
multi = clean[clean["participant_count"].isin([3, 4])].copy()
# Compare Covered vs Indirect within same ordering_bic8!
print("\nCovered vs Indirect duration controlling for participant_count & ordering_bic8 (top 5 high-Covered banks):")
hi_cov_bics = ["0004IMD0", "0412FRP0", "0085BEB0", "0309CHZ0", "0300CHZ0"]
print(multi[(multi["participant_count"] == 3) & (multi["ordering_bic8"].isin(hi_cov_bics))].groupby(["ordering_bic8", "tx_type"])["total_duration"].agg(["count", "mean", "median"]))

# Check if duration depends on HOUR of day or TIMEZONE difference or DAYLIGHT/BUSINESS HOURS!
dt_d2 = pd.to_datetime(d2["epoch_seconds"], unit="s", utc=True)
d2["utc_hour"] = dt_d2.dt.hour
# Check US->US vs US->SG vs SG->US duration by utc_hour!
for o_c, b_c in [("US", "US"), ("US", "SG"), ("DE", "FR")]:
    sub = d2[(d2["ord_bic_country"] == o_c) & (d2["ben_bic_country"] == b_c)]
    print(f"\nDuration median by utc_hour for {o_c}->{b_c} (every 3h):")
    print(sub.groupby(sub["utc_hour"] // 3 * 3)["total_duration"].agg(["count", "median", "mean"]).round(1))

# === 3. INTERMEDIARY ROUTING HUBS IN BIC8_PATH ===
print("\n=== 3. INTERMEDIARY BANKS & COUNTRIES IN BIC8_PATH ===")
p3_full = clean[(clean["participant_count"] == 3) & (clean["continuity_qflag"] == "Y")].copy()
p3_sp = p3_full["bic8_path"].str.split(";", expand=True)
p3_full["mid_bic"] = p3_sp[1]
p3_full["mid_cc"] = p3_full["mid_bic"].str.slice(4, 6)
print("Top 15 intermediary countries (part=3):")
print(p3_full["mid_cc"].value_counts(normalize=True).head(15).round(4))
print("Top 15 intermediary BIC8s (part=3):")
print(p3_full["mid_bic"].value_counts(normalize=True).head(15).round(4))
print("\nTop 5 intermediary countries when CURRENCY_CODE is USD / EUR / GBP / CHF / CNY:")
for cur_c in ["USD", "EUR", "GBP", "CHF", "CNY", "JPY"]:
    sub = p3_full[p3_full["currency_code"] == cur_c]
    print(f"  {cur_c} (n={len(sub):,}):", sub["mid_cc"].value_counts(normalize=True).head(6).round(4).to_dict())

# === 4. USD_AMOUNT REGRESSION & COUNTRY MULTIPLIER ===
print("\n=== 4. USD_AMOUNT MODEL & ROUND NUMBERS ===")
clean["party_pair"] = clean["ordering_type"] + "2" + clean["beneficiary_type"]
clean["log_amt"] = np.log(clean["usd_amount"])
# Fit additive effects: party_pair + ord_bic_country + ben_bic_country on sample of 1M rows
s1m = clean.sample(1_000_000, random_state=42).copy()
# Demean by party_pair
pp_means = s1m.groupby("party_pair")["log_amt"].mean()
s1m["res1"] = s1m["log_amt"] - s1m["party_pair"].map(pp_means)
r2_party = 1 - s1m["res1"].var() / s1m["log_amt"].var()
ord_means = s1m.groupby("ord_bic_country")["res1"].mean()
s1m["res2"] = s1m["res1"] - s1m["ord_bic_country"].map(ord_means)
r2_ord = 1 - s1m["res2"].var() / s1m["log_amt"].var()
ben_means = s1m.groupby("ben_bic_country")["res2"].mean()
s1m["res3"] = s1m["res2"] - s1m["ben_bic_country"].map(ben_means)
r2_ben = 1 - s1m["res3"].var() / s1m["log_amt"].var()
cur_means = s1m.groupby("currency_code")["res3"].mean()
s1m["res4"] = s1m["res3"] - s1m["currency_code"].map(cur_means)
r2_cur = 1 - s1m["res4"].var() / s1m["log_amt"].var()
print(f"Cumulative R2 on ln(USD_AMOUNT): party_pair={r2_party:.4f}, +ord_cc={r2_ord:.4f}, +ben_cc={r2_ben:.4f}, +cur={r2_cur:.4f}")
print("std(log_amt) =", s1m["log_amt"].std(), "-> std(res1) =", s1m["res1"].std(), "-> std(res2) =", s1m["res2"].std(), "-> std(res3) =", s1m["res3"].std())
print("Top 10 & Bottom 10 ord_bic_country effects (exp(delta) multiplier):")
ord_eff = np.exp(ord_means).sort_values(ascending=False)
ord_cnts = s1m["ord_bic_country"].value_counts()
ord_eff_big = ord_eff[ord_eff.index.isin(ord_cnts[ord_cnts >= 500].index)]
print("  Top 10:", ord_eff_big.head(10).round(3).to_dict())
print("  Bottom 10:", ord_eff_big.tail(10).round(3).to_dict())

# Check round amounts: are round amounts round in USD_AMOUNT or in local currency?
# Check USD_AMOUNT == 1000.0 or 10000.0 rate across currencies!
clean["is_exact_int"] = (clean["usd_amount"] % 1) == 0
clean["is_mult100"] = (clean["usd_amount"] % 100) == 0
print("\nExact integer / mult100 rate in USD_AMOUNT by top 8 currencies:")
top8c = clean["currency_code"].value_counts().head(8).index
print(clean[clean["currency_code"].isin(top8c)].groupby("currency_code")[["is_exact_int", "is_mult100"]].mean().round(4))
print("Exact integer / mult100 rate by party_pair:")
print(clean.groupby("party_pair")[["is_exact_int", "is_mult100"]].mean().round(4))

# === 5. FULL 92-DAY DAILY VOLUME SERIES ===
print("\n=== 5. DAILY VOLUME SERIES (ALL 92 DAYS) ===")
dt = pd.to_datetime(clean["epoch_seconds"], unit="s", utc=True)
clean["date"] = dt.dt.date
clean["dow"] = dt.dt.dayofweek
daily = clean.groupby("date").agg(n=("usd_amount", "size"), dow=("dow", "first"), med_amt=("usd_amount", "median"), b2b=("party_pair", lambda s: (s == "B2B").mean()))
for d, r in daily.iterrows():
    print(f"  {d} (dow={int(r['dow'])}): n={int(r['n']):7,d}, med_amt={r['med_amt']:8.2f}, b2b={r['b2b']:.3f}")

print(f"\nElapsed: {time.time()-t0:.1f}s")
