import time
import numpy as np
import pandas as pd

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
    "ord_has_double_slash", "ben_has_double_slash",
    "ord_has_biz", "ord_has_person", "ben_has_biz", "ben_has_person",
]
df = pd.read_parquet("work/core_parquet", columns=cols)
clean = df[df["anomaly_type"] == ""].copy()
print(f"Loaded in {time.time()-t0:.1f}s")

# ==================== 1. FX_COUNT EXACT RULE ====================
print("\n=== 1. FX_COUNT EXACT CHECK ===")
print("Any row with fx_count > participant_count - 1?", (df["fx_count"] > df["participant_count"] - 1).sum())
print("Any row with fx_count < 0?", (df["fx_count"] < 0).sum())

# Use path_first and path_last countries!
d2 = clean[clean["participant_count"] == 2].copy()
d2["pf_bic"] = d2["bic8_path"].str.slice(0, 8)
d2["pl_bic"] = d2["bic8_path"].str.slice(-8)
d2["pf_cc"] = d2["pf_bic"].str.slice(4, 6)
d2["pl_cc"] = d2["pl_bic"].str.slice(4, 6)

# Infer home currency per country from pf_cc==pl_cc & fx_count==0
d2_same = d2[(d2["pf_cc"] == d2["pl_cc"]) & (d2["fx_count"] == 0)]
cc_cur = d2_same.groupby("pf_cc")["currency_code"].agg(lambda s: s.value_counts().index[0])
d2["pf_cur"] = d2["pf_cc"].map(cc_cur).fillna("?")
d2["pl_cur"] = d2["pl_cc"].map(cc_cur).fillna("?")

d2["case_path"] = (
    np.where(d2["currency_code"] == d2["pf_cur"], "O", "-")
    + np.where(d2["currency_code"] == d2["pl_cur"], "B", "-")
    + np.where(d2["pf_cur"] == d2["pl_cur"], "_SAME", "_DIFF")
)
print("\nDirect (part=2) fx_count using BIC8_PATH endpoints:")
print(pd.crosstab(d2["case_path"], d2["fx_count"], normalize="index").round(5))
print("Counts:", d2["case_path"].value_counts().to_dict())

# Inspect -B_DIFF where fx_count == 0! Which countries / currencies / BICs?
sub_b0 = d2[(d2["case_path"] == "-B_DIFF") & (d2["fx_count"] == 0)]
print("\nSample -B_DIFF with fx_count == 0:")
print("  Top pf_cc:", sub_b0["pf_cc"].value_counts().head(10).to_dict())
print("  Top pl_cc:", sub_b0["pl_cc"].value_counts().head(10).to_dict())
print("  Top currency:", sub_b0["currency_code"].value_counts().head(10).to_dict())
print("  Wait: in -B_DIFF, rate of fx==0 by pf_cc (top 15):")
bdiff = d2[d2["case_path"] == "-B_DIFF"]
print(bdiff.groupby("pf_cc").agg(n=("fx_count", "size"), fx0_rate=("fx_count", lambda s: (s == 0).mean())).sort_values("n", ascending=False).head(15))
print("  And in -B_DIFF, rate of fx==0 by currency_code (top 15):")
print(bdiff.groupby("currency_code").agg(n=("fx_count", "size"), fx0_rate=("fx_count", lambda s: (s == 0).mean())).sort_values("n", ascending=False).head(15))

# Also check O-_DIFF where fx_count == 0!
print("\nIn O-_DIFF, rate of fx==0 by pl_cc (top 15):")
odiff = d2[d2["case_path"] == "O-_DIFF"]
print(odiff.groupby("pl_cc").agg(n=("fx_count", "size"), fx0_rate=("fx_count", lambda s: (s == 0).mean())).sort_values("n", ascending=False).head(15))

# ==================== 2. TOTAL_DURATION CAUSAL DRIVERS ====================
print("\n=== 2. TOTAL_DURATION CAUSAL DRIVERS ===")
# Check Direct (part=2) duration quantiles/modes by ordering_bic8, beneficiary_bic8, countries, tx_type!
d2["is_fast"] = d2["total_duration"] < 100
d2["is_med"] = (d2["total_duration"] >= 100) & (d2["total_duration"] < 900)
d2["is_slow"] = d2["total_duration"] >= 900

print("Direct fast(<100s) rate overall:", d2["is_fast"].mean())
print("\nDirect fast/med/slow & median dur by top 12 ord_bic_country:")
print(d2[d2["ord_bic_country"].isin(d2["ord_bic_country"].value_counts().head(12).index)].groupby("ord_bic_country").agg(
    n=("total_duration", "size"),
    fast=("is_fast", "mean"),
    med=("is_med", "mean"),
    slow=("is_slow", "mean"),
    median_dur=("total_duration", "median"),
))

print("\nDirect fast/med/slow & median dur by top 12 ben_bic_country:")
print(d2[d2["ben_bic_country"].isin(d2["ben_bic_country"].value_counts().head(12).index)].groupby("ben_bic_country").agg(
    n=("total_duration", "size"),
    fast=("is_fast", "mean"),
    med=("is_med", "mean"),
    slow=("is_slow", "mean"),
    median_dur=("total_duration", "median"),
))

print("\nDirect fast/med/slow by top 15 ordering_bic8:")
top_ob = d2["ordering_bic8"].value_counts().head(15).index
print(d2[d2["ordering_bic8"].isin(top_ob)].groupby("ordering_bic8").agg(
    n=("total_duration", "size"),
    fast=("is_fast", "mean"),
    med=("is_med", "mean"),
    slow=("is_slow", "mean"),
    median_dur=("total_duration", "median"),
))

print("\nDirect fast/med/slow by top 15 beneficiary_bic8:")
top_bb = d2["beneficiary_bic8"].value_counts().head(15).index
print(d2[d2["beneficiary_bic8"].isin(top_bb)].groupby("beneficiary_bic8").agg(
    n=("total_duration", "size"),
    fast=("is_fast", "mean"),
    med=("is_med", "mean"),
    slow=("is_slow", "mean"),
    median_dur=("total_duration", "median"),
))

# Check if ALL banks in Direct have ~same fast rate OR huge variation across banks!
ob_stats = d2.groupby("ordering_bic8")["is_fast"].agg(["size", "mean"])
ob_stats = ob_stats[ob_stats["size"] >= 500]
print("\nAcross ordering_bic8 (n>=500), fast(<100s) rate quantiles:", ob_stats["mean"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).round(4).to_dict())
bb_stats = d2.groupby("beneficiary_bic8")["is_fast"].agg(["size", "mean"])
bb_stats = bb_stats[bb_stats["size"] >= 500]
print("Across beneficiary_bic8 (n>=500), fast(<100s) rate quantiles:", bb_stats["mean"].quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).round(4).to_dict())

# Also check if fast(<100s) rate varies by party_pair or currency_code!
d2["party_pair"] = d2["ordering_type"] + "2" + d2["beneficiary_type"]
print("\nDirect duration stats by party_pair:")
print(d2.groupby("party_pair")[["is_fast", "is_med", "is_slow"]].mean())
print("\nDirect duration stats by top 10 currencies:")
top_cur = d2["currency_code"].value_counts().head(10).index
print(d2[d2["currency_code"].isin(top_cur)].groupby("currency_code")[["is_fast", "is_med", "is_slow"]].mean())

# ==================== 3. USD_AMOUNT CAUSAL SEPARATION: COUNTRY vs CURRENCY & TIME ====================
print("\n=== 3. USD_AMOUNT: COUNTRY vs CURRENCY & TIME ===")
clean["party_pair"] = clean["ordering_type"] + "2" + clean["beneficiary_type"]
clean["log_amt"] = np.log(clean["usd_amount"])

# Check B2B only: vary currency_code holding ord_bic_country=='US' fixed vs vary ord_bic_country holding currency_code=='USD' fixed!
b2b = clean[clean["party_pair"] == "B2B"]
print("B2B, ord_bic_country=='US', median USD_AMOUNT by top 6 currencies:")
print(b2b[(b2b["ord_bic_country"] == "US") & (b2b["currency_code"].isin(["USD", "EUR", "GBP", "CHF", "CNY", "JPY"]))].groupby("currency_code")["usd_amount"].agg(["count", "median"]))

print("\nB2B, currency_code=='USD', median USD_AMOUNT by top 8 ord_bic_country:")
print(b2b[(b2b["currency_code"] == "USD") & (b2b["ord_bic_country"].isin(["US", "CH", "GB", "DE", "FR", "SG", "CN", "IT"]))].groupby("ord_bic_country")["usd_amount"].agg(["count", "median"]))

print("\nB2B, currency_code=='EUR', median USD_AMOUNT by top 8 ord_bic_country:")
print(b2b[(b2b["currency_code"] == "EUR") & (b2b["ord_bic_country"].isin(["US", "CH", "GB", "DE", "FR", "SG", "CN", "IT"]))].groupby("ord_bic_country")["usd_amount"].agg(["count", "median"]))

print("\nB2B, ord_bic_country=='CH', median USD_AMOUNT by top 6 ben_bic_country (when cur=='CHF'):")
print(b2b[(b2b["ord_bic_country"] == "CH") & (b2b["currency_code"] == "CHF") & (b2b["ben_bic_country"].isin(["US", "CH", "GB", "DE", "CN", "SG"]))].groupby("ben_bic_country")["usd_amount"].agg(["count", "median"]))

# Check daily volume & daily median amount across the 92 days!
dt = pd.to_datetime(clean["epoch_seconds"], unit="s", utc=True)
clean["date"] = dt.dt.date
daily = clean.groupby("date").agg(n=("usd_amount", "size"), dow=("epoch_seconds", lambda s: pd.to_datetime(s.iloc[0], unit="s", utc=True).dayofweek))
print("\nDaily counts summary (min/max/quantiles):", daily["n"].describe().to_dict())
print("Top 10 busiest days:")
print(daily.sort_values("n", ascending=False).head(10))
print("Bottom 10 quietest days:")
print(daily.sort_values("n", ascending=True).head(10))

# ==================== 4. PARTY DETAILS & TOKENS MYSTERY ====================
print("\n=== 4. PARTY DETAILS & TOKENS MYSTERY ===")
print("ben_inferred_cc top 15 when non-empty:")
print(clean.loc[clean["ben_inferred_cc"] != "", "ben_inferred_cc"].value_counts().head(15).to_dict())
print("ord_inferred_cc top 15 when non-empty:")
print(clean.loc[clean["ord_inferred_cc"] != "", "ord_inferred_cc"].value_counts().head(15).to_dict())

# Check if ben_inferred_cc matches ord_inferred_cc? Or is ben_inferred_cc uniform across countries?
has_both = (clean["ord_inferred_cc"] != "") & (clean["ben_inferred_cc"] != "")
print("ben_inferred_cc == ord_inferred_cc rate:", (clean.loc[has_both, "ben_inferred_cc"] == clean.loc[has_both, "ord_inferred_cc"]).mean())
print("ben_inferred_cc == ben_bic_country rate:", (clean.loc[clean["ben_inferred_cc"] != "", "ben_inferred_cc"] == clean.loc[clean["ben_inferred_cc"] != "", "ben_bic_country"]).mean())
print("ben_inferred_cc == ben_bic_country rate BY anomaly_type:")
for at, g in df[df["ben_inferred_cc"] != ""].groupby("anomaly_type"):
    print(f"  {at or 'NORMAL'}: n={len(g)}, match={(g['ben_inferred_cc'] == g['ben_bic_country']).mean():.4f}")

# Check IBAN rate by country! Is IBAN 100% for IBAN countries (EU/DE/FR/GB/CH...) and 0% for US/CN/JP/SG?!
print("\nord_has_iban rate by top 15 ord_bic_country:")
top15 = clean["ord_bic_country"].value_counts().head(15).index
print(clean[clean["ord_bic_country"].isin(top15)].groupby("ord_bic_country")[["ord_has_iban", "ord_is_struct_123", "ord_has_lb35", "ord_has_double_slash"]].mean().round(4))

print("\nben_has_iban rate by top 15 ben_bic_country:")
print(clean[clean["ben_bic_country"].isin(top15)].groupby("ben_bic_country")[["ben_has_iban", "ben_is_struct_123", "ben_has_lb35", "ben_has_double_slash"]].mean().round(4))

print("\nord_has_biz / ord_has_person by ordering_type:")
print(clean.groupby("ordering_type")[["ord_has_biz", "ord_has_person", "ord_is_struct_123", "ord_has_iban", "ord_has_double_slash"]].mean().round(4))
print("ben_has_biz / ben_has_person by beneficiary_type:")
print(clean.groupby("beneficiary_type")[["ben_has_biz", "ben_has_person", "ben_is_struct_123", "ben_has_iban", "ben_has_double_slash"]].mean().round(4))

print("ord_fin_word rate by ordering_type / party_pair:")
clean["has_ord_fin"] = clean["ord_fin_word"] != ""
clean["has_ben_fin"] = clean["ben_fin_word"] != ""
print(clean.groupby("party_pair")[["has_ord_fin", "has_ben_fin"]].mean().round(4))
print("ord_fin_word rate by ord_is_struct_123:")
print(clean.groupby("ord_is_struct_123")["has_ord_fin"].mean())

print(f"\nElapsed: {time.time()-t0:.1f}s")
