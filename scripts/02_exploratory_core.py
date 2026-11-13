import json
import os
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

t0 = time.time()
cols = [
    "row_id", "epoch_seconds", "currency_code", "usd_amount",
    "ordering_bic8", "beneficiary_bic8", "bic8_path",
    "ord_bic_bank", "ord_bic_country", "ord_bic_loc",
    "ben_bic_bank", "ben_bic_country", "ben_bic_loc",
    "path_has_gap", "path_visible_bics",
    "total_duration", "continuity_qflag", "fx_count", "participant_count", "tx_type",
    "ordering_account", "beneficiary_account",
    "ordering_type", "beneficiary_type", "anomaly_type",
    "ord_inferred_cc", "ben_inferred_cc",
    "ord_fin_word", "ben_fin_word",
    "ord_has_iban", "ben_has_iban",
    "ord_has_lb35", "ben_has_lb35",
    "ord_has_trunc", "ben_has_trunc",
    "ord_has_biz", "ben_has_biz",
    "ord_has_person", "ben_has_person",
    "ord_is_struct_123", "ben_is_struct_123",
    "ord_has_acct_line", "ben_has_acct_line",
    "ord_has_double_slash", "ben_has_double_slash",
]
df = pd.read_parquet("work/core_parquet", columns=cols)
print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s, mem={df.memory_usage(deep=False).sum()/1e9:.2f} GB")

# 1. Party type pair
print("\n=== 1. ORDERING_TYPE x BENEFICIARY_TYPE ===")
ct_party = pd.crosstab(df["ordering_type"], df["beneficiary_type"], margins=True)
print(ct_party)
print((ct_party / len(df) * 100).round(3))

# 2. Routing & Continuity
print("\n=== 2. TX_TYPE x CONTINUITY_QFLAG & PATH_HAS_GAP ===")
print(pd.crosstab(df["tx_type"], df["continuity_qflag"], margins=True))
print("continuity_qflag=='N' == path_has_gap exact match?", ( (df["continuity_qflag"] == "N") == df["path_has_gap"] ).all())
mism = df[(df["continuity_qflag"] == "N") != df["path_has_gap"]]
print("Mismatch count:", len(mism))
if len(mism) > 0:
    print(mism[["bic8_path", "continuity_qflag", "path_has_gap", "tx_type", "participant_count"]].head(5))

print("\n=== TX_TYPE x PARTICIPANT_COUNT (2..12) ===")
print(pd.crosstab(df["participant_count"].clip(upper=12), df["tx_type"]))

print("\n=== TX_TYPE x CONTINUITY_QFLAG x PARTICIPANT_COUNT (mean/rate of N) ===")
print(df.groupby(["tx_type", df["participant_count"].clip(upper=10)], observed=True)["path_has_gap"].agg(["count", "mean"]))

# Check start/end of BIC8_PATH
starts_match = df["bic8_path"].str.slice(0, 8) == df["ordering_bic8"]
ends_match = df["bic8_path"].str.slice(-8) == df["beneficiary_bic8"]
print(f"\nBIC8_PATH starts with ORDERING_BIC8: {starts_match.mean():.6f} (mismatches={(~starts_match).sum()})")
print(f"BIC8_PATH ends with BENEFICIARY_BIC8: {ends_match.mean():.6f} (mismatches={(~ends_match).sum()})")
if (~starts_match).sum() > 0 or (~ends_match).sum() > 0:
    print("Sample start/end mismatch:", df[~starts_match | ~ends_match][["ordering_bic8", "beneficiary_bic8", "bic8_path", "tx_type", "continuity_qflag"]].head(5))

# Check visible BICs vs participant_count
df["vis_diff"] = df["participant_count"] - df["path_visible_bics"]
print("\nparticipant_count - path_visible_bics by continuity_qflag:")
print(pd.crosstab(df["vis_diff"].clip(-2, 10), df["continuity_qflag"]))

# 3. Domestic vs Cross-border BIC countries
df["same_bic_country"] = df["ord_bic_country"] == df["ben_bic_country"]
print("\n=== 3. DOMESTIC (same_bic_country) vs CROSS-BORDER ===")
print("same_bic_country:", df["same_bic_country"].value_counts().to_dict())
print(pd.crosstab(df["same_bic_country"], df["tx_type"], normalize="index").round(4))
print(pd.crosstab(df["same_bic_country"], df["fx_count"].clip(upper=5), normalize="index").round(4))

# 4. Unique entities
print("\n=== 4. ENTITY COUNTS ===")
print("Unique ordering_bic8:", df["ordering_bic8"].nunique())
print("Unique beneficiary_bic8:", df["beneficiary_bic8"].nunique())
print("Unique ord_bic_country:", df["ord_bic_country"].nunique())
print("Unique ben_bic_country:", df["ben_bic_country"].nunique())
print("Unique ordering_account:", df["ordering_account"].nunique())
print("Unique beneficiary_account:", df["beneficiary_account"].nunique())

print("\nTop 20 ordering BIC countries:")
print(df["ord_bic_country"].value_counts().head(20))
print("\nTop 20 beneficiary BIC countries:")
print(df["ben_bic_country"].value_counts().head(20))

# 5. USD_AMOUNT by segment
print("\n=== 5. USD_AMOUNT BY PARTY PAIR & TX_TYPE ===")
df["party_pair"] = df["ordering_type"] + "2" + df["beneficiary_type"]
print(df.groupby("party_pair")["usd_amount"].agg(["count", "mean", "median", "std", "min", "max"]))
print(df.groupby("party_pair")["usd_amount"].quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).unstack())

print("\nUSD_AMOUNT by ANOMALY_TYPE:")
print(df.groupby("anomaly_type")["usd_amount"].agg(["count", "mean", "median", "min", "max"]))

# 6. TOTAL_DURATION by fx_count, participant_count, tx_type, continuity_qflag
print("\n=== 6. TOTAL_DURATION SUMMARY ===")
print("Duration quantiles:", df["total_duration"].quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).to_dict())
print("\nMedian & Mean Duration by tx_type x continuity_qflag:")
print(df.groupby(["tx_type", "continuity_qflag"])["total_duration"].agg(["count", "mean", "median", "std"]))
print("\nMedian & Mean Duration by participant_count (2..7):")
print(df[df["participant_count"] <= 7].groupby("participant_count")["total_duration"].agg(["count", "mean", "median", "std"]))
print("\nMedian & Mean Duration by fx_count (0..5):")
print(df[df["fx_count"] <= 5].groupby("fx_count")["total_duration"].agg(["count", "mean", "median", "std"]))

# 7. Token & Country match stats
print("\n=== 7. TOKEN & DETAIL FEATURES ===")
df["ord_cc_match"] = df["ord_inferred_cc"] == df["ord_bic_country"]
df["ben_cc_match"] = df["ben_inferred_cc"] == df["ben_bic_country"]
for side in ["ord", "ben"]:
    has_cc = df[f"{side}_inferred_cc"] != ""
    print(f"{side}: has_inferred_cc={has_cc.mean():.4f}, matches_bic_country_when_present={(df.loc[has_cc, f'{side}_cc_match']).mean():.4f}")
for col in ["ord_has_iban", "ben_has_iban", "ord_has_lb35", "ben_has_lb35", "ord_has_trunc", "ben_has_trunc",
            "ord_is_struct_123", "ben_is_struct_123", "ord_has_acct_line", "ben_has_acct_line",
            "ord_has_double_slash", "ben_has_double_slash", "ord_has_biz", "ord_has_person", "ben_has_biz", "ben_has_person"]:
    print(f"  {col}: {df[col].mean():.5f}")

print("ord_fin_word non-empty rate:", (df["ord_fin_word"] != "").mean())
print("ben_fin_word non-empty rate:", (df["ben_fin_word"] != "").mean())

print(f"\nElapsed: {time.time()-t0:.1f}s")
