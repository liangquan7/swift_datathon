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
]
df = pd.read_parquet("work/core_parquet", columns=cols)
print(f"Loaded in {time.time()-t0:.1f}s")

anom = df[df["anomaly_type"] != ""].copy()
print("\n=== 1. ANOMALY OVERVIEW CROSS-TABS ===")
anom["party_pair"] = anom["ordering_type"] + "2" + anom["beneficiary_type"]
print("Party pair by ANOMALY_TYPE:")
print(pd.crosstab(anom["anomaly_type"], anom["party_pair"], margins=True))
print("\nTX_TYPE by ANOMALY_TYPE:")
print(pd.crosstab(anom["anomaly_type"], anom["tx_type"], normalize="index").round(4))
print("\nCONTINUITY_QFLAG by ANOMALY_TYPE:")
print(pd.crosstab(anom["anomaly_type"], anom["continuity_qflag"], normalize="index").round(4))

anom["path_first"] = anom["bic8_path"].str.slice(0, 8)
anom["path_last"] = anom["bic8_path"].str.slice(-8)
anom["start_mis"] = anom["path_first"] != anom["ordering_bic8"]
anom["end_mis"] = anom["path_last"] != anom["beneficiary_bic8"]
print("\nstart_mis, end_mis, median dur, median amt, fin_word rates by ANOMALY_TYPE:")
anom["has_ord_fin"] = anom["ord_fin_word"] != ""
anom["has_ben_fin"] = anom["ben_fin_word"] != ""
print(anom.groupby("anomaly_type").agg(
    n=("row_id", "size"),
    start_mis=("start_mis", "mean"),
    end_mis=("end_mis", "mean"),
    med_amt=("usd_amount", "median"),
    mean_amt=("usd_amount", "mean"),
    med_dur=("total_duration", "median"),
    mean_dur=("total_duration", "mean"),
    ord_fin=("has_ord_fin", "mean"),
    ben_fin=("has_ben_fin", "mean"),
    n_ord_acct=("ordering_account", "nunique"),
    n_ben_acct=("beneficiary_account", "nunique"),
))

# === 2. DUPLICATE_PAYMENT & ALL EXACT / NEAR DUPLICATES ===
print("\n=== 2. DUPLICATE_PAYMENT & ALL NO-TS DUPLICATES ===")
# Key: (ordering_account, beneficiary_account, currency_code, usd_amount)
dups_mask = df.duplicated(subset=["ordering_account", "beneficiary_account", "currency_code", "usd_amount"], keep=False)
dup_candidates = df[dups_mask].copy()
print("Rows sharing (ord_acct, ben_acct, currency, usd_amount):", len(dup_candidates))
print("Their anomaly_type counts:", dup_candidates["anomaly_type"].value_counts().to_dict())

# Also check all DUPLICATE_PAYMENT rows: how many have a partner in dup_candidates?
dp = df[df["anomaly_type"] == "DUPLICATE_PAYMENT"]
print("DUPLICATE_PAYMENT total:", len(dp), "in dup_candidates:", dup_candidates[dup_candidates["anomaly_type"] == "DUPLICATE_PAYMENT"].shape[0])
# Inspect first 10 groups containing DUPLICATE_PAYMENT
dp_keys = set(zip(dp["ordering_account"], dp["beneficiary_account"], dp["currency_code"], dp["usd_amount"]))
in_dp_grp = dup_candidates[dup_candidates.apply(lambda r: (r["ordering_account"], r["beneficiary_account"], r["currency_code"], r["usd_amount"]) in dp_keys, axis=1)].sort_values(["ordering_account", "beneficiary_account", "usd_amount", "epoch_seconds"])
print("Total rows in groups containing at least one DUPLICATE_PAYMENT:", len(in_dp_grp))
print("Anomaly breakdown in those groups:", in_dp_grp["anomaly_type"].value_counts().to_dict())
print("Sample DUPLICATE_PAYMENT groups (first 16 rows):")
print(in_dp_grp[["row_id", "epoch_seconds", "currency_code", "usd_amount", "ordering_account", "beneficiary_account", "bic8_path", "total_duration", "anomaly_type"]].head(16).to_string())

# Distribution of group size & time diff for DUPLICATE_PAYMENT groups
grp_sizes = in_dp_grp.groupby(["ordering_account", "beneficiary_account", "currency_code", "usd_amount"]).size()
print("DUPLICATE_PAYMENT group size distribution:", grp_sizes.value_counts().to_dict())
diffs = in_dp_grp.groupby(["ordering_account", "beneficiary_account", "currency_code", "usd_amount"])["epoch_seconds"].diff().dropna()
print("Time delta (seconds) between duplicate pairs quantiles:", diffs.quantile([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).to_dict())

# Check rows in dup_candidates that are NOT DUPLICATE_PAYMENT!
non_dp_dups = dup_candidates[~dup_candidates.index.isin(in_dp_grp.index)]
print("\nRows sharing (ord, ben, cur, amt) WITHOUT DUPLICATE_PAYMENT label:", len(non_dp_dups))
print("Their anomaly_type:", non_dp_dups["anomaly_type"].value_counts().to_dict())
print("Sample non-DP repeated same-amount pairs (top 10):")
print(non_dp_dups.sort_values(["ordering_account", "beneficiary_account", "usd_amount", "epoch_seconds"])[["row_id", "epoch_seconds", "currency_code", "usd_amount", "ordering_account", "beneficiary_account", "anomaly_type"]].head(12).to_string())

# === 3. ROMANCE_SCAM ===
print("\n=== 3. ROMANCE_SCAM ===")
rs = df[df["anomaly_type"] == "ROMANCE_SCAM"].sort_values(["ordering_account", "beneficiary_account", "epoch_seconds"])
rs_pairs = rs.groupby(["ordering_account", "beneficiary_account"]).agg(
    n=("row_id", "size"),
    min_t=("epoch_seconds", "min"),
    max_t=("epoch_seconds", "max"),
    first_amt=("usd_amount", "first"),
    last_amt=("usd_amount", "last"),
    med_amt=("usd_amount", "median"),
    n_cur=("currency_code", "nunique"),
    n_path=("bic8_path", "nunique"),
)
print("Unique (ord_acct, ben_acct) pairs in ROMANCE_SCAM:", len(rs_pairs))
print("Payments per pair distribution:", rs_pairs["n"].value_counts().sort_index().to_dict())
print("Unique ord_accts:", rs["ordering_account"].nunique(), "Unique ben_accts:", rs["beneficiary_account"].nunique())
# Check if victims/scammers also appear in normal transactions!
rs_ord_set = set(rs["ordering_account"])
rs_ben_set = set(rs["beneficiary_account"])
print("ROMANCE_SCAM ord_accts in normal tx:", df[(df["anomaly_type"] == "") & (df["ordering_account"].isin(rs_ord_set))].shape[0])
print("ROMANCE_SCAM ben_accts in normal tx:", df[(df["anomaly_type"] == "") & (df["beneficiary_account"].isin(rs_ben_set))].shape[0])
# Check amount progression within pairs (ratio of payment k to payment 1)
rs["seq_idx"] = rs.groupby(["ordering_account", "beneficiary_account"]).cumcount() + 1
rs["dt_prev"] = rs.groupby(["ordering_account", "beneficiary_account"])["epoch_seconds"].diff()
print("ROMANCE_SCAM median usd_amount by seq_idx (1..10):")
print(rs[rs["seq_idx"] <= 10].groupby("seq_idx")["usd_amount"].agg(["count", "median", "mean"]))
print("ROMANCE_SCAM dt_prev quantiles (seconds):", rs["dt_prev"].dropna().quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).to_dict())
print("Sample 2 ROMANCE_SCAM chains:")
sample_pairs = rs_pairs[rs_pairs["n"] >= 5].head(2).index
for p in sample_pairs:
    print(rs[(rs["ordering_account"] == p[0]) & (rs["beneficiary_account"] == p[1])][["epoch_seconds", "currency_code", "usd_amount", "ordering_bic8", "beneficiary_bic8", "bic8_path", "ord_inferred_cc", "ben_inferred_cc"]].to_string())

# === 4. FAT_FINGER_MISTAKE ===
print("\n=== 4. FAT_FINGER_MISTAKE ===")
ff = df[df["anomaly_type"] == "FAT_FINGER_MISTAKE"].copy()
ff["party_pair"] = ff["ordering_type"] + "2" + ff["beneficiary_type"]
ff_ord_set = set(ff["ordering_account"])
# Compare each fat-finger tx against sender's normal median!
norm_ff_ord = df[(df["anomaly_type"] == "") & (df["ordering_account"].isin(ff_ord_set))].groupby("ordering_account")["usd_amount"].agg(["count", "median"])
ff = ff.merge(norm_ff_ord, on="ordering_account", how="left")
ff["ratio_to_ord_med"] = ff["usd_amount"] / ff["median"]
print("FAT_FINGER rows with sender normal history:", ff["count"].notna().sum(), "/", len(ff))
print("FAT_FINGER ratio_to_ord_med quantiles:", ff["ratio_to_ord_med"].dropna().quantile([0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).round(2).to_dict())
# Also check exact log10(ratio) or digit structure / cents!
ff["cents"] = np.round((ff["usd_amount"] % 1) * 100).astype(int)
print("FAT_FINGER cents==0 rate:", (ff["cents"] == 0).mean(), "vs clean rate ~0.074")
print("FAT_FINGER divisible by 10 rate:", ((ff["usd_amount"] % 10) == 0).mean(), "divisible by 100:", ((ff["usd_amount"] % 100) == 0).mean())
# Check if fat finger pairs with a normal transaction of 1/10, 1/100, 1/1000!
print("Sample 10 FAT_FINGER rows with sender median:")
print(ff[["row_id", "party_pair", "currency_code", "usd_amount", "median", "ratio_to_ord_med", "count"]].head(10).to_string())

# === 5. CEO_SCAM, ACCOUNT_REPLACEMENT_SCAM, CUSTOMER_RECALL ===
for atype in ["CEO_SCAM", "ACCOUNT_REPLACEMENT_SCAM", "CUSTOMER_RECALL"]:
    print(f"\n=== 5. {atype} ===")
    sub = df[df["anomaly_type"] == atype].copy()
    sub_ord = set(sub["ordering_account"])
    sub_ben = set(sub["beneficiary_account"])
    norm_ord = df[(df["anomaly_type"] == "") & (df["ordering_account"].isin(sub_ord))]
    norm_ben = df[(df["anomaly_type"] == "") & (df["beneficiary_account"].isin(sub_ben))]
    print(f"  n={len(sub)}, unique ord_acct={len(sub_ord)}, unique ben_acct={len(sub_ben)}")
    print(f"  ord_acct appears in normal tx: {norm_ord['ordering_account'].nunique()} accounts ({len(norm_ord)} rows)")
    print(f"  ben_acct appears in normal tx: {norm_ben['beneficiary_account'].nunique()} accounts ({len(norm_ben)} rows)")
    # Check if (ord_acct, ben_acct) ever appeared together normally!
    sub_pairs = set(zip(sub["ordering_account"], sub["beneficiary_account"]))
    norm_sub_pairs = norm_ord[norm_ord.apply(lambda r: (r["ordering_account"], r["beneficiary_account"]) in sub_pairs, axis=1)]
    print(f"  Exact (ord_acct, ben_acct) pair ALSO in normal tx: {len(norm_sub_pairs)} rows")
    # Check reverse pair (ben_acct, ord_acct) in normal tx!
    rev_pairs = set(zip(sub["beneficiary_account"], sub["ordering_account"]))
    rev_hits = df[df["ordering_account"].isin(sub_ben) & df["beneficiary_account"].isin(sub_ord)]
    rev_hits = rev_hits[rev_hits.apply(lambda r: (r["ordering_account"], r["beneficiary_account"]) in rev_pairs, axis=1)]
    print(f"  Reverse pair (ben->ord) in dataset: {len(rev_hits)} rows")
    # Check ben_inferred_cc vs ben_bic_country and ben_has_iban!
    has_bcc = sub["ben_inferred_cc"] != ""
    print(f"  ben_inferred_cc == ben_bic_country: {(sub.loc[has_bcc, 'ben_inferred_cc'] == sub.loc[has_bcc, 'ben_bic_country']).mean():.4f}")
    print(f"  ben_has_iban: {sub['ben_has_iban'].mean():.4f}, ord_has_iban: {sub['ord_has_iban'].mean():.4f}")
    print(f"  ord_is_struct_123: {sub['ord_is_struct_123'].mean():.4f}, ben_is_struct_123: {sub['ben_is_struct_123'].mean():.4f}")
    print(f"  Top 10 ben_bic_country: {sub['ben_bic_country'].value_counts().head(10).to_dict()}")
    print(f"  Top 10 ord_bic_country: {sub['ord_bic_country'].value_counts().head(10).to_dict()}")
    print(f"  Sample 5 rows:")
    print(sub[["row_id", "epoch_seconds", "currency_code", "usd_amount", "ordering_bic8", "beneficiary_bic8", "tx_type", "total_duration", "ordering_type", "beneficiary_type", "ord_inferred_cc", "ben_inferred_cc"]].head(5).to_string())

print(f"\nElapsed: {time.time()-t0:.1f}s")
