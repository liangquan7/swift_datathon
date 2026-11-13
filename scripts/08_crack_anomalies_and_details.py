import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

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
]
df = pd.read_parquet("work/core_parquet", columns=cols)
print(f"Loaded core in {time.time()-t0:.1f}s")

def get_details(row_ids):
    """Fetch raw DETAILS/TOKENS for specific row_ids using partition index (250k per part)."""
    row_ids = sorted(set(int(r) for r in row_ids))
    by_part = {}
    for r in row_ids:
        by_part.setdefault(r // 250_000, []).append(r)
    dfs = []
    for p, rids in by_part.items():
        part = pd.read_parquet(f"work/details_parquet/part_{p:02d}.parquet")
        dfs.append(part[part["row_id"].isin(rids)])
    return pd.concat(dfs, ignore_index=True)

# === 1. CUSTOMER_RECALL DEEP INVESTIGATION ===
print("\n=== 1. CUSTOMER_RECALL DEEP DIVE ===")
cr = df[df["anomaly_type"] == "CUSTOMER_RECALL"].copy()
cr["party_pair"] = cr["ordering_type"] + "2" + cr["beneficiary_type"]
print("CUSTOMER_RECALL count:", len(cr))
print("Compare CUSTOMER_RECALL vs NORMAL on key stats:")
norm = df[df["anomaly_type"] == ""]
print("  dur median/mean/p90:", cr["total_duration"].median(), cr["total_duration"].mean(), cr["total_duration"].quantile(0.9),
      "vs normal:", norm["total_duration"].median(), norm["total_duration"].mean(), norm["total_duration"].quantile(0.9))
print("  amt median/mean/p90:", cr["usd_amount"].median(), cr["usd_amount"].mean(), cr["usd_amount"].quantile(0.9),
      "vs normal:", norm["usd_amount"].median(), norm["usd_amount"].mean(), norm["usd_amount"].quantile(0.9))
print("  epoch min/max:", cr["epoch_seconds"].min(), cr["epoch_seconds"].max())
print("  QFLAG N rate:", (cr["continuity_qflag"] == "N").mean(), "vs normal:", (norm["continuity_qflag"] == "N").mean())
print("  fx_count mean:", cr["fx_count"].mean(), "vs normal:", norm["fx_count"].mean())
print("  participant_count mean:", cr["participant_count"].mean(), "vs normal:", norm["participant_count"].mean())
print("  tx_type dist:", cr["tx_type"].value_counts(normalize=True).round(4).to_dict())
print("  party_pair dist:", cr["party_pair"].value_counts(normalize=True).round(4).to_dict())
print("  ord_fin_word non-empty:", (cr["ord_fin_word"] != "").mean(), "ben_fin_word non-empty:", (cr["ben_fin_word"] != "").mean())
print("  ord_fin_word dist:", cr.loc[cr["ord_fin_word"] != "", "ord_fin_word"].value_counts().to_dict())
print("  ben_fin_word dist:", cr.loc[cr["ben_fin_word"] != "", "ben_fin_word"].value_counts().to_dict())

# Check row_id neighbors of CUSTOMER_RECALL & also check if same (ord, ben) or reverse (ben, ord) has matching amounts/close timestamps!
cr_set = set(cr["row_id"])
cr_neighbors = []
for rid in cr["row_id"].head(5):
    cr_neighbors.extend([rid - 1, rid, rid + 1])
print("\nNeighbors of first 5 CUSTOMER_RECALL by row_id:")
print(df[df["row_id"].isin(cr_neighbors)][["row_id", "epoch_seconds", "currency_code", "usd_amount", "ordering_account", "beneficiary_account", "anomaly_type"]].to_string())

# Check reverse pair (ben -> ord) or same pair (ord -> ben) close in time or exact same amount!
cr_ord = set(cr["ordering_account"])
cr_ben = set(cr["beneficiary_account"])
rel = df[(df["ordering_account"].isin(cr_ord) & df["beneficiary_account"].isin(cr_ben)) |
         (df["ordering_account"].isin(cr_ben) & df["beneficiary_account"].isin(cr_ord))].copy()
# Check for each CUSTOMER_RECALL row whether there is a reverse tx with same/close amount or close timestamp!
same_amt_rev = 0
close_t_rev = 0
same_amt_fwd = 0
close_t_fwd = 0
for _, r in cr.iterrows():
    rev = rel[(rel["ordering_account"] == r["beneficiary_account"]) & (rel["beneficiary_account"] == r["ordering_account"])]
    if len(rev) > 0:
        if (np.abs(rev["usd_amount"] - r["usd_amount"]) / r["usd_amount"] < 0.05).any():
            same_amt_rev += 1
        if (np.abs(rev["epoch_seconds"] - r["epoch_seconds"]) < 86400).any():
            close_t_rev += 1
    fwd = rel[(rel["ordering_account"] == r["ordering_account"]) & (rel["beneficiary_account"] == r["beneficiary_account"]) & (rel["row_id"] != r["row_id"])]
    if len(fwd) > 0:
        if (np.abs(fwd["usd_amount"] - r["usd_amount"]) / r["usd_amount"] < 0.05).any():
            same_amt_fwd += 1
        if (np.abs(fwd["epoch_seconds"] - r["epoch_seconds"]) < 86400).any():
            close_t_fwd += 1
print(f"CUSTOMER_RECALL match stats: same_amt_rev={same_amt_rev}, close_t_rev(<24h)={close_t_rev}, same_amt_fwd={same_amt_fwd}, close_t_fwd(<24h)={close_t_fwd}")

# Inspect raw DETAILS/TOKENS for 8 CUSTOMER_RECALL rows!
cr_det = get_details(cr["row_id"].head(8))
cr_merged = cr.head(8).merge(cr_det, on="row_id")
print("\nRaw DETAILS/TOKENS of 8 CUSTOMER_RECALL rows:")
for _, r in cr_merged.iterrows():
    print(f"  row={r['row_id']} cur={r['currency_code']} amt={r['usd_amount']} {r['ordering_bic8']}->{r['beneficiary_bic8']} dur={r['total_duration']} {r['ordering_type']}2{r['beneficiary_type']}")
    print("    ORD:", repr(r["ORDERING_DETAILS"]), "| TOK:", repr(r["ORDERING_TOKENS"]))
    print("    BEN:", repr(r["BENEFICIARY_DETAILS"]), "| TOK:", repr(r["BENEFICIARY_TOKENS"]))

# === 2. ACCOUNT_REPLACEMENT_SCAM & CEO_SCAM DETAILS ===
print("\n=== 2. ACCOUNT_REPLACEMENT_SCAM & CEO_SCAM DETAILS ===")
ars = df[df["anomaly_type"] == "ACCOUNT_REPLACEMENT_SCAM"]
# Check if ord_acct has prior normal tx with matching BENEFICIARY_DETAILS (name/city) under a DIFFERENT beneficiary_account!
ars_sample = ars.head(5)
ars_det = get_details(ars_sample["row_id"])
print("Sample ACCOUNT_REPLACEMENT_SCAM raw details + check against ord_acct's normal beneficiaries:")
for _, r in ars_sample.merge(ars_det, on="row_id").iterrows():
    print(f"  row={r['row_id']} ord={r['ordering_account']} -> ben={r['beneficiary_account']} ({r['beneficiary_bic8']}) amt={r['usd_amount']}")
    print("    BEN_DET:", repr(r["BENEFICIARY_DETAILS"]))
    print("    BEN_TOK:", repr(r["BENEFICIARY_TOKENS"]))
    # Find normal tx from same ord_acct
    ord_norm = df[(df["anomaly_type"] == "") & (df["ordering_account"] == r["ordering_account"])]
    print(f"    ord_acct has {len(ord_norm)} normal tx across {ord_norm['beneficiary_account'].nunique()} ben_accts; same ben_bic_country? {(ord_norm['ben_bic_country']==r['ben_bic_country']).sum()}")

# Check shared mule beneficiary_accounts across CEO_SCAM, ACCOUNT_REPLACEMENT_SCAM, ROMANCE_SCAM!
ceo = df[df["anomaly_type"] == "CEO_SCAM"]
rs = df[df["anomaly_type"] == "ROMANCE_SCAM"]
ceo_bens = set(ceo["beneficiary_account"])
ars_bens = set(ars["beneficiary_account"])
rs_bens = set(rs["beneficiary_account"])
print("\nOverlap of mule beneficiary_accounts:")
print("  CEO & ARS overlap:", len(ceo_bens & ars_bens))
print("  CEO & RS overlap:", len(ceo_bens & rs_bens))
print("  ARS & RS overlap:", len(ars_bens & rs_bens))
print("  Top 10 repeated ben_accts in CEO_SCAM:", ceo["beneficiary_account"].value_counts().head(10).to_dict())
print("  Top 10 repeated ben_accts in ARS:", ars["beneficiary_account"].value_counts().head(10).to_dict())

# Check if a mule ben_acct in CEO/ARS has multiple beneficiary_bic8s or different names in normal tx!
top_ceo_ben = ceo["beneficiary_account"].value_counts().index[0]
top_ceo_rows = df[df["beneficiary_account"] == top_ceo_ben].head(6)
print(f"\nRows for top CEO_SCAM beneficiary_account ({top_ceo_ben}):")
print(top_ceo_rows[["row_id", "ordering_account", "beneficiary_bic8", "usd_amount", "anomaly_type", "ben_inferred_cc"]].to_string())
print(get_details(top_ceo_rows["row_id"])[["row_id", "BENEFICIARY_DETAILS"]].to_string())

# === 3. FAT_FINGER_MISTAKE EXACT MULTIPLIER DISTRIBUTION ===
print("\n=== 3. FAT_FINGER_MISTAKE EXACT MULTIPLIER ===")
ff = df[df["anomaly_type"] == "FAT_FINGER_MISTAKE"].copy()
norm_ord_med = df[df["anomaly_type"] == ""].groupby("ordering_account")["usd_amount"].median()
ff["ord_med"] = ff["ordering_account"].map(norm_ord_med)
# Also compute baseline median by (ord_bic_country, ordering_type, beneficiary_type)
norm_grp_med = df[df["anomaly_type"] == ""].groupby(["ord_bic_country", "ordering_type", "beneficiary_type"])["usd_amount"].median()
ff = ff.merge(norm_grp_med.rename("grp_med"), on=["ord_bic_country", "ordering_type", "beneficiary_type"], how="left")
ff["log10_ratio_ord"] = np.log10(ff["usd_amount"] / ff["ord_med"])
ff["log10_ratio_grp"] = np.log10(ff["usd_amount"] / ff["grp_med"])

bins = np.arange(-4.5, 6.5, 0.5)
h_ord, e = np.histogram(ff["log10_ratio_ord"].dropna(), bins=bins)
print("Histogram of log10(fat_finger / sender_median):")
for b0, b1, c in zip(e[:-1], e[1:], h_ord):
    print(f"  [{b0:5.1f}, {b1:5.1f}): {c:4d} ({c/len(ff)*100:5.1f}%)")

# Check trailing zeros of usd_amount! E.g. string representation or integer part!
ff["is_mult_10"] = (ff["usd_amount"] % 10) == 0
ff["is_mult_100"] = (ff["usd_amount"] % 100) == 0
ff["is_mult_1000"] = (ff["usd_amount"] % 1000) == 0
ff["is_int"] = (ff["usd_amount"] % 1) == 0
ff["has_1dec"] = ((ff["usd_amount"] * 10) % 1 == 0) & (~ff["is_int"])
print("Decimal pattern in FAT_FINGER vs NORMAL:")
print("  FF: int=", ff["is_int"].mean(), "mult10=", ff["is_mult_10"].mean(), "mult100=", ff["is_mult_100"].mean(), "mult1000=", ff["is_mult_1000"].mean())

# === 4. ROMANCE_SCAM EXACT ESCALATION LAW ===
print("\n=== 4. ROMANCE_SCAM EXACT ESCALATION LAW ===")
rs = df[df["anomaly_type"] == "ROMANCE_SCAM"].sort_values(["ordering_account", "beneficiary_account", "epoch_seconds"]).copy()
rs["prev_amt"] = rs.groupby(["ordering_account", "beneficiary_account"])["usd_amount"].shift(1)
rs["ratio_step"] = rs["usd_amount"] / rs["prev_amt"]
print("ROMANCE_SCAM step-to-step ratio (amt_k / amt_{k-1}) quantiles:")
print(rs["ratio_step"].dropna().quantile([0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]).round(4).to_dict())
rs["log_ratio"] = np.log(rs["ratio_step"])
print("ROMANCE_SCAM ln(amt_k / amt_{k-1}) mean/std/median:", rs["log_ratio"].mean(), rs["log_ratio"].std(), rs["log_ratio"].median())
print("ROMANCE_SCAM first payment amount quantiles:", rs.groupby(["ordering_account", "beneficiary_account"])["usd_amount"].first().quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).round(2).to_dict())

# === 5. WHY DO 61,606 BENEFICIARY ACCOUNTS HAVE >1 BENEFICIARY_BIC8? ===
print("\n=== 5. BENEFICIARY ACCOUNTS WITH >1 BIC8 & BENEFICIARY_DETAILS COUNTRY CONSISTENCY ===")
ben_bic_cnt = df.groupby("beneficiary_account")["beneficiary_bic8"].nunique()
multi_ben = set(ben_bic_cnt[ben_bic_cnt > 1].index)
sub_multi = df[df["beneficiary_account"].isin(list(multi_ben)[:5000])].copy()
print("Among beneficiary_accounts with >1 BIC8:")
print("  ben_has_iban rate:", sub_multi["ben_has_iban"].mean(), "vs overall:", df["ben_has_iban"].mean())
print("  Sample 3 multi-BIC beneficiary accounts:")
for acct in list(multi_ben)[:3]:
    print(" ", acct, df[df["beneficiary_account"] == acct][["beneficiary_bic8", "ben_inferred_cc", "beneficiary_type", "anomaly_type"]].drop_duplicates().to_dict("records"))

# Also check if a single beneficiary_account (with 1 BIC8) ALWAYS has the same ben_inferred_cc!
ben_cc_cnt = df[df["ben_inferred_cc"] != ""].groupby("beneficiary_account")["ben_inferred_cc"].nunique()
print("Distinct ben_inferred_cc per beneficiary_account value counts (head):", ben_cc_cnt.value_counts().head(5).to_dict())

# === 6. CHECK THE 1 EXACT FULL DUPLICATE & THE 12 EXTRA NO-TS DUPLICATES ===
print("\n=== 6. EXACT FULL DUPLICATES & NON-DP NO-TS DUPLICATES ===")
cols_all = [c for c in df.columns if c not in ("row_id",)]
exact_dups = df[df.duplicated(subset=cols_all, keep=False)]
print("Exact full duplicates (all core cols match):", len(exact_dups))
if len(exact_dups) > 0:
    print(exact_dups[["row_id", "epoch_seconds", "currency_code", "usd_amount", "ordering_account", "beneficiary_account", "anomaly_type"]].to_string())

cols_nots = [c for c in cols_all if c != "epoch_seconds"]
nots_dups = df[df.duplicated(subset=cols_nots, keep=False)]
print("No-TS duplicates total rows:", len(nots_dups), "by anomaly_type:", nots_dups["anomaly_type"].value_counts().to_dict())
nots_non_dp = nots_dups[nots_dups["anomaly_type"] != "DUPLICATE_PAYMENT"]
if len(nots_non_dp) > 0:
    print("Sample non-DP no-TS duplicates:")
    print(nots_non_dp.sort_values(["ordering_account", "beneficiary_account", "usd_amount", "epoch_seconds"])[["row_id", "epoch_seconds", "currency_code", "usd_amount", "ordering_account", "beneficiary_account", "bic8_path", "total_duration", "anomaly_type"]].head(20).to_string())

print(f"\nElapsed: {time.time()-t0:.1f}s")
