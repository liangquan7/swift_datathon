import time
import numpy as np
import pandas as pd

t0 = time.time()
cols = [
    "row_id", "epoch_seconds", "currency_code", "usd_amount",
    "ordering_bic8", "beneficiary_bic8", "bic8_path",
    "ord_bic_bank", "ord_bic_country", "ben_bic_bank", "ben_bic_country",
    "path_has_gap", "total_duration", "continuity_qflag", "fx_count",
    "participant_count", "tx_type",
    "ordering_account", "beneficiary_account",
    "ordering_type", "beneficiary_type", "anomaly_type",
    "ord_inferred_cc", "ben_inferred_cc",
    "ord_has_iban", "ben_has_iban",
]
df = pd.read_parquet("work/core_parquet", columns=cols)
print(f"Loaded in {time.time()-t0:.1f}s")

# 1. Start/End mismatch diagnostic: compare BIC8_PATH[0] vs ORDERING_BIC8 against IBAN bank code & ord_inferred_cc
df["path_first"] = df["bic8_path"].str.slice(0, 8)
df["path_last"] = df["bic8_path"].str.slice(-8)
df["start_mis"] = df["path_first"] != df["ordering_bic8"]
df["end_mis"] = df["path_last"] != df["beneficiary_bic8"]

print("\n=== 1. DIAGNOSTIC ON START/END MISMATCHES ===")
# When ordering_account is an IBAN (_FAKE_CCKKBBBB...), extract country (chars 6:8) and bank (chars 10:14)
iban_ord = df[df["ord_has_iban"]].copy()
iban_ord["iban_cc"] = iban_ord["ordering_account"].str.slice(6, 8)
iban_ord["iban_bank"] = iban_ord["ordering_account"].str.slice(10, 14)
iban_ord["pf_bank"] = iban_ord["path_first"].str.slice(0, 4)
iban_ord["pf_cc"] = iban_ord["path_first"].str.slice(4, 6)
print("When start_mis == False (IBAN ord):")
m0 = iban_ord[~iban_ord["start_mis"]]
print("  iban_bank == ord_bic_bank:", (m0["iban_bank"] == m0["ord_bic_bank"]).mean())
print("  iban_cc == ord_bic_country:", (m0["iban_cc"] == m0["ord_bic_country"]).mean())
print("When start_mis == True (IBAN ord):")
m1 = iban_ord[iban_ord["start_mis"]]
print("  count:", len(m1))
print("  iban_bank == ord_bic_bank:", (m1["iban_bank"] == m1["ord_bic_bank"]).mean())
print("  iban_bank == path_first_bank:", (m1["iban_bank"] == m1["pf_bank"]).mean())
print("  ord_inferred_cc == ord_bic_country (when non-empty):", (m1.loc[m1["ord_inferred_cc"] != "", "ord_inferred_cc"] == m1.loc[m1["ord_inferred_cc"] != "", "ord_bic_country"]).mean())
print("  ord_inferred_cc == path_first_cc (when non-empty):", (m1.loc[m1["ord_inferred_cc"] != "", "ord_inferred_cc"] == m1.loc[m1["ord_inferred_cc"] != "", "pf_cc"]).mean())

# Also check BENEFICIARY IBAN!
iban_ben = df[df["ben_has_iban"]].copy()
iban_ben["iban_cc"] = iban_ben["beneficiary_account"].str.slice(6, 8)
iban_ben["iban_bank"] = iban_ben["beneficiary_account"].str.slice(10, 14)
iban_ben["pl_bank"] = iban_ben["path_last"].str.slice(0, 4)
iban_ben["pl_cc"] = iban_ben["path_last"].str.slice(4, 6)
print("\nWhen end_mis == False (IBAN ben):")
b0 = iban_ben[~iban_ben["end_mis"]]
print("  iban_bank == ben_bic_bank:", (b0["iban_bank"] == b0["ben_bic_bank"]).mean())
print("  iban_cc == ben_bic_country:", (b0["iban_cc"] == b0["ben_bic_country"]).mean())
print("When end_mis == True (IBAN ben):")
b1 = iban_ben[iban_ben["end_mis"]]
print("  count:", len(b1))
print("  iban_bank == ben_bic_bank:", (b1["iban_bank"] == b1["ben_bic_bank"]).mean())
print("  iban_bank == path_last_bank:", (b1["iban_bank"] == b1["pl_bank"]).mean())

# Check if start_mis / end_mis affects fx_count (when path_first country != ord_bic_country)!
d2 = df[df["participant_count"] == 2].copy()
d2["pf_cc"] = d2["path_first"].str.slice(4, 6)
d2["pl_cc"] = d2["path_last"].str.slice(4, 6)
print("\nDirect (part=2): fx_count==0 rate by (ord==ben==cur) vs (path_first==path_last==cur):")
# Check start_mis in Direct: when ord_bic_country==ben_bic_country!=pf_cc, is fx_count 0 or 1?
sub_mis = d2[d2["start_mis"] & ~d2["end_mis"]]
print("Direct start_mis only, mean fx_count when (ord_cc==ben_cc) vs (pf_cc==ben_cc):")
print("  ord_cc==ben_cc & pf_cc!=ben_cc:", sub_mis[(sub_mis["ord_bic_country"] == sub_mis["ben_bic_country"]) & (sub_mis["pf_cc"] != sub_mis["ben_bic_country"])]["fx_count"].value_counts().to_dict())
print("  ord_cc!=ben_cc & pf_cc==ben_cc:", sub_mis[(sub_mis["ord_bic_country"] != sub_mis["ben_bic_country"]) & (sub_mis["pf_cc"] == sub_mis["ben_bic_country"])]["fx_count"].value_counts().to_dict())

# Also check distribution of path_first when start_mis==True!
print("\nTop 15 path_first BICs when start_mis==True vs overall ordering_bic8:")
print("When start_mis==True:", df.loc[df["start_mis"], "path_first"].value_counts().head(10).to_dict())
print("Overall ordering_bic8:", df["ordering_bic8"].value_counts().head(10).to_dict())
print("Overall path_last when end_mis==True:", df.loc[df["end_mis"], "path_last"].value_counts().head(10).to_dict())
print("Overall beneficiary_bic8:", df["beneficiary_bic8"].value_counts().head(10).to_dict())

# 2. Check Anomaly start_mis & end_mis
print("\n=== 2. ANOMALY START/END MISMATCHES ===")
print(df.groupby("anomaly_type")[["start_mis", "end_mis", "path_has_gap"]].agg(["count", "mean"]))

# 3. Covered vs Indirect at BANK (ordering_bic8 / beneficiary_bic8) level!
print("\n=== 3. COVERED RATE BY ORDERING_BIC8 ===")
multi = df[df["participant_count"] >= 3].copy()
multi["is_covered"] = multi["tx_type"] == "Covered"
ord_cov = multi.groupby("ordering_bic8").agg(n=("is_covered", "size"), cov_rate=("is_covered", "mean"), country=("ord_bic_country", "first"))
print("Ordering BIC8s with >= 1000 multi-part tx, top 15 cov_rate:")
print(ord_cov[ord_cov["n"] >= 1000].sort_values("cov_rate", ascending=False).head(15))
print("Ordering BIC8s with >= 1000 multi-part tx, bottom 10 cov_rate:")
print(ord_cov[ord_cov["n"] >= 1000].sort_values("cov_rate", ascending=True).head(10))

# Check IM banks!
print("\nAll IM ordering BIC8s in multi-part:")
print(ord_cov[ord_cov["country"] == "IM"].sort_values("n", ascending=False).head(15))

# Check if Covered depends on BIC8_PATH intermediaries or repeated BICs!
print("\n=== 4. DURATION vs QFLAG/N_DOTS CONTROLLING FOR PARTICIPANT_COUNT ===")
df["n_dots"] = df["bic8_path"].str.count(r"\.\.\.")
print(df[df["participant_count"] <= 6].groupby(["tx_type", "participant_count", df["n_dots"].clip(upper=2)])["total_duration"].agg(["count", "mean", "median"]))

# 5. Entity Graph: Account persistence & cardinality
print("\n=== 5. ENTITY GRAPH CARDINALITIES ===")
print("Unique ordering_account:", df["ordering_account"].nunique())
print("Unique beneficiary_account:", df["beneficiary_account"].nunique())
overlap_acct = len(set(df["ordering_account"].unique()) & set(df["beneficiary_account"].unique()))
print("Accounts appearing as BOTH ordering and beneficiary:", overlap_acct)

# How many distinct ordering_bic8 per ordering_account?
ord_acct_bics = df.groupby("ordering_account")["ordering_bic8"].nunique()
print("ordering_bic8 per ordering_account value counts (head):", ord_acct_bics.value_counts().head(10).to_dict())
ord_acct_types = df.groupby("ordering_account")["ordering_type"].nunique()
print("ordering_type per ordering_account value counts:", ord_acct_types.value_counts().to_dict())

# Wait! For beneficiary_account, check how many distinct beneficiary_bic8 and beneficiary_type!
ben_acct_bics = df.groupby("beneficiary_account")["beneficiary_bic8"].nunique()
print("beneficiary_bic8 per beneficiary_account value counts (head):", ben_acct_bics.value_counts().head(10).to_dict())
ben_acct_types = df.groupby("beneficiary_account")["beneficiary_type"].nunique()
print("beneficiary_type per beneficiary_account value counts:", ben_acct_types.value_counts().to_dict())

print(f"\nElapsed: {time.time()-t0:.1f}s")
