import time
import numpy as np
import pandas as pd

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
]
df = pd.read_parquet("work/core_parquet", columns=cols)

print("=== 1. START/END MISMATCHES IN BIC8_PATH ===")
s_match = df["bic8_path"].str.slice(0, 8) == df["ordering_bic8"]
e_match = df["bic8_path"].str.slice(-8) == df["beneficiary_bic8"]
print("Start match:", s_match.value_counts().to_dict())
print("End match:", e_match.value_counts().to_dict())
print("Both mismatch:", (~s_match & ~e_match).sum())
print("\nStart mismatch x continuity_qflag:")
print(pd.crosstab(~s_match, df["continuity_qflag"]))
print("\nEnd mismatch x continuity_qflag:")
print(pd.crosstab(~e_match, df["continuity_qflag"]))
print("\nSample start mismatch where QFLAG==Y:")
print(df[~s_match & (df["continuity_qflag"] == "Y")][["row_id", "ordering_bic8", "beneficiary_bic8", "bic8_path", "tx_type", "participant_count", "anomaly_type"]].head(15).to_string())
print("\nSample end mismatch where QFLAG==Y:")
print(df[~e_match & (df["continuity_qflag"] == "Y")][["row_id", "ordering_bic8", "beneficiary_bic8", "bic8_path", "tx_type", "participant_count", "anomaly_type"]].head(15).to_string())
print("\nSample start mismatch where QFLAG==N:")
print(df[~s_match & (df["continuity_qflag"] == "N")][["row_id", "ordering_bic8", "beneficiary_bic8", "bic8_path", "tx_type", "participant_count", "anomaly_type"]].head(10).to_string())

print("\n=== 2. COUNT OF '...' IN BIC8_PATH ===")
n_dots = df["bic8_path"].str.count(r"\.\.\.")
print("n_dots value counts:")
print(n_dots.value_counts().sort_index())
n_semis = df["bic8_path"].str.count(";")
vis_bics_exact = (n_semis + 1) - n_dots
print("participant_count - vis_bics_exact value counts:")
print((df["participant_count"] - vis_bics_exact).value_counts().sort_index())
print("\nCross-tab n_dots vs participant_count (2..7):")
print(pd.crosstab(df["participant_count"].clip(upper=7), n_dots))

# Also check if Direct/Indirect/Covered differ on start/end mismatch or n_dots or anything!
print("\n=== 3. START/END MISMATCH RATE BY TX_TYPE, PARTICIPANT_COUNT, ANOMALY ===")
df["start_mis"] = ~s_match
df["end_mis"] = ~e_match
print(df.groupby("tx_type")[["start_mis", "end_mis", "path_has_gap"]].mean())
print(df.groupby("anomaly_type")[["start_mis", "end_mis", "path_has_gap"]].agg(["count", "mean"]))

# Check if start_mis/end_mis is linked to same_bic_country or specific countries/BICs or Covered!
print("\n=== 4. WHAT DISTINGUISHES COVERED FROM INDIRECT? ===")
multi = df[df["participant_count"] >= 3].copy()
print("Multi-participant count:", len(multi))
print("Covered rate overall (given part>=3):", (multi["tx_type"] == "Covered").mean())
print("\nCovered rate by participant_count (3..10):")
print(multi[multi["participant_count"] <= 10].groupby("participant_count").apply(lambda g: (g["tx_type"] == "Covered").mean(), include_groups=False))
print("\nCovered rate by same_bic_country:")
multi["same_country"] = multi["ord_bic_country"] == multi["ben_bic_country"]
print(multi.groupby("same_country").apply(lambda g: pd.Series({
    "n": len(g),
    "covered_rate": (g["tx_type"] == "Covered").mean(),
    "fx0_rate": (g["fx_count"] == 0).mean(),
    "gap_rate": g["path_has_gap"].mean(),
}), include_groups=False))

print("\nCovered rate by (ordering_type, beneficiary_type):")
print(multi.groupby(["ordering_type", "beneficiary_type"]).apply(lambda g: (g["tx_type"] == "Covered").mean(), include_groups=False))

print("\nCovered rate by fx_count (0..5):")
print(multi[multi["fx_count"] <= 5].groupby("fx_count").apply(lambda g: (g["tx_type"] == "Covered").mean(), include_groups=False))

print("\nCovered rate by log10(usd_amount) deciles:")
multi["amt_q"] = pd.qcut(multi["usd_amount"], 10)
print(multi.groupby("amt_q", observed=True).apply(lambda g: (g["tx_type"] == "Covered").mean(), include_groups=False))

print("\nCovered rate by top 15 ord_bic_country:")
top_c = multi["ord_bic_country"].value_counts().head(15).index
print(multi[multi["ord_bic_country"].isin(top_c)].groupby("ord_bic_country").apply(lambda g: (g["tx_type"] == "Covered").mean(), include_groups=False).sort_values(ascending=False))

print("\nCovered rate by top 15 ben_bic_country:")
print(multi[multi["ben_bic_country"].isin(top_c)].groupby("ben_bic_country").apply(lambda g: (g["tx_type"] == "Covered").mean(), include_groups=False).sort_values(ascending=False))

print("\nCovered rate by top 15 currencies:")
top_cur = multi["currency_code"].value_counts().head(15).index
print(multi[multi["currency_code"].isin(top_cur)].groupby("currency_code").apply(lambda g: (g["tx_type"] == "Covered").mean(), include_groups=False).sort_values(ascending=False))

# Check BIC8_PATH pattern for Covered vs Indirect (e.g., repeating BICs? intermediary countries?)
print("\nSample 10 Covered rows vs 10 Indirect rows:")
print(multi[multi["tx_type"] == "Covered"][["currency_code", "usd_amount", "ordering_bic8", "beneficiary_bic8", "bic8_path", "fx_count", "participant_count", "total_duration", "continuity_qflag"]].head(10).to_string())
print(multi[multi["tx_type"] == "Indirect"][["currency_code", "usd_amount", "ordering_bic8", "beneficiary_bic8", "bic8_path", "fx_count", "participant_count", "total_duration", "continuity_qflag"]].head(10).to_string())

print(f"\nElapsed: {time.time()-t0:.1f}s")
