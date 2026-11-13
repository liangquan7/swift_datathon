import pyarrow.parquet as pq
import numpy as np
import pandas as pd

dfs = [pq.read_table(f"work/core_parquet/part_{i:02d}.parquet").to_pandas() for i in range(8)]
df = pd.concat(dfs, ignore_index=True)

d2 = df[(df["participant_count"] == 2) & (df["anomaly_type"] == "")].copy()
d2["p0"] = d2["bic8_path"].str.slice(0, 8)
d2["p1"] = d2["bic8_path"].str.slice(-8)

pair_df = d2.groupby(["p0", "p1"]).agg(
    count=("total_duration", "count"),
    d_min=("total_duration", "min"),
    d_mean=("total_duration", "mean"),
    fx=("fx_count", "first"),
).reset_index()
pair_df = pair_df[pair_df["count"] >= 10].copy()
pair_df["c0"] = pair_df["p0"].str.slice(4, 6)
pair_df["c1"] = pair_df["p1"].str.slice(4, 6)
pair_df["same_cc"] = pair_df["c0"] == pair_df["c1"]

# Check bidirectional pairs with count >= 10 in both directions
rev = pair_df.merge(pair_df, left_on=["p0", "p1"], right_on=["p1", "p0"], suffixes=("_ab", "_ba"))
rev = rev[rev["p0_ab"] < rev["p1_ab"]].copy()
rev["diff_mean"] = rev["d_mean_ab"] - rev["d_mean_ba"]
rev["abs_diff"] = rev["diff_mean"].abs()

print(f"Bidirectional pairs (n>=10 each way): {len(rev):,}")
print(f"  |diff_mean| <= 5s:  {(rev['abs_diff'] <= 5).mean()*100:.2f}%")
print(f"  |diff_mean| <= 10s: {(rev['abs_diff'] <= 10).mean()*100:.2f}%")
print(f"  |diff_mean| > 20s:  {(rev['abs_diff'] > 20).mean()*100:.2f}%")

# Among asymmetric pairs (|diff_mean| > 20s), check if diff_mean is determined by (c0, c1) country pair or bank-specific!
asym = rev[rev["abs_diff"] > 20].copy()
print("\nSample 15 asymmetric pairs (|diff_mean| > 20s):")
print(asym[["p0_ab", "p1_ab", "c0_ab", "c1_ab", "d_min_ab", "d_min_ba", "diff_mean"]].head(15).to_string(index=False))

# Check if diff_mean is constant within (c0_ab, c1_ab) country pairs!
cc_grp = rev.groupby(["c0_ab", "c1_ab"])["diff_mean"].agg(["count", "mean", "std"]).dropna()
cc_grp = cc_grp[cc_grp["count"] >= 5]
print("\nWithin-country-pair std of directional diff (median across country pairs with >=5 bank pairs):", cc_grp["std"].median())
print("Top country pairs by |mean directional diff|:")
print(cc_grp.loc[cc_grp["mean"].abs().nlargest(10).index])
