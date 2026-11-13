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
]
df = pd.read_parquet("work/core_parquet", columns=cols)
print(f"Loaded in {time.time()-t0:.1f}s")

# 1. Inspect Direct duration inside specific big bank pairs (e.g. 0004IMD0 -> 0309CHZ0, 0309CHZ0 -> 0085BEB0)
d2 = df[(df["anomaly_type"] == "") & (df["participant_count"] == 2) &
        (df["bic8_path"].str.slice(0, 8) == df["ordering_bic8"]) &
        (df["bic8_path"].str.slice(-8) == df["beneficiary_bic8"])]
for ob, bb in [("0004IMD0", "0309CHZ0"), ("0309CHZ0", "0004IMD0"), ("0309CHZ0", "0085BEB0"), ("0412FRP0", "0085BEB0")]:
    s = d2[(d2["ordering_bic8"] == ob) & (d2["beneficiary_bic8"] == bb)]["total_duration"]
    print(f"Pair {ob}->{bb} (n={len(s)}): min={s.min()}, p10={s.quantile(0.1):.1f}, med={s.median():.1f}, p90={s.quantile(0.9):.1f}, max={s.max()}, std={s.std():.1f}")
    # Print a few values
    print("  sample values:", sorted(s.head(20).tolist()))

# Also check if a bank has a bimodal distribution or if std is ~7s vs ~200s for 0004IMD0!
# Check distribution of std across top 50 Direct pairs with n >= 100!
pair_stats = d2.groupby(["ordering_bic8", "beneficiary_bic8"])["total_duration"].agg(["count", "mean", "std", "min", "max"])
p100 = pair_stats[pair_stats["count"] >= 100]
print("Direct pairs with n>=100 (count =", len(p100), "), std quantiles:", p100["std"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]).round(2).to_dict())
print("Top 5 highest-std pairs with n>=100:")
print(p100.sort_values("std", ascending=False).head(5))

# 2. Check ACCOUNT_REPLACEMENT_SCAM beneficiary details vs ord_acct's normal beneficiary details!
def get_details_map(row_ids):
    row_ids = sorted(set(int(r) for r in row_ids))
    by_part = {}
    for r in row_ids:
        by_part.setdefault(r // 250_000, []).append(r)
    dfs = []
    for p, rids in by_part.items():
        part = pd.read_parquet(f"work/details_parquet/part_{p:02d}.parquet", columns=["row_id", "BENEFICIARY_DETAILS"])
        dfs.append(part[part["row_id"].isin(rids)])
    res = pd.concat(dfs, ignore_index=True)
    return dict(zip(res["row_id"], res["BENEFICIARY_DETAILS"]))

ars = df[df["anomaly_type"] == "ACCOUNT_REPLACEMENT_SCAM"].head(15)
ars_det_map = get_details_map(ars["row_id"].tolist())
ars_ords = set(ars["ordering_account"])
norm_for_ars_ords = df[(df["anomaly_type"] == "") & (df["ordering_account"].isin(ars_ords))]
print("\n=== 2. ACCOUNT_REPLACEMENT_SCAM NAME INTERCEPTION CHECK ===")
for _, r in ars.head(8).iterrows():
    bdet = ars_det_map[int(r["row_id"])]
    lines = bdet.split("\n")
    name_part = lines[1][:15] if len(lines) > 1 else lines[0][:15]
    # Check if ANY normal beneficiary of this ordering_account shares that name snippet or country!
    sub_n = norm_for_ars_ords[norm_for_ars_ords["ordering_account"] == r["ordering_account"]]
    print(f"  ARS row={r['row_id']} ord={r['ordering_account']} ben={r['beneficiary_account']} ben_cc={r['ben_inferred_cc']} | line1={repr(lines[1] if len(lines)>1 else lines[0])}")
    # Fetch details for up to 50 normal tx of this ord
    if len(sub_n) > 0:
        n_det = get_details_map(sub_n["row_id"].head(50).tolist())
        hits = [v for v in n_det.values() if name_part and name_part in v]
        print(f"    normal tx of ord={len(sub_n)}, matches on '{name_part}': {len(hits)}")

# 3. Check PARTICIPANT_COUNT exact geometric law & total unique BIC8s across path!
print("\n=== 3. PARTICIPANT_COUNT GEOMETRIC FIT ===")
p_cnt = df["participant_count"].value_counts().sort_index()
ratios = {k: round(p_cnt.get(k, 0) / p_cnt.get(k - 1, 1), 5) for k in range(3, 12)}
print("Sequential ratios P(N=k) / P(N=k-1):", ratios)
print("Note 1/e =", round(np.exp(-1), 5))
# Check if Direct vs Multi differs by ordering_type, beneficiary_type, ord_bic_country!
df["is_multi"] = df["participant_count"] >= 3
print("Multi-participant rate by party_pair:", df.groupby([df["ordering_type"], df["beneficiary_type"]])["is_multi"].mean().round(5).to_dict())
print("Multi-participant rate by domestic (ord_cc==ben_cc):", df.groupby(df["ord_bic_country"] == df["ben_bic_country"])["is_multi"].mean().round(5).to_dict())

# 4. Check CUSTOMER_RECALL text words (<other_word>) vs normal!
cr = df[df["anomaly_type"] == "CUSTOMER_RECALL"]
cr_det_all = pd.concat([
    pd.read_parquet(f"work/details_parquet/part_{p:02d}.parquet")
    for p in sorted(set(cr["row_id"] // 250_000))
])
cr_det = cr_det_all[cr_det_all["row_id"].isin(set(cr["row_id"]))]
# Check last line of details or uppercase words in ORDERING_DETAILS / BENEFICIARY_DETAILS
print("\n=== 4. CUSTOMER_RECALL EXTRA TEXT CHECK ===")
print("Sample last lines of CR BENEFICIARY_DETAILS:", [s.split("\n")[-1] for s in cr_det["BENEFICIARY_DETAILS"].head(12)])
print("Sample last lines of CR ORDERING_DETAILS:", [s.split("\n")[-1] for s in cr_det["ORDERING_DETAILS"].head(12)])

# 5. Check total unique BIC8 across all BIC8_PATHs!
all_bics = set(df["ordering_bic8"].unique()) | set(df["beneficiary_bic8"].unique())
print("\nTotal unique endpoint BIC8s:", len(all_bics))
p3_bics = set(df.loc[df["participant_count"] == 3, "bic8_path"].str.split(";").str[1].unique()) - {"..."}
print("Unique intermediary BIC8s (part=3):", len(p3_bics), "new BIC8s not in endpoints:", len(p3_bics - all_bics))

print(f"\nElapsed: {time.time()-t0:.1f}s")
