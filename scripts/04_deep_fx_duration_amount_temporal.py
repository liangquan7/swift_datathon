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
print(f"Loaded in {time.time()-t0:.1f}s")

# Exclude anomalies for baseline law analysis (keep clean copy flag)
clean = df[df["anomaly_type"] == ""].copy()

# ==================== 1. CURRENCY_CODE & FX_COUNT ====================
print("\n=== 1. CURRENCY & FX CONVERSION MECHANICS ===")
# Infer each BIC country's home currency from Direct, same-country, fx_count==0 transactions
d_same_fx0 = clean[(clean["participant_count"] == 2) & (clean["ord_bic_country"] == clean["ben_bic_country"]) & (clean["fx_count"] == 0)]
country_cur_mode = d_same_fx0.groupby("ord_bic_country")["currency_code"].agg(lambda s: s.value_counts().index[0])
print("Sample inferred country->currency (top 20 countries):", {c: country_cur_mode.get(c, "?") for c in clean["ord_bic_country"].value_counts().head(20).index})

clean["ord_home_cur"] = clean["ord_bic_country"].map(country_cur_mode).fillna("?")
clean["ben_home_cur"] = clean["ben_bic_country"].map(country_cur_mode).fillna("?")
clean["cur_is_ord_home"] = clean["currency_code"] == clean["ord_home_cur"]
clean["cur_is_ben_home"] = clean["currency_code"] == clean["ben_home_cur"]
clean["cur_is_usd"] = clean["currency_code"] == "USD"
clean["cur_is_eur"] = clean["currency_code"] == "EUR"

cross = clean[clean["ord_home_cur"] != clean["ben_home_cur"]]
print("\nCross-currency-zone payments currency choice (n =", len(cross), "):")
print("  currency == ord_home_cur:", cross["cur_is_ord_home"].mean())
print("  currency == ben_home_cur:", cross["cur_is_ben_home"].mean())
print("  currency is NEITHER:", (~cross["cur_is_ord_home"] & ~cross["cur_is_ben_home"]).mean())
neither = cross[~cross["cur_is_ord_home"] & ~cross["cur_is_ben_home"]]
print("  When NEITHER, top 10 currencies:", neither["currency_code"].value_counts(normalize=True).head(10).round(4).to_dict())

print("\nCurrency choice by (ordering_type, beneficiary_type) in cross-zone:")
print(cross.groupby(["ordering_type", "beneficiary_type"])[["cur_is_ord_home", "cur_is_ben_home"]].mean())

same_zone = clean[clean["ord_home_cur"] == clean["ben_home_cur"]]
print("\nSame-currency-zone payments (n =", len(same_zone), "):")
print("  currency == home_cur:", same_zone["cur_is_ord_home"].mean())
print("  fx_count == 0 rate:", (same_zone["fx_count"] == 0).mean())

# Check Direct (part=2) fx_count exact distribution by whether currency matches ord/ben home currency!
d2 = clean[clean["participant_count"] == 2].copy()
d2["case"] = (
    np.where(d2["cur_is_ord_home"], "O", "-")
    + np.where(d2["cur_is_ben_home"], "B", "-")
    + np.where(d2["ord_home_cur"] == d2["ben_home_cur"], "_SAME", "_DIFF")
)
print("\nDirect (part=2) fx_count distribution by currency vs home currency:")
print(pd.crosstab(d2["case"], d2["fx_count"].clip(upper=4), normalize="index").round(4))
print("Counts per case:", d2["case"].value_counts().to_dict())

# Check Indirect (part=3,4) fx_count distribution!
print("\nfx_count distribution by participant_count (2..6) for clean records:")
print(pd.crosstab(clean["participant_count"].clip(upper=6), clean["fx_count"].clip(upper=6), normalize="index").round(4))

# ==================== 2. USD_AMOUNT GENERATIVE DISTRIBUTION ====================
print("\n=== 2. USD_AMOUNT GENERATIVE DISTRIBUTION ===")
clean["log_amt"] = np.log(clean["usd_amount"])
clean["log10_amt"] = np.log10(clean["usd_amount"])
clean["party_pair"] = clean["ordering_type"] + "2" + clean["beneficiary_type"]
print("Natural log(USD_AMOUNT) stats by party_pair:")
print(clean.groupby("party_pair")["log_amt"].agg(["count", "mean", "std", "median", "skew"]))
# Check kurtosis & normality per group
for pp, g in clean.groupby("party_pair"):
    la = g["log_amt"]
    print(f"  {pp}: mu={la.mean():.4f} (exp={np.exp(la.mean()):.2f}), sigma={la.std():.4f}, kurt={la.kurtosis():.4f}")

# Check if usd_amount depends on currency_code, tx_type, same_country, participant_count!
print("\nMedian USD_AMOUNT by party_pair x tx_type:")
print(clean.groupby(["party_pair", "tx_type"])["usd_amount"].median().unstack().round(2))
print("\nMedian USD_AMOUNT by party_pair x top 8 currencies:")
top8_cur = clean["currency_code"].value_counts().head(8).index
print(clean[clean["currency_code"].isin(top8_cur)].groupby(["currency_code", "party_pair"])["usd_amount"].median().unstack().round(2))
print("\nMedian USD_AMOUNT by party_pair x top 8 ord_bic_country:")
top8_c = clean["ord_bic_country"].value_counts().head(8).index
print(clean[clean["ord_bic_country"].isin(top8_c)].groupby(["ord_bic_country", "party_pair"])["usd_amount"].median().unstack().round(2))

# Check round amounts & fractional cents!
clean["cents"] = np.round((clean["usd_amount"] % 1) * 100).astype(int)
print("\nTop 10 cent endings (.00, .50, etc.):", clean["cents"].value_counts(normalize=True).head(10).round(4).to_dict())
print("Exact integer amounts rate:", (clean["cents"] == 0).mean())
print("Exact multiples of 100 rate:", ((clean["usd_amount"] % 100) == 0).mean())
print("Top 15 most frequent exact USD_AMOUNT values:", clean["usd_amount"].value_counts().head(15).to_dict())

# ==================== 3. TOTAL_DURATION DISTRIBUTION ====================
print("\n=== 3. TOTAL_DURATION GENERATIVE DISTRIBUTION ===")
clean["log10_dur"] = np.log10(clean["total_duration"])
# Histogram of log10_dur in 0.25 bins!
bins = np.arange(0, 4.75, 0.25)
hist, edges = np.histogram(clean["log10_dur"], bins=bins)
print("Histogram of log10(TOTAL_DURATION) [overall]:")
for b0, b1, h in zip(edges[:-1], edges[1:], hist):
    print(f"  [{b0:.2f}, {b1:.2f}) ({10**b0:6.1f}s - {10**b1:7.1f}s): {h:9,d} ({h/len(clean)*100:5.2f}%)")

# Histogram for Direct (part=2) fx=0 vs fx=1!
for fxv in [0, 1]:
    sub = clean[(clean["participant_count"] == 2) & (clean["fx_count"] == fxv)]
    h2, _ = np.histogram(sub["log10_dur"], bins=bins)
    print(f"\nDirect (part=2, fx={fxv}, n={len(sub):,}) log10_dur:")
    for b0, b1, h in zip(edges[:-1], edges[1:], h2):
        if h > 0:
            print(f"  [{b0:.2f}, {b1:.2f}) ({10**b0:6.1f}s - {10**b1:7.1f}s): {h:9,d} ({h/len(sub)*100:5.2f}%)")

# Check correlation of duration with usd_amount, hour, weekday!
dt = pd.to_datetime(clean["epoch_seconds"], unit="s", utc=True)
clean["utc_hour"] = dt.dt.hour
clean["dow"] = dt.dt.dayofweek # 0=Mon..6=Sun
clean["dom"] = dt.dt.day
clean["date"] = dt.dt.date

print("\nDuration mean/median by day of week (0=Mon..6=Sun):")
print(clean.groupby("dow")["total_duration"].agg(["count", "mean", "median"]))
print("\nDuration mean/median by utc_hour:")
print(clean.groupby("utc_hour")["total_duration"].agg(["count", "mean", "median"]).head(24))
print("\nDuration mean/median by log10(usd_amount) decile:")
clean["amt_dec"] = pd.qcut(clean["usd_amount"], 10)
print(clean.groupby("amt_dec", observed=True)["total_duration"].agg(["count", "mean", "median"]))

# ==================== 4. TEMPORAL PATTERNS (EPOCH_SECONDS) ====================
print("\n=== 4. TEMPORAL PATTERNS ===")
print("UTC date min/max:", dt.min(), dt.max())
print("Count by day of week (0=Mon..6=Sun):")
print(clean["dow"].value_counts().sort_index())
print("\nCount by day of month (1..31):")
print(clean["dom"].value_counts().sort_index().to_dict())
print("\nHourly profile (UTC 0..23) overall & by top 6 ord_bic_country:")
top6_c = ["US", "CH", "GB", "FR", "DE", "SG", "CN", "JP", "AU"]
hr_tbl = pd.crosstab(clean["utc_hour"], clean.loc[clean["ord_bic_country"].isin(top6_c), "ord_bic_country"], normalize="columns") * 100
print(hr_tbl.round(2))
print("\nHourly profile (UTC 0..23) by party_pair:")
print((pd.crosstab(clean["utc_hour"], clean["party_pair"], normalize="columns") * 100).round(2))
print("\nDay-of-week profile by party_pair:")
print((pd.crosstab(clean["dow"], clean["party_pair"], normalize="columns") * 100).round(2))

print(f"\nElapsed: {time.time()-t0:.1f}s")
