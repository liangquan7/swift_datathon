import json
import os
import re
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

os.makedirs("work/chart_data", exist_ok=True)
t0 = time.time()

CORE_PARTS = [f"work/core_parquet/part_{i:02d}.parquet" for i in range(40)]
schema_cols = pq.read_schema(CORE_PARTS[0]).names
print("Core schema cols:", schema_cols)

needed_cols = [
    "row_id", "epoch_seconds", "currency_code", "usd_amount",
    "ordering_bic8", "beneficiary_bic8", "bic8_path",
    "ord_bic_country", "ben_bic_country", "ord_bic_bank", "ben_bic_bank",
    "path_has_gap", "path_visible_bics", "total_duration",
    "continuity_qflag", "fx_count", "participant_count", "tx_type",
    "ordering_account", "beneficiary_account", "ordering_type", "beneficiary_type",
    "anomaly_type", "ord_inferred_cc", "ben_inferred_cc",
    "ord_has_iban", "ben_has_iban", "ord_has_lb35", "ben_has_lb35",
    "ord_has_trunc", "ben_has_trunc", "ord_is_struct_123", "ben_is_struct_123",
    "ord_fin_word", "ben_fin_word", "ord_det_len", "ben_det_len",
    "ord_has_acct_line", "ben_has_acct_line"
]
use_cols = [c for c in needed_cols if c in schema_cols]

print(f"Loading 40 core parts ({len(use_cols)} cols)...")
dfs = []
for p in CORE_PARTS:
    df_p = pd.read_parquet(p, columns=use_cols)
    for col in ["currency_code", "ord_bic_country", "ben_bic_country", "continuity_qflag",
                "tx_type", "ordering_type", "beneficiary_type", "anomaly_type",
                "ord_inferred_cc", "ben_inferred_cc", "ord_fin_word", "ben_fin_word"]:
        if col in df_p.columns:
            df_p[col] = df_p[col].astype("string")
    dfs.append(df_p)
df = pd.concat(dfs, ignore_index=True)
del dfs
print(f"Loaded {len(df):,} rows in {time.time() - t0:.1f}s")

master = {"row_count": int(len(df))}

# Derive path endpoints and dot counts across all 10M rows cleanly
print("Extracting path endpoints and dot counts...")
first_bic = df["bic8_path"].str.slice(0, 8)
last_bic = df["bic8_path"].str.slice(-8)
lead_dots = (df["bic8_path"].str.startswith("...")).sum()
trail_dots = (df["bic8_path"].str.endswith("...")).sum()
print(f"lead_dots={lead_dots}, trail_dots={trail_dots}")
n_dots = df["bic8_path"].str.count(r"\.\.\.")
start_mis = (first_bic != df["ordering_bic8"])
end_mis = (last_bic != df["beneficiary_bic8"])
is_clean = (df["anomaly_type"] == "") | df["anomaly_type"].isna()
is_qflag_n = df["continuity_qflag"] == "N"
is_qflag_y = df["continuity_qflag"] == "Y"
is_direct = df["tx_type"] == "Direct"
is_indirect = df["tx_type"] == "Indirect"
is_covered = df["tx_type"] == "Covered"

non_dot_bics = (df["bic8_path"].str.count(";") + 1) - n_dots
non_dot_mis = int((non_dot_bics != df["participant_count"]).sum())
print(f"non_dot_bics != participant_count count: {non_dot_mis}")

master["path_checks"] = {
    "lead_dots": int(lead_dots),
    "trail_dots": int(trail_dots),
    "qflag_n_count": int(is_qflag_n.sum()),
    "path_has_gap_count": int((n_dots > 0).sum()),
    "qflag_n_vs_dots_mismatch": int((is_qflag_n != (n_dots > 0)).sum()),
    "non_dot_bics_vs_participant_mismatch": non_dot_mis,
    "start_mis_count": int(start_mis.sum()),
    "start_mis_rate": float(start_mis.mean()),
    "end_mis_count": int(end_mis.sum()),
    "end_mis_rate": float(end_mis.mean()),
    "both_mis_count": int((start_mis & end_mis).sum()),
    "both_mis_rate": float((start_mis & end_mis).mean()),
    "expected_independent_both_rate": float(start_mis.mean() * end_mis.mean()),
}
print("Path checks:", master["path_checks"])

# ------------------------------------------------------------------
# 0. DATA PREP SENSITIVITY TABLE
# ------------------------------------------------------------------
def subset_metrics(mask, label):
    sub = df.loc[mask]
    n = int(len(sub))
    return {
        "subset": label,
        "n": n,
        "pct_of_raw": round(100.0 * n / len(df), 4),
        "usd_mean": round(float(sub["usd_amount"].mean()), 2),
        "usd_median": round(float(sub["usd_amount"].median()), 2),
        "usd_p95": round(float(sub["usd_amount"].quantile(0.95)), 2),
        "usd_max": round(float(sub["usd_amount"].max()), 2),
        "dur_mean": round(float(sub["total_duration"].mean()), 2),
        "dur_median": round(float(sub["total_duration"].median()), 1),
        "direct_pct": round(100.0 * float((sub["tx_type"] == "Direct").mean()), 4),
        "covered_among_multihop_pct": round(
            100.0 * float((sub["tx_type"] == "Covered").sum() / max(1, (sub["participant_count"] >= 3).sum())), 4
        ),
        "fx_pos_pct": round(100.0 * float((sub["fx_count"] > 0).mean()), 4),
        "qflag_n_pct": round(100.0 * float((sub["continuity_qflag"] == "N").mean()), 4),
    }

sensitivity = [
    subset_metrics(np.ones(len(df), dtype=bool), "Raw All Rows"),
    subset_metrics(is_clean, "Exclude Labeled Anomalies"),
    subset_metrics(is_clean & (~start_mis) & (~end_mis), "Clean + Endpoint Match"),
    subset_metrics(is_clean & is_qflag_y, "Clean + QFLAG='Y'"),
    subset_metrics(is_clean & (~start_mis) & (~end_mis) & is_qflag_y, "Clean + Endpoint Match + QFLAG='Y'"),
]
master["data_prep_sensitivity"] = sensitivity
print("Sensitivity:", sensitivity)

# ------------------------------------------------------------------
# 1. ENTITY GRAPH & PARTY-TYPE STRUCTURE
# ------------------------------------------------------------------
pt_counts = df.groupby(["ordering_type", "beneficiary_type"], observed=True).size().to_dict()
ord_accts = df["ordering_account"].nunique()
ben_accts = df["beneficiary_account"].nunique()
ord_bics = df["ordering_bic8"].nunique()
ben_bics = df["beneficiary_bic8"].nunique()
all_endpoint_bics = len(set(df["ordering_bic8"].unique()) | set(df["beneficiary_bic8"].unique()))

ben_bic_per_acct = df.groupby("beneficiary_account", observed=True)["beneficiary_bic8"].nunique()
ben_bic_dist = ben_bic_per_acct.value_counts().sort_index().to_dict()

ord_acct_meta = df.drop_duplicates("ordering_account")[["ordering_account", "ordering_bic8", "ordering_type", "ord_bic_country"]]
ben_acct_meta = df.drop_duplicates("beneficiary_account")[["beneficiary_account", "beneficiary_bic8", "beneficiary_type", "ben_bic_country"]]
dual = ord_acct_meta.merge(ben_acct_meta, left_on="ordering_account", right_on="beneficiary_account", how="inner")

master["entity_graph"] = {
    "party_pairs": {f"{k[0]}->{k[1]}": int(v) for k, v in pt_counts.items()},
    "unique_ordering_accounts": int(ord_accts),
    "unique_beneficiary_accounts": int(ben_accts),
    "unique_ordering_bic8": int(ord_bics),
    "unique_beneficiary_bic8": int(ben_bics),
    "unique_endpoint_bic8": int(all_endpoint_bics),
    "ord_countries": int(df["ord_bic_country"].nunique()),
    "ben_countries": int(df["ben_bic_country"].nunique()),
    "ben_bic_per_acct_dist": {int(k): int(v) for k, v in ben_bic_dist.items()},
    "ben_acct_1_bic_pct": round(100.0 * float((ben_bic_per_acct == 1).mean()), 4),
    "dual_role_accounts": int(len(dual)),
    "dual_same_bic_pct": round(100.0 * float((dual["ordering_bic8"] == dual["beneficiary_bic8"]).mean()), 4),
    "dual_same_type_pct": round(100.0 * float((dual["ordering_type"] == dual["beneficiary_type"]).mean()), 4),
    "dual_same_country_pct": round(100.0 * float((dual["ord_bic_country"] == dual["ben_bic_country"]).mean()), 4),
}
del ord_acct_meta, ben_acct_meta, dual, ben_bic_per_acct
print("Entity graph done.")

# ------------------------------------------------------------------
# 2. TEMPORAL DYNAMICS & TIMEZONE ALIGNMENT
# ------------------------------------------------------------------
min_ep = int(df["epoch_seconds"].min())
max_ep = int(df["epoch_seconds"].max())
ts = pd.to_datetime(df["epoch_seconds"], unit="s", utc=True)
utc_hour = ts.dt.hour.to_numpy()
utc_dow = ts.dt.dayofweek.to_numpy()  # 0=Mon..6=Sun
date_str = ts.dt.strftime("%Y-%m-%d")

df_tmp = pd.DataFrame({
    "date": date_str,
    "is_b2b": (df["ordering_type"] == "B") & (df["beneficiary_type"] == "B"),
    "is_direct": is_direct,
    "amt": df["usd_amount"],
    "clean": is_clean
})
daily_fast = df_tmp.groupby("date", observed=True).agg(
    tx_count=("is_b2b", "count"),
    b2b_sum=("is_b2b", "sum"),
    direct_sum=("is_direct", "sum")
).reset_index()
daily_med = df_tmp[df_tmp["clean"]].groupby("date", observed=True)["amt"].median().reset_index().rename(columns={"amt": "clean_med_amt"})
daily_full = daily_fast.merge(daily_med, on="date")
daily_full["b2b_pct"] = (100.0 * daily_full["b2b_sum"] / daily_full["tx_count"]).round(3)
daily_full["direct_pct"] = (100.0 * daily_full["direct_sum"] / daily_full["tx_count"]).round(3)
daily_full["dow"] = pd.to_datetime(daily_full["date"]).dt.day_name().str.slice(0, 3)
daily_full.to_csv("work/chart_data/daily_series.csv", index=False)
del df_tmp, daily_fast, daily_med

tz_countries = ["US", "CA", "GB", "DE", "FR", "CH", "NL", "IL", "EG", "SA", "AE", "IN", "CN", "SG", "JP", "AU"]
country_hours = {}
ref_gb = np.bincount(utc_hour[df["ord_bic_country"] == "GB"], minlength=24).astype(float)
ref_gb /= ref_gb.sum()
tz_offsets = {}
for cc in tz_countries:
    m = (df["ord_bic_country"] == cc).to_numpy()
    h_vec = np.bincount(utc_hour[m], minlength=24).astype(float)
    h_pct = 100.0 * h_vec / h_vec.sum()
    country_hours[cc] = h_pct.round(4).tolist()
    best_r, best_s = -1.0, 0
    for s in range(-12, 13):
        shifted = np.roll(h_pct, s)
        r = float(np.corrcoef(shifted, ref_gb)[0, 1])
        if r > best_r:
            best_r = r
            best_s = s
    tz_offsets[cc] = {"shift_rel_gb": int(best_s), "corr_with_gb": round(best_r, 5)}

gb_curve = np.array(country_hours["GB"])
print("GB UTC hourly %:", np.round(gb_curve, 2))
print("GB min hour UTC:", int(np.argmin(gb_curve)), "peaks UTC:", np.argsort(gb_curve)[-4:])

gb_utc_offset = 1  # Summer BST UTC+1
tz_summary = []
for cc in tz_countries:
    m = (df["ord_bic_country"] == cc).to_numpy()
    off = gb_utc_offset + tz_offsets[cc]["shift_rel_gb"]
    local_ep = df["epoch_seconds"].to_numpy()[m] + off * 3600
    local_dt = pd.to_datetime(local_ep, unit="s", utc=True)
    lh = local_dt.hour.to_numpy()
    ldow = local_dt.dayofweek.to_numpy()
    udow = utc_dow[m]
    tz_summary.append({
        "country": cc,
        "n": int(m.sum()),
        "utc_offset_best": int(off),
        "corr_with_gb": tz_offsets[cc]["corr_with_gb"],
        "utc_dow_pct": (100.0 * np.bincount(udow, minlength=7) / m.sum()).round(3).tolist(),
        "local_dow_pct": (100.0 * np.bincount(ldow, minlength=7) / m.sum()).round(3).tolist(),
        "local_hour_pct": (100.0 * np.bincount(lh, minlength=24) / m.sum()).round(3).tolist(),
        "utc_hour_pct": country_hours[cc],
    })

master["temporal"] = {
    "min_epoch": min_ep,
    "max_epoch": max_ep,
    "min_utc": str(pd.to_datetime(min_ep, unit="s", utc=True)),
    "max_utc": str(pd.to_datetime(max_ep, unit="s", utc=True)),
    "global_utc_dow_pct": (100.0 * np.bincount(utc_dow, minlength=7) / len(df)).round(4).tolist(),
    "tz_countries": tz_summary,
}
pd.DataFrame(tz_summary).to_json("work/chart_data/tz_profiles.json", orient="records", indent=2)
print("Temporal done.")

# ------------------------------------------------------------------
# 3. PAYMENT VALUE (USD_AMOUNT) LAWS
# ------------------------------------------------------------------
clean_df = df.loc[is_clean, ["usd_amount", "currency_code", "ordering_type", "beneficiary_type", "ord_bic_country", "ben_bic_country", "ordering_bic8", "epoch_seconds"]].copy()
clean_df["ln_amt"] = np.log(clean_df["usd_amount"].to_numpy())
clean_df["party_pair"] = clean_df["ordering_type"].astype(str) + "->" + clean_df["beneficiary_type"].astype(str)

seg_stats = {}
for seg, grp in clean_df.groupby("party_pair", observed=True):
    x = grp["ln_amt"].to_numpy()
    seg_stats[seg] = {
        "n": int(len(x)),
        "ln_mean": round(float(x.mean()), 4),
        "ln_std": round(float(x.std()), 4),
        "geom_mean": round(float(np.exp(x.mean())), 2),
        "median": round(float(np.median(grp["usd_amount"])), 2),
        "mean": round(float(grp["usd_amount"].mean()), 2),
        "p95": round(float(np.quantile(grp["usd_amount"], 0.95)), 2),
    }

def r2_grouped_ols(y, cols_list):
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if len(cols_list) == 1:
        g_mean = pd.Series(y - y.mean()).groupby(cols_list[0], observed=True).transform("mean").to_numpy()
        resid = (y - y.mean()) - g_mean
        return 1.0 - float(np.sum(resid ** 2)) / ss_tot
    tmp = pd.DataFrame({f"c{idx}": c for idx, c in enumerate(cols_list)})
    r = y.copy() - y.mean()
    effects = [np.zeros(len(y)) for _ in cols_list]
    for _ in range(12):
        for idx in range(len(cols_list)):
            target = r + effects[idx]
            new_eff = pd.Series(target).groupby(tmp[f"c{idx}"], observed=True).transform("mean").to_numpy()
            effects[idx] = new_eff
            r = target - new_eff
    return 1.0 - float(np.sum(r ** 2)) / ss_tot

y_amt = clean_df["ln_amt"].to_numpy()
r2_party = r2_grouped_ols(y_amt, [clean_df["party_pair"].to_numpy()])
r2_party_ord_cc = r2_grouped_ols(y_amt, [clean_df["party_pair"].to_numpy(), clean_df["ord_bic_country"].to_numpy()])
r2_party_ord_cc_curr = r2_grouped_ols(y_amt, [clean_df["party_pair"].to_numpy(), clean_df["ord_bic_country"].to_numpy(), clean_df["currency_code"].to_numpy()])
r2_party_ord_cc_ben_cc = r2_grouped_ols(y_amt, [clean_df["party_pair"].to_numpy(), clean_df["ord_bic_country"].to_numpy(), clean_df["ben_bic_country"].to_numpy()])
r2_party_ord_bic = r2_grouped_ols(y_amt, [clean_df["party_pair"].to_numpy(), clean_df["ordering_bic8"].to_numpy()])

p_means = clean_df.groupby("party_pair", observed=True)["ln_amt"].transform("mean")
clean_df["ln_amt_party_adj"] = clean_df["ln_amt"] - p_means
cc_mult = clean_df.groupby("ord_bic_country", observed=True).agg(
    n=("ln_amt", "count"),
    adj_ln_mean=("ln_amt_party_adj", "mean"),
    raw_med=("usd_amount", "median"),
).reset_index()
cc_mult["mult_vs_global"] = np.exp(cc_mult["adj_ln_mean"])
cc_mult_filtered = cc_mult[cc_mult["n"] >= 500].sort_values("mult_vs_global", ascending=False)
cc_mult_filtered.to_csv("work/chart_data/country_amount_multipliers.csv", index=False)

# Segment histogram of ln_amt
ln_edges = np.linspace(-5, 22, 81)
seg_hist = {"bin_left": np.round(ln_edges[:-1], 4), "bin_right": np.round(ln_edges[1:], 4)}
for seg in ["C->C", "B->C", "C->B", "B->B"]:
    sub_ln = clean_df.loc[clean_df["party_pair"] == seg, "ln_amt"].to_numpy()
    h_seg, _ = np.histogram(sub_ln, bins=ln_edges, density=True)
    seg_hist[seg.replace("->", "")] = np.round(h_seg, 6)
pd.DataFrame(seg_hist).to_csv("work/chart_data/ln_amt_segment_hist.csv", index=False)

top10_curr = clean_df["currency_code"].value_counts().head(10).index.tolist()
curr_round_rows = []
for cur in top10_curr:
    sub_a = clean_df.loc[clean_df["currency_code"] == cur, "usd_amount"].to_numpy()
    is_int = np.abs(sub_a - np.round(sub_a)) < 1e-6
    is_m100 = is_int & ((np.round(sub_a).astype(np.int64) % 100) == 0) & (sub_a >= 100)
    curr_round_rows.append({
        "currency": str(cur),
        "n": int(len(sub_a)),
        "exact_int_pct": round(100.0 * float(is_int.mean()), 4),
        "mult100_pct": round(100.0 * float(is_m100.mean()), 4),
    })

eur_sub = clean_df[clean_df["currency_code"] == "EUR"].copy()
eur_sub["date"] = pd.to_datetime(eur_sub["epoch_seconds"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
eur_1k = eur_sub[(eur_sub["usd_amount"] >= 1050) & (eur_sub["usd_amount"] <= 1250)]
eur_daily_mode = []
for dt_val, g in eur_1k.groupby("date"):
    vc = g["usd_amount"].value_counts()
    if len(vc) > 0 and vc.iloc[0] >= 5:
        eur_daily_mode.append({"date": dt_val, "usd_for_1000eur": float(vc.index[0]), "count": int(vc.iloc[0])})
pd.DataFrame(eur_daily_mode).to_csv("work/chart_data/eur_daily_fx_1000.csv", index=False)

master["payment_value"] = {
    "seg_stats": seg_stats,
    "r2_party": round(r2_party, 5),
    "r2_party_ord_cc": round(r2_party_ord_cc, 5),
    "r2_party_ord_cc_curr": round(r2_party_ord_cc_curr, 5),
    "r2_party_ord_cc_ben_cc": round(r2_party_ord_cc_ben_cc, 5),
    "r2_party_ord_bic": round(r2_party_ord_bic, 5),
    "curr_round_rates": curr_round_rows,
    "eur_1000_days_found": len(eur_daily_mode),
}
print("Payment value done. R2:", master["payment_value"]["r2_party"], master["payment_value"]["r2_party_ord_cc"], master["payment_value"]["r2_party_ord_cc_curr"], master["payment_value"]["r2_party_ord_cc_ben_cc"], master["payment_value"]["r2_party_ord_bic"])
del clean_df, eur_sub

# ------------------------------------------------------------------
# 4. CURRENCY CHOICE & EXACT 10,000,000-ROW FX TRANSITION LAW
# ------------------------------------------------------------------
cc_curr_mode = df.groupby("ord_bic_country", observed=True)["currency_code"].agg(lambda s: s.value_counts().index[0]).to_dict()
ben_cc_mode = df.groupby("ben_bic_country", observed=True)["currency_code"].agg(lambda s: s.value_counts().index[0]).to_dict()
for k, v in ben_cc_mode.items():
    if k not in cc_curr_mode:
        cc_curr_mode[k] = v

ord_home = df["ord_bic_country"].map(cc_curr_mode)
ben_home = df["ben_bic_country"].map(cc_curr_mode)
same_zone = ord_home == ben_home
m_ord = df["currency_code"] == ord_home
m_ben = df["currency_code"] == ben_home
cross_df = df[~same_zone]
third_top = cross_df.loc[~(m_ord[~same_zone] | m_ben[~same_zone]), "currency_code"].value_counts().head(10).to_dict()

# Inspect path separator first
sample_p = str(df.loc[df["participant_count"] >= 3, "bic8_path"].iloc[0])
print("Sample multi-hop bic8_path:", sample_p)
sep = ";" if ";" in sample_p else ("->" if "->" in sample_p else ("-" if "-" in sample_p else ","))
print("Detected path separator:", repr(sep))

print("Verifying exact FX transition rule on all 10,000,000 rows...")
total_fx_match = 0
fx_mismatches_examples = []
for i in range(40):
    sl = slice(i * 250000, (i + 1) * 250000)
    paths = df["bic8_path"].iloc[sl].tolist()
    fxs = df["fx_count"].iloc[sl].to_numpy()
    rows_id = df["row_id"].iloc[sl].to_numpy()
    calc = np.empty(len(paths), dtype=np.int16)
    for j, p_str in enumerate(paths):
        bics = [tok for tok in p_str.split(sep) if tok != "..."]
        cnt = 0
        prev_c = cc_curr_mode.get(bics[0][4:6], bics[0][4:6])
        for b in bics[1:]:
            cur_c = cc_curr_mode.get(b[4:6], b[4:6])
            if cur_c != prev_c:
                cnt += 1
            prev_c = cur_c
        calc[j] = cnt
    eq = (calc == fxs)
    total_fx_match += int(eq.sum())
    if not eq.all() and len(fx_mismatches_examples) < 10:
        bad = np.where(~eq)[0][:5]
        for b_idx in bad:
            fx_mismatches_examples.append({
                "row_id": int(rows_id[b_idx]),
                "path": paths[b_idx],
                "currency": str(df["currency_code"].iloc[sl].iloc[b_idx]),
                "obs_fx": int(fxs[b_idx]),
                "calc_fx": int(calc[b_idx]),
            })

print(f"FX match across ALL 10M rows: {total_fx_match:,} / {len(df):,} ({100.0*total_fx_match/len(df):.6f}%)")
if fx_mismatches_examples:
    print("FX mismatch samples:", fx_mismatches_examples[:5])
    print("ER ord currency dist:", df.loc[df["ord_bic_country"] == "ER", "currency_code"].value_counts().to_dict())
    print("ER ben currency dist:", df.loc[df["ben_bic_country"] == "ER", "currency_code"].value_counts().to_dict())

master["fx_and_currency"] = {
    "path_separator": sep,
    "same_zone_count": int(same_zone.sum()),
    "same_zone_shared_curr_pct": round(100.0 * float(m_ord[same_zone].mean()), 4),
    "cross_zone_count": int((~same_zone).sum()),
    "cross_zone_ord_home_pct": round(100.0 * float(m_ord[~same_zone].mean()), 4),
    "cross_zone_ben_home_pct": round(100.0 * float(m_ben[~same_zone].mean()), 4),
    "cross_zone_third_curr_pct": round(100.0 * float((~(m_ord[~same_zone] | m_ben[~same_zone])).mean()), 4),
    "top_currencies": {str(k): int(v) for k, v in df["currency_code"].value_counts().head(12).items()},
    "top_third_currencies": {str(k): int(v) for k, v in third_top.items()},
    "fx_rule_exact_matches_10m": int(total_fx_match),
    "fx_rule_accuracy_pct": round(100.0 * total_fx_match / len(df), 6),
    "fx_mismatch_examples": fx_mismatches_examples,
}
del ord_home, ben_home, same_zone, m_ord, m_ben, cross_df

# ------------------------------------------------------------------
# 5. ROUTING TOPOLOGY & COVERED BANK MECHANICS
# ------------------------------------------------------------------
pc_vc = df["participant_count"].value_counts().sort_index()
pc_rows = []
for k, cnt in pc_vc.items():
    sub = df[df["participant_count"] == k]
    pc_rows.append({
        "participant_count": int(k),
        "count": int(cnt),
        "pct": round(100.0 * cnt / len(df), 5),
        "direct": int((sub["tx_type"] == "Direct").sum()),
        "indirect": int((sub["tx_type"] == "Indirect").sum()),
        "covered": int((sub["tx_type"] == "Covered").sum()),
        "qflag_n_pct": round(100.0 * float((sub["continuity_qflag"] == "N").mean()), 4),
        "expected_qflag_n_pct": round(100.0 * (1.0 - (0.95 ** (int(k) - 1))), 4),
    })

mh = df.loc[df["participant_count"] >= 3, ["ordering_bic8", "beneficiary_bic8", "ord_bic_country", "tx_type", "ordering_type", "beneficiary_type"]].copy()
mh["is_cov"] = (mh["tx_type"] == "Covered").astype(np.float32)
ord_bank_cov = mh.groupby("ordering_bic8", observed=True).agg(n=("is_cov", "count"), cov_rate=("is_cov", "mean")).reset_index()
ben_bank_cov = mh.groupby("beneficiary_bic8", observed=True).agg(n=("is_cov", "count"), cov_rate=("is_cov", "mean")).reset_index()
ord_bank_cov_200 = ord_bank_cov[ord_bank_cov["n"] >= 200]
ben_bank_cov_200 = ben_bank_cov[ben_bank_cov["n"] >= 200]
ord_bank_cov_200.to_csv("work/chart_data/ord_bank_covered_rates.csv", index=False)

r2_cov_ord_cc = r2_grouped_ols(mh["is_cov"].to_numpy(), [mh["ord_bic_country"].to_numpy()])
r2_cov_ord_bic = r2_grouped_ols(mh["is_cov"].to_numpy(), [mh["ordering_bic8"].to_numpy()])
r2_cov_ben_bic = r2_grouped_ols(mh["is_cov"].to_numpy(), [mh["beneficiary_bic8"].to_numpy()])

dots_fit = []
for n_val in [2, 3, 4, 5]:
    sub_d = n_dots[df["participant_count"] == n_val].to_numpy()
    r = n_val - 1
    p = 0.95
    for d in range(5):
        obs_p = float((sub_d == d).mean())
        exp_p = float(stats.nbinom.pmf(d, r, p))
        dots_fit.append({
            "participant_count": n_val, "n_dots": d,
            "obs_pct": round(100.0 * obs_p, 4),
            "theo_pct": round(100.0 * exp_p, 4),
            "count": int((sub_d == d).sum()),
        })

p3_idx = (df["participant_count"] == 3) & is_qflag_y
mid_paths = df.loc[p3_idx, "bic8_path"].to_numpy()
mid_splits = [str(s).split(sep) for s in mid_paths]
n_p3_non3 = sum(1 for sp in mid_splits if len(sp) != 3)
print(f"p3 QFLAG=Y non-3 token count: {n_p3_non3} / {len(mid_splits)}")
mid_valid = np.array([len(sp) == 3 for sp in mid_splits])
mid_cc = pd.Series([sp[1][4:6] for sp, ok in zip(mid_splits, mid_valid) if ok])
mid_curr = df.loc[p3_idx, "currency_code"].reset_index(drop=True)[mid_valid].reset_index(drop=True)
mid_global = (100.0 * mid_cc.value_counts().head(10) / len(mid_cc)).round(3).to_dict()
mid_by_curr = {}
for cur in ["USD", "EUR", "CHF", "GBP", "CNY", "JPY"]:
    m_c = (mid_curr == cur).to_numpy()
    vc_c = (100.0 * mid_cc[m_c].value_counts().head(8) / m_c.sum()).round(3).to_dict()
    mid_by_curr[cur] = vc_c

master["routing"] = {
    "participant_dist": pc_rows,
    "multihop_covered_overall_pct": round(100.0 * float(mh["is_cov"].mean()), 4),
    "cov_r2_ord_country": round(r2_cov_ord_cc, 5),
    "cov_r2_ord_bic8": round(r2_cov_ord_bic, 5),
    "cov_r2_ben_bic8": round(r2_cov_ben_bic, 5),
    "ord_bank_cov_p10_p50_p90_p99": np.round(100.0 * np.quantile(ord_bank_cov_200["cov_rate"], [0.1, 0.5, 0.9, 0.99]), 2).tolist(),
    "ben_bank_cov_p10_p50_p90_p99": np.round(100.0 * np.quantile(ben_bank_cov_200["cov_rate"], [0.1, 0.5, 0.9, 0.99]), 2).tolist(),
    "dots_fit": dots_fit,
    "mid_global_top10_pct": mid_global,
    "mid_by_curr_pct": mid_by_curr,
}

dur_qflag = []
for (tt, pc, qf), g in df[df["participant_count"] <= 6].groupby(["tx_type", "participant_count", "continuity_qflag"], observed=True):
    dur_qflag.append({
        "tx_type": str(tt), "participant_count": int(pc), "qflag": str(qf),
        "n": int(len(g)), "mean_dur": round(float(g["total_duration"].mean()), 2),
        "med_dur": round(float(g["total_duration"].median()), 1),
    })
master["qflag_duration_confounder"] = dur_qflag
del mh, mid_cc, mid_curr

# ------------------------------------------------------------------
# 6. SETTLEMENT DURATION (TOTAL_DURATION) ARCHITECTURE
# ------------------------------------------------------------------
print("Cracking Direct & Multi-hop TOTAL_DURATION formula...")
dir_mask = is_clean & is_direct & (~start_mis) & (~end_mis)
dir_df = df.loc[dir_mask, ["ordering_bic8", "beneficiary_bic8", "ord_bic_country", "ben_bic_country", "total_duration", "epoch_seconds"]].copy()

pair_stats = dir_df.groupby(["ordering_bic8", "beneficiary_bic8"], observed=True)["total_duration"].agg(
    n="count", mean="mean", median="median", std="std", min="min"
).reset_index()

pair_ge5 = pair_stats[pair_stats["n"] >= 5]
pair_ge20 = pair_stats[pair_stats["n"] >= 20]
pair_ge100 = pair_stats[pair_stats["n"] >= 100]

dir_df["pair_mean"] = dir_df.groupby(["ordering_bic8", "beneficiary_bic8"], observed=True)["total_duration"].transform("mean")
dir_df["pair_n"] = dir_df.groupby(["ordering_bic8", "beneficiary_bic8"], observed=True)["total_duration"].transform("count")
sub5 = dir_df[dir_df["pair_n"] >= 5]
r2_pair_ge5 = 1.0 - float(((sub5["total_duration"] - sub5["pair_mean"]) ** 2).sum()) / float(((sub5["total_duration"] - sub5["total_duration"].mean()) ** 2).sum())
rmse_pair_ge5 = float(np.sqrt(((sub5["total_duration"] - sub5["pair_mean"]) ** 2).mean()))

sub20 = dir_df[dir_df["pair_n"] >= 20]
r2_pair_ge20 = 1.0 - float(((sub20["total_duration"] - sub20["pair_mean"]) ** 2).sum()) / float(((sub20["total_duration"] - sub20["total_duration"].mean()) ** 2).sum())
rmse_pair_ge20 = float(np.sqrt(((sub20["total_duration"] - sub20["pair_mean"]) ** 2).mean()))

rev_stats = pair_ge20.rename(columns={"ordering_bic8": "beneficiary_bic8", "beneficiary_bic8": "ordering_bic8", "mean": "rev_mean", "n": "rev_n"})[["ordering_bic8", "beneficiary_bic8", "rev_mean", "rev_n"]]
sym = pair_ge20.merge(rev_stats, on=["ordering_bic8", "beneficiary_bic8"], how="inner")
sym_diff = (sym["mean"] - sym["rev_mean"]).abs()

dir_df["ln_dur"] = np.log(dir_df["total_duration"].to_numpy())
r2_add_lvl = r2_grouped_ols(dir_df["total_duration"].to_numpy(), [dir_df["ordering_bic8"].to_numpy(), dir_df["beneficiary_bic8"].to_numpy()])
r2_mult_ln = r2_grouped_ols(dir_df["ln_dur"].to_numpy(), [dir_df["ordering_bic8"].to_numpy(), dir_df["beneficiary_bic8"].to_numpy()])

# ALSO TEST: D(u,v) = bank_u[?] or max(T(u), T(v))? Check each pair's min relative to ord_bic8's median/min/max pair_min!
# Check within a bank u: what does pair_ge20[ordering_bic8 == u]['min'] look like across destinations v?
top_ord_bics = pair_ge20["ordering_bic8"].value_counts().head(10).index.tolist()
bank_spread_sample = []
for b in top_ord_bics:
    vals = pair_ge20.loc[pair_ge20["ordering_bic8"] == b, "min"].to_numpy()
    bank_spread_sample.append({
        "ord_bic8": str(b), "n_pairs": int(len(vals)),
        "min": int(vals.min()), "p25": float(np.quantile(vals, 0.25)),
        "med": float(np.median(vals)), "p75": float(np.quantile(vals, 0.75)), "max": int(vals.max()),
    })

bin_edges = np.linspace(0.5, 4.5, 61)
hist_pair_min, _ = np.histogram(np.log10(pair_ge20["min"].clip(lower=1)), bins=bin_edges)
edges_80 = np.linspace(0.3, 4.5, 81)
hist_global_dir, _ = np.histogram(np.log10(dir_df["total_duration"].clip(lower=1)), bins=edges_80)
hist_global_ind, _ = np.histogram(np.log10(df.loc[is_clean & is_indirect, "total_duration"].clip(lower=1)), bins=edges_80)
hist_global_cov, _ = np.histogram(np.log10(df.loc[is_clean & is_covered, "total_duration"].clip(lower=1)), bins=edges_80)

p3_clean_mask = is_clean & (df["participant_count"] == 3) & is_qflag_y & (~start_mis) & (~end_mis)
p3_sub = df.loc[p3_clean_mask, ["bic8_path", "tx_type", "total_duration", "ordering_bic8", "beneficiary_bic8"]].copy()
split_bics = p3_sub["bic8_path"].str.split(sep, expand=True)
if split_bics.shape[1] != 3:
    print(f"WARNING: p3_clean_mask split width = {split_bics.shape[1]}, filtering to 3-token rows")
    valid3 = p3_sub["bic8_path"].str.count(sep) == 2
    p3_sub = p3_sub.loc[valid3].copy()
    split_bics = p3_sub["bic8_path"].str.split(sep, expand=True)
p3_sub["A"] = split_bics[0]
p3_sub["B"] = split_bics[1]
p3_sub["C"] = split_bics[2]

pair_lookup = pair_stats[pair_stats["n"] >= 3].set_index(["ordering_bic8", "beneficiary_bic8"])["mean"]
p3_sub["d_AB"] = p3_sub.set_index(["A", "B"]).index.map(pair_lookup)
p3_sub["d_BC"] = p3_sub.set_index(["B", "C"]).index.map(pair_lookup)
p3_sub["d_AC"] = p3_sub.set_index(["A", "C"]).index.map(pair_lookup)

both_hops = p3_sub.dropna(subset=["d_AB", "d_BC"]).copy()
both_hops["sum_AB_BC"] = both_hops["d_AB"] + both_hops["d_BC"]
has_all3 = both_hops.dropna(subset=["d_AC"]).copy()

p3_comp = {}
for tt in ["Indirect", "Covered"]:
    g = both_hops[both_hops["tx_type"] == tt]
    if len(g) > 0:
        ratio = (g["total_duration"] / g["sum_AB_BC"])
        diff = (g["total_duration"] - g["sum_AB_BC"])
        corr_sum = float(np.corrcoef(g["total_duration"], g["sum_AB_BC"])[0, 1]) if len(g) > 10 else None
        g3 = has_all3[has_all3["tx_type"] == tt]
        corr_ac = float(np.corrcoef(g3["total_duration"], g3["d_AC"])[0, 1]) if len(g3) > 10 else None
        p3_comp[tt] = {
            "n_with_AB_BC": int(len(g)),
            "mean_obs": round(float(g["total_duration"].mean()), 2),
            "mean_sum_AB_BC": round(float(g["sum_AB_BC"].mean()), 2),
            "median_ratio_obs_over_sum": round(float(ratio.median()), 4),
            "p10_ratio": round(float(ratio.quantile(0.1)), 4),
            "p90_ratio": round(float(ratio.quantile(0.9)), 4),
            "median_diff": round(float(diff.median()), 2),
            "corr_with_AB_plus_BC": round(corr_sum, 5) if corr_sum is not None else None,
            "n_with_all3": int(len(g3)),
            "corr_with_dAC": round(corr_ac, 5) if corr_ac is not None else None,
        }

p3_ind = p3_sub[p3_sub["tx_type"] == "Indirect"]
p3_cov = p3_sub[p3_sub["tx_type"] == "Covered"]
r2_p3_ind_banks = r2_grouped_ols(p3_ind["total_duration"].to_numpy(), [p3_ind["A"].to_numpy(), p3_ind["B"].to_numpy(), p3_ind["C"].to_numpy()])
r2_p3_cov_banks = r2_grouped_ols(p3_cov["total_duration"].to_numpy(), [p3_cov["A"].to_numpy(), p3_cov["B"].to_numpy(), p3_cov["C"].to_numpy()])
r2_p3_ind_noB = r2_grouped_ols(p3_ind["total_duration"].to_numpy(), [p3_ind["A"].to_numpy(), p3_ind["C"].to_numpy()])

top10_pairs = pair_stats.sort_values("n", ascending=False).head(10)
noise_rows = []
for _, r in top10_pairs.iterrows():
    u, v = r["ordering_bic8"], r["beneficiary_bic8"]
    vals = dir_df.loc[(dir_df["ordering_bic8"] == u) & (dir_df["beneficiary_bic8"] == v), "total_duration"].to_numpy()
    mn = vals.min()
    exc = vals - mn
    noise_rows.append({
        "pair": f"{u}->{v}", "n": int(len(vals)), "min": int(mn),
        "mean": round(float(vals.mean()), 2), "std": round(float(vals.std()), 2),
        "excess_mean": round(float(exc.mean()), 2), "excess_var": round(float(exc.var()), 2),
        "p50": float(np.median(vals)), "p95": float(np.quantile(vals, 0.95)),
    })

master["duration"] = {
    "direct_pair_r2_n_ge_5": round(r2_pair_ge5, 6),
    "direct_pair_rmse_n_ge_5": round(rmse_pair_ge5, 2),
    "direct_pair_r2_n_ge_20": round(r2_pair_ge20, 6),
    "direct_pair_rmse_n_ge_20": round(rmse_pair_ge20, 2),
    "direct_add_bank_fe_r2_lvl": round(r2_add_lvl, 6),
    "direct_mult_bank_fe_r2_ln": round(r2_mult_ln, 6),
    "direct_pair_std_median_ge100": round(float(pair_ge100["std"].median()), 3),
    "direct_pair_std_max_ge100": round(float(pair_ge100["std"].max()), 3),
    "sym_pairs_ge20": int(len(sym)),
    "sym_abs_diff_median": round(float(sym_diff.median()), 2),
    "sym_within_10s_pct": round(100.0 * float((sym_diff <= 10).mean()), 2) if len(sym) > 0 else None,
    "sym_corr": round(float(np.corrcoef(sym["mean"], sym["rev_mean"])[0, 1]), 5) if len(sym) > 5 else None,
    "sym_top5_asym": sym.assign(abs_diff=sym_diff).sort_values("abs_diff", ascending=False).head(5)[["ordering_bic8", "beneficiary_bic8", "mean", "rev_mean", "n", "rev_n"]].to_dict("records") if len(sym) > 0 else [],
    "bank_pair_spread_sample": bank_spread_sample,
    "n3_composition": p3_comp,
    "r2_p3_ind_A_B_C": round(r2_p3_ind_banks, 5),
    "r2_p3_ind_A_C_only": round(r2_p3_ind_noB, 5),
    "r2_p3_cov_A_B_C": round(r2_p3_cov_banks, 5),
    "top10_pair_noise": noise_rows,
}
print("Duration diagnostics:", master["duration"])

pd.DataFrame({
    "bin_left": np.round(bin_edges[:-1], 4),
    "bin_right": np.round(bin_edges[1:], 4),
    "pair_min_count": hist_pair_min,
}).to_csv("work/chart_data/dur_pair_min_hist.csv", index=False)

pd.DataFrame({
    "bin_left": np.round(edges_80[:-1], 4),
    "bin_right": np.round(edges_80[1:], 4),
    "direct_count": hist_global_dir,
    "indirect_count": hist_global_ind,
    "covered_count": hist_global_cov,
}).to_csv("work/chart_data/dur_global_trimodal_hist.csv", index=False)

p3_bank_cmp = p3_sub.groupby(["ordering_bic8", "tx_type"], observed=True)["total_duration"].agg(["count", "mean", "median"]).unstack()
cmp_rows = []
for bic in p3_bank_cmp.index:
    v_ind = p3_bank_cmp.loc[bic, ("count", "Indirect")] if ("count", "Indirect") in p3_bank_cmp.columns else np.nan
    v_cov = p3_bank_cmp.loc[bic, ("count", "Covered")] if ("count", "Covered") in p3_bank_cmp.columns else np.nan
    if pd.notna(v_ind) and pd.notna(v_cov):
        c_ind = int(v_ind)
        c_cov = int(v_cov)
        if c_ind >= 200 and c_cov >= 200:
            cmp_rows.append({
                "bic8": str(bic), "n_ind": c_ind, "n_cov": c_cov,
                "mean_ind": round(float(p3_bank_cmp.loc[bic, ("mean", "Indirect")]), 1),
                "mean_cov": round(float(p3_bank_cmp.loc[bic, ("mean", "Covered")]), 1),
                "med_ind": round(float(p3_bank_cmp.loc[bic, ("median", "Indirect")]), 1),
                "med_cov": round(float(p3_bank_cmp.loc[bic, ("median", "Covered")]), 1),
                "ratio": round(float(p3_bank_cmp.loc[bic, ("mean", "Covered")] / p3_bank_cmp.loc[bic, ("mean", "Indirect")]), 4),
            })
pd.DataFrame(cmp_rows).sort_values("n_cov", ascending=False).head(25).to_csv("work/chart_data/covered_vs_indirect_bank_dur.csv", index=False)
del dir_df, p3_sub, sub5, sub20

# ------------------------------------------------------------------
# 7. PARTY DETAILS & TOKENIZATION GRAMMAR
# ------------------------------------------------------------------
print("Computing Party Details & Tokenization stats...")
ord_cc_nonempty = (df["ord_inferred_cc"] != "")
ord_cc_match = ord_cc_nonempty & (df["ord_inferred_cc"] == df["ord_bic_country"])
ben_cc_nonempty = (df["ben_inferred_cc"] != "")
ben_cc_match = ben_cc_nonempty & (df["ben_inferred_cc"] == df["ben_bic_country"])

iban_countries = ["DE", "FR", "GB", "CH", "IT", "NL", "BE", "ES", "AT", "LU"]
non_iban_countries = ["US", "CN", "JP", "SG", "AU", "HK", "CA", "IM"]
iban_by_country = []
for cc in iban_countries + non_iban_countries:
    m_o = df["ord_bic_country"] == cc
    m_b = df["ben_bic_country"] == cc
    iban_by_country.append({
        "country": cc,
        "ord_n": int(m_o.sum()), "ord_iban_pct": round(100.0 * float(df.loc[m_o, "ord_has_iban"].mean()), 3),
        "ben_n": int(m_b.sum()), "ben_iban_pct": round(100.0 * float(df.loc[m_b, "ben_has_iban"].mean()), 3),
    })

top15_ben_inf = (100.0 * df.loc[ben_cc_nonempty, "ben_inferred_cc"].value_counts().head(15) / ben_cc_nonempty.sum()).round(3).to_dict()
top15_ord_bic_cc = (100.0 * df["ord_bic_country"].value_counts().head(15) / len(df)).round(3).to_dict()

ord_fw_vc = df.loc[df["ord_fin_word"] != "", "ord_fin_word"].value_counts().to_dict()
ben_fw_vc = df.loc[df["ben_fin_word"] != "", "ben_fin_word"].value_counts().to_dict()
ord_fw_in_struct = float((df.loc[df["ord_is_struct_123"], "ord_fin_word"] != "").mean())
ord_fw_in_unstruct = float((df.loc[~df["ord_is_struct_123"], "ord_fin_word"] != "").mean())
ben_fw_in_struct = float((df.loc[df["ben_is_struct_123"], "ben_fin_word"] != "").mean())
ben_fw_in_unstruct = float((df.loc[~df["ben_is_struct_123"], "ben_fin_word"] != "").mean())

master["party_details"] = {
    "ord_inferred_nonempty_pct": round(100.0 * float(ord_cc_nonempty.mean()), 3),
    "ord_cc_match_among_nonempty_pct": round(100.0 * float(ord_cc_match.sum() / ord_cc_nonempty.sum()), 4),
    "ben_inferred_nonempty_pct": round(100.0 * float(ben_cc_nonempty.mean()), 3),
    "ben_cc_match_among_nonempty_pct": round(100.0 * float(ben_cc_match.sum() / ben_cc_nonempty.sum()), 4),
    "ord_iban_pct": round(100.0 * float(df["ord_has_iban"].mean()), 3),
    "ben_iban_pct": round(100.0 * float(df["ben_has_iban"].mean()), 3),
    "ord_lb35_pct": round(100.0 * float(df["ord_has_lb35"].mean()), 3),
    "ben_lb35_pct": round(100.0 * float(df["ben_has_lb35"].mean()), 3),
    "ord_trunc_pct": round(100.0 * float(df["ord_has_trunc"].mean()), 3),
    "ben_trunc_pct": round(100.0 * float(df["ben_has_trunc"].mean()), 3),
    "ord_struct123_pct": round(100.0 * float(df["ord_is_struct_123"].mean()), 3),
    "ben_struct123_pct": round(100.0 * float(df["ben_is_struct_123"].mean()), 3),
    "ord_finword_pct": round(100.0 * float((df["ord_fin_word"] != "").mean()), 3),
    "ben_finword_pct": round(100.0 * float((df["ben_fin_word"] != "").mean()), 3),
    "ord_finword_in_struct_pct": round(100.0 * ord_fw_in_struct, 5),
    "ord_finword_in_unstruct_pct": round(100.0 * ord_fw_in_unstruct, 3),
    "ben_finword_in_struct_pct": round(100.0 * ben_fw_in_struct, 5),
    "ben_finword_in_unstruct_pct": round(100.0 * ben_fw_in_unstruct, 3),
    "ord_fin_words": {str(k): int(v) for k, v in ord_fw_vc.items()},
    "ben_fin_words": {str(k): int(v) for k, v in ben_fw_vc.items()},
    "iban_by_country": iban_by_country,
    "top15_ben_inferred_cc_pct": top15_ben_inf,
    "top15_ord_bic_cc_pct": top15_ord_bic_cc,
}

# ------------------------------------------------------------------
# 8. ANOMALY & FRAUD MECHANICS (INCL. COVERED CONFOUNDER CHECK)
# ------------------------------------------------------------------
print("Computing Anomaly deep tables...")
anom_labels = ["", "ROMANCE_SCAM", "FAT_FINGER_MISTAKE", "CEO_SCAM", "DUPLICATE_PAYMENT", "ACCOUNT_REPLACEMENT_SCAM", "CUSTOMER_RECALL"]
anom_summary = []
for lbl in anom_labels:
    sub = df[df["anomaly_type"].fillna("") == lbl]
    n = int(len(sub))
    mh_sub = sub[sub["participant_count"] >= 3]
    cov_rate_all = 100.0 * float((sub["tx_type"] == "Covered").mean())
    cov_rate_mh = 100.0 * float((mh_sub["tx_type"] == "Covered").mean()) if len(mh_sub) > 0 else 0.0
    direct_rate = 100.0 * float((sub["tx_type"] == "Direct").mean())
    idx = sub.index.to_numpy()
    anom_summary.append({
        "anomaly": "NORMAL" if lbl == "" else lbl,
        "n": n,
        "pct_all": round(100.0 * n / len(df), 5),
        "b2b_pct": round(100.0 * float(((sub["ordering_type"] == "B") & (sub["beneficiary_type"] == "B")).mean()), 2),
        "ord_B_pct": round(100.0 * float((sub["ordering_type"] == "B").mean()), 2),
        "ben_B_pct": round(100.0 * float((sub["beneficiary_type"] == "B").mean()), 2),
        "usd_median": round(float(sub["usd_amount"].median()), 2),
        "usd_mean": round(float(sub["usd_amount"].mean()), 2),
        "usd_max": round(float(sub["usd_amount"].max()), 2),
        "dur_median": round(float(sub["total_duration"].median()), 1),
        "direct_pct": round(direct_rate, 2),
        "covered_all_pct": round(cov_rate_all, 3),
        "covered_among_multihop_pct": round(cov_rate_mh, 3),
        "qflag_n_pct": round(100.0 * float((sub["continuity_qflag"] == "N").mean()), 3),
        "start_mis_pct": round(100.0 * float(start_mis.iloc[idx].mean()), 3),
        "end_mis_pct": round(100.0 * float(end_mis.iloc[idx].mean()), 3),
        "ben_iban_pct": round(100.0 * float(sub["ben_has_iban"].mean()), 2),
    })

rom = df[df["anomaly_type"] == "ROMANCE_SCAM"].sort_values(["ordering_account", "epoch_seconds", "row_id"]).copy()
rom["step"] = rom.groupby("ordering_account", observed=True).cumcount() + 1
rom["chain_len"] = rom.groupby("ordering_account", observed=True)["row_id"].transform("count")
rom_steps = rom.groupby("step", observed=True).agg(
    n=("usd_amount", "count"),
    median_usd=("usd_amount", "median"),
    mean_usd=("usd_amount", "mean"),
    p25_usd=("usd_amount", lambda s: float(np.quantile(s, 0.25))),
    p75_usd=("usd_amount", lambda s: float(np.quantile(s, 0.75))),
    floor_pct=("usd_amount", lambda s: 100.0 * float((s <= 0.05).mean())),
).reset_index()
rom_steps.to_csv("work/chart_data/romance_scam_steps.csv", index=False)

ceo_accts = set(df.loc[df["anomaly_type"] == "CEO_SCAM", "beneficiary_account"].unique())
ars_accts = set(df.loc[df["anomaly_type"] == "ACCOUNT_REPLACEMENT_SCAM", "beneficiary_account"].unique())
rom_accts = set(df.loc[df["anomaly_type"] == "ROMANCE_SCAM", "beneficiary_account"].unique())
norm_accts = set(df.loc[is_clean, "beneficiary_account"].unique())
mule_union = ceo_accts | ars_accts | rom_accts

mule_venn = {
    "ceo_only": len(ceo_accts - ars_accts - rom_accts),
    "ars_only": len(ars_accts - ceo_accts - rom_accts),
    "rom_only": len(rom_accts - ceo_accts - ars_accts),
    "ceo_ars_only": len((ceo_accts & ars_accts) - rom_accts),
    "ceo_rom_only": len((ceo_accts & rom_accts) - ars_accts),
    "ars_rom_only": len((ars_accts & rom_accts) - ceo_accts),
    "all_three": len(ceo_accts & ars_accts & rom_accts),
    "total_ceo": len(ceo_accts),
    "total_ars": len(ars_accts),
    "total_rom": len(rom_accts),
    "union_size": len(mule_union),
    "union_also_in_normal": len(mule_union & norm_accts),
    "normal_rows_to_mule_union": int(df.loc[is_clean, "beneficiary_account"].isin(mule_union).sum()),
}

mule_rows = df[df["anomaly_type"].isin(["CEO_SCAM", "ACCOUNT_REPLACEMENT_SCAM"])]
mule_ben_cc = (100.0 * mule_rows["ben_bic_country"].value_counts().head(12) / len(mule_rows)).round(2).to_dict()
norm_ben_cc = (100.0 * df.loc[is_clean, "ben_bic_country"].value_counts() / is_clean.sum()).round(3).to_dict()
mule_country_cmp = [
    {"country": str(k), "mule_pct": float(v), "normal_pct": float(norm_ben_cc.get(k, 0.0)), "lift": round(float(v) / max(0.01, float(norm_ben_cc.get(k, 0.01))), 2)}
    for k, v in mule_ben_cc.items()
]

ff_log10 = np.log10(df.loc[df["anomaly_type"] == "FAT_FINGER_MISTAKE", "usd_amount"].clip(lower=0.01))
cl_log10 = np.log10(df.loc[is_clean, "usd_amount"].clip(lower=0.01))
b_ff = np.linspace(-2, 11, 53)
h_ff, _ = np.histogram(ff_log10, bins=b_ff, density=True)
h_cl, _ = np.histogram(cl_log10, bins=b_ff, density=True)
pd.DataFrame({
    "bin_left": np.round(b_ff[:-1], 3), "bin_right": np.round(b_ff[1:], 3),
    "fat_finger_density": np.round(h_ff, 5), "normal_density": np.round(h_cl, 5),
}).to_csv("work/chart_data/fat_finger_log10_hist.csv", index=False)

cr_mask = (df["anomaly_type"] == "CUSTOMER_RECALL").to_numpy()
rng = np.random.default_rng(42)
norm_sample_idx = rng.choice(np.where(is_clean)[0], size=200000, replace=False)
cr_tests = {}
for col in ["usd_amount", "total_duration", "epoch_seconds", "fx_count", "participant_count", "ord_det_len", "ben_det_len"]:
    s_cr = df[col].to_numpy()[cr_mask]
    s_no = df[col].to_numpy()[norm_sample_idx]
    ks_stat, ks_p = stats.ks_2samp(s_cr, s_no)
    cr_tests[col] = {
        "cr_mean": round(float(s_cr.mean()), 2), "norm_mean": round(float(s_no.mean()), 2),
        "cr_med": round(float(np.median(s_cr)), 2), "norm_med": round(float(np.median(s_no)), 2),
        "ks_stat": round(float(ks_stat), 4), "ks_p": round(float(ks_p), 4),
    }

dup = df[df["anomaly_type"] == "DUPLICATE_PAYMENT"].sort_values(["row_id"])
grp_cols = ["currency_code", "usd_amount", "ordering_bic8", "beneficiary_bic8", "bic8_path",
            "total_duration", "continuity_qflag", "fx_count", "participant_count", "tx_type",
            "ordering_account", "beneficiary_account", "ordering_type", "beneficiary_type"]
dup_groups = dup.groupby(grp_cols, observed=True).agg(
    size=("row_id", "count"),
    dt_span=("epoch_seconds", lambda s: int(s.max() - s.min())),
    id_span=("row_id", lambda s: int(s.max() - s.min() + 1)),
).reset_index()
dup_size_dist = dup_groups["size"].value_counts().sort_index().to_dict()

master["anomalies"] = {
    "summary_table": anom_summary,
    "mule_venn": mule_venn,
    "mule_country_lift": mule_country_cmp,
    "customer_recall_ks": cr_tests,
    "duplicate_groups_count": int(len(dup_groups)),
    "duplicate_size_dist": {int(k): int(v) for k, v in dup_size_dist.items()},
    "duplicate_all_consecutive_pct": round(100.0 * float((dup_groups["size"] == dup_groups["id_span"]).mean()), 2),
    "duplicate_dt_span_quantiles": [int(x) for x in np.quantile(dup_groups["dt_span"], [0, 0.25, 0.5, 0.75, 1.0])],
}

with open("work/master_metrics.json", "w", encoding="utf-8") as f:
    json.dump(master, f, indent=2)

print(f"ALL DONE in {time.time() - t0:.1f}s -> saved work/master_metrics.json and work/chart_data/")
