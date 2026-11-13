import json

with open("work/master_metrics.json", "r", encoding="utf-8") as f:
    M = json.load(f)
with open("work/chart_data/tz_profiles.json", "r", encoding="utf-8") as f:
    tz = json.load(f)
with open("work/audit_summary.json", "r", encoding="utf-8") as f:
    A = json.load(f)

print("=== TZ PROFILES ===")
for r in tz:
    print(f"{r['country']}: n={r['n']}, off={r['utc_offset_best']:+d}, r={r['corr_with_gb']:.4f}, "
          f"utc_sat={r['utc_dow_pct'][5]:.2f}%, utc_sun={r['utc_dow_pct'][6]:.2f}%, "
          f"loc_fri={r['local_dow_pct'][4]:.2f}%, loc_sat={r['local_dow_pct'][5]:.2f}%, loc_sun={r['local_dow_pct'][6]:.2f}%")

print("\n=== SENSITIVITY TABLE ===")
for r in M["data_prep_sensitivity"]:
    print(r)

print("\n=== TOP KEYS & SCALARS ===")
for k, v in M.items():
    if isinstance(v, dict):
        print(f"[{k}]: keys={list(v.keys())}")
print("\n=== ROUTING DETAILS ===")
print(M["routing"])
print("\n=== DURATION DETAILS ===")
for k, v in M["duration"].items():
    if not isinstance(v, list):
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: len={len(v)}, top3={v[:3]}")
print("\n=== ANOMALIES SUMMARY ===")
for r in M["anomalies"]["summary_table"]:
    print(r)
print("anom extra:", {k: v for k, v in M["anomalies"].items() if k not in ("summary_table", "romance_steps", "ks_pvals")})
print("ks_pvals:", M["anomalies"].get("ks_pvals"))
print("\n=== AUDIT EXTRA ===")
print({k: v for k, v in A.items() if k not in ("top15_currencies", "token_tag_counts")})
print("token_tag_counts:", A.get("token_tag_counts"))
