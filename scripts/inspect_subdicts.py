import json

with open("work/master_metrics.json", "r", encoding="utf-8") as f:
    M = json.load(f)

for k in ("path_checks", "entity_graph", "payment_value", "fx_and_currency", "party_details"):
    print(f"\n=== {k} ===")
    for sk, sv in M[k].items():
        if isinstance(sv, list) and len(sv) > 12:
            print(f"  {sk}: len={len(sv)}, head3={sv[:3]}, tail2={sv[-2:]}")
        elif isinstance(sv, dict) and len(sv) > 15:
            print(f"  {sk}: len={len(sv)}, sample={list(sv.items())[:8]}")
        else:
            print(f"  {sk}: {sv}")

print("\n=== romance_steps ===")
for r in M["anomalies"]["romance_steps"]:
    print(" ", r)
print("dur top10_pair_noise:")
for r in M["duration"]["top10_pair_noise"][:5]:
    print(" ", r)
print("dur sym_top5_asym:")
for r in M["duration"]["sym_top5_asym"]:
    print(" ", r)
