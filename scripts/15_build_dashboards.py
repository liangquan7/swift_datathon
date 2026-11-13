import base64
import html
import json
import os
import shutil
import subprocess
from pathlib import Path

with open("work/master_metrics.json", "r", encoding="utf-8") as f:
    M = json.load(f)
with open("work/audit_summary.json", "r", encoding="utf-8") as f:
    A = json.load(f)

def b64_img(rel_path):
    with open(rel_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

imgs = {p.name: b64_img(str(p)) for p in sorted(Path("work/charts").glob("*.png"))}

# Dynamic helper formatters from verified M and A
sens = {r["subset"]: r for r in M["data_prep_sensitivity"]}
pc = M["path_checks"]
eg = M["entity_graph"]
pv = M["payment_value"]
fc = M["fx_and_currency"]
rt = M["routing"]
dur = M["duration"]
pd_m = M["party_details"]
an = M["anomalies"]
anom_map = {r["anomaly"]: r for r in an["summary_table"]}

CSS = """
@page {
  size: 14in 8.5in;
  margin: 0;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', Arial, Helvetica, sans-serif;
  color: #0F172A;
  background: #E2E8F0;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
.page {
  width: 14in;
  height: 8.5in;
  background: #F8FAFC;
  padding: 0.24in 0.30in 0.20in 0.30in;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  page-break-after: always;
  overflow: hidden;
  position: relative;
}
.page:last-child { page-break-after: auto; }

/* Header */
.hdr {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 2.5px solid #0F2942;
  padding-bottom: 6px;
  margin-bottom: 7px;
  flex-shrink: 0;
}
.hdr-left { flex: 1; }
.eyebrow {
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #2563EB;
  margin-bottom: 2px;
}
.title {
  font-size: 16.5px;
  font-weight: 800;
  color: #0F2942;
  line-height: 1.15;
}
.hdr-badges {
  display: flex;
  gap: 6px;
  align-items: center;
  flex-shrink: 0;
  margin-left: 14px;
}
.badge {
  font-size: 8.5px;
  font-weight: 700;
  padding: 3px 8px;
  border-radius: 4px;
  border: 1px solid #CBD5E1;
  background: #FFFFFF;
  color: #334155;
  white-space: nowrap;
}
.badge.high { background: #ECFDF5; border-color: #10B981; color: #065F46; }
.badge.mixed { background: #FFFBEB; border-color: #F59E0B; color: #92400E; }
.badge.nullb { background: #FEF2F2; border-color: #EF4444; color: #991B1B; }

/* KPI Row */
.kpis {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 8px;
  margin-bottom: 7px;
  flex-shrink: 0;
}
.kpis.k4 { grid-template-columns: repeat(4, 1fr); }
.kpi {
  background: #FFFFFF;
  border: 1px solid #CBD5E1;
  border-left: 3.5px solid #0F2942;
  border-radius: 5px;
  padding: 5px 8px;
}
.kpi.blue { border-left-color: #2563EB; }
.kpi.teal { border-left-color: #0D9488; }
.kpi.amber { border-left-color: #D97706; }
.kpi.red { border-left-color: #DC2626; }
.kpi-lbl { font-size: 8.3px; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.03em; }
.kpi-val { font-size: 15px; font-weight: 800; color: #0F172A; line-height: 1.15; margin: 1px 0; }
.kpi-sub { font-size: 8.2px; color: #475569; line-height: 1.15; }

/* 3-Column Rule / Confounder / Econ strip */
.meta-grid {
  display: grid;
  grid-template-columns: 1.15fr 1fr 1fr;
  gap: 8px;
  margin-bottom: 8px;
  flex-shrink: 0;
}
.card {
  background: #FFFFFF;
  border: 1px solid #CBD5E1;
  border-radius: 6px;
  padding: 6px 9px;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
}
.card-hdr {
  font-size: 8.8px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 4px;
  display: flex;
  align-items: center;
  gap: 5px;
}
.tag {
  font-size: 7.5px;
  padding: 1px 5px;
  border-radius: 3px;
  font-weight: 800;
  color: #FFFFFF;
}
.tag.obs { background: #0F2942; }
.tag.test { background: #D97706; }
.tag.interp { background: #0D9488; }
.card ul { list-style: none; font-size: 9.0px; line-height: 1.32; color: #1E293B; }
.card ul li { margin-bottom: 2.5px; padding-left: 8px; position: relative; }
.card ul li::before { content: "•"; position: absolute; left: 0; color: #2563EB; font-weight: 900; }
.card b { color: #0F2942; }

/* Visuals area */
.viz-area {
  flex: 1;
  min-height: 0;
  display: flex;
  gap: 8px;
  align-items: stretch;
}
.viz-col {
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 0;
}
.viz-box {
  background: #FFFFFF;
  border: 1px solid #CBD5E1;
  border-radius: 6px;
  padding: 5px 7px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  flex: 1;
  min-height: 0;
  overflow: hidden;
}
.viz-box img {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  object-fit: contain;
  display: block;
}
.viz-cap {
  font-size: 8.2px;
  color: #475569;
  margin-top: 3px;
  text-align: center;
  line-height: 1.2;
  flex-shrink: 0;
}

/* Compact inline data table */
table.mini {
  width: 100%;
  border-collapse: collapse;
  font-size: 8.3px;
}
table.mini th {
  background: #0F2942;
  color: #FFFFFF;
  text-align: left;
  padding: 3.5px 5px;
  font-weight: 700;
}
table.mini td {
  padding: 3px 5px;
  border-bottom: 1px solid #E2E8F0;
  color: #1E293B;
}
table.mini tr:nth-child(even) td { background: #F8FAFC; }
.num { text-align: right; font-variant-numeric: tabular-nums; }

/* Footer */
.ftr {
  display: flex;
  justify-content: space-between;
  font-size: 8.0px;
  color: #64748B;
  border-top: 1px solid #CBD5E1;
  padding-top: 4px;
  margin-top: 6px;
  flex-shrink: 0;
}
"""

def kpi_html(items, cols=5):
    cls = "kpis k4" if cols == 4 else "kpis"
    out = [f'<div class="{cls}">']
    for accent, lbl, val, sub in items:
        out.append(f'<div class="kpi {accent}"><div class="kpi-lbl">{lbl}</div><div class="kpi-val">{val}</div><div class="kpi-sub">{sub}</div></div>')
    out.append('</div>')
    return "\n".join(out)

def meta_html(obs_bullets, test_bullets, interp_bullets):
    def li_list(bs): return "".join(f"<li>{b}</li>" for b in bs)
    return f"""
    <div class="meta-grid">
      <div class="card">
        <div class="card-hdr"><span class="tag obs">OBSERVED RULE</span> Condition &bull; Effect &bull; Size &bull; Scope</div>
        <ul>{li_list(obs_bullets)}</ul>
      </div>
      <div class="card">
        <div class="card-hdr"><span class="tag test">TESTED &amp; CONTROLS</span> Confounders &amp; Rejected Hypotheses</div>
        <ul>{li_list(test_bullets)}</ul>
      </div>
      <div class="card">
        <div class="card-hdr"><span class="tag interp">INTERPRETATION</span> Economic Logic, Deviations &amp; Limits</div>
        <ul>{li_list(interp_bullets)}</ul>
      </div>
    </div>
    """

def render_page(idx, group_tag, conf_cls, conf_txt, title, scope_txt, kpis, obs_b, test_b, interp_b, viz_html):
    cols = len(kpis)
    return f"""
    <div class="page">
      <div>
        <div class="hdr">
          <div class="hdr-left">
            <div class="eyebrow">Dashboard {idx} of 10 &bull; {group_tag}</div>
            <div class="title">{title}</div>
          </div>
          <div class="hdr-badges">
            <span class="badge">{scope_txt}</span>
            <span class="badge {conf_cls}">{conf_txt}</span>
          </div>
        </div>
        {kpi_html(kpis, cols=cols)}
        {meta_html(obs_b, test_b, interp_b)}
      </div>
      <div class="viz-area">
        {viz_html}
      </div>
      <div class="ftr">
        <span>UMDAC Datathon 2026 (Preliminary) &bull; Exclusive Analysis of Provided Synthetic Swift Dataset (No External Data)</span>
        <span>Standalone Dashboard {idx}/10 &bull; Observed Facts Strictly Separated from Economic Interpretations</span>
      </div>
    </div>
    """

pages = []

# D1: Schema Audit, Cleaning Sensitivity & Anomaly Catalog
raw_s = sens["Raw All Rows"]
cln_s = sens["Exclude Labeled Anomalies"]
qfy_s = sens["Clean + QFLAG='Y'"]
d1_rows = []
for r in M["data_prep_sensitivity"]:
    d1_rows.append(
        f"<tr><td><b>{r['subset']}</b></td>"
        f"<td class='num'>{r['n']:,}</td>"
        f"<td class='num'>${r['usd_mean']:,.0f}</td>"
        f"<td class='num'>${r['usd_median']:,.0f}</td>"
        f"<td class='num'>${r['usd_p95']:,.0f}</td>"
        f"<td class='num'>{r['direct_pct']:.2f}%</td>"
        f"<td class='num'>{r['covered_among_multihop_pct']:.2f}%</td></tr>"
    )
d1_viz = f"""
<div class="viz-col" style="flex: 1.38;">
  <div class="viz-box" style="flex: 1.25;">
    <img src="{imgs['c01_sensitivity_mean_bias.png']}" />
    <div class="viz-cap"><b>Sensitivity Log:</b> Excluding 1,940 fat-finger rows lowers mean USD by -18.1% (p95 &amp; median steady); dropping QFLAG='N' shifts Direct +1.84 pp via route-length selection.</div>
  </div>
  <div class="viz-box" style="flex: 0.78; padding: 5px 8px; align-items: stretch; justify-content: flex-start;">
    <div style="font-size:8.4px; font-weight:800; color:#0F2942; margin-bottom:2px;">DATA PREPARATION SENSITIVITY MATRIX (ALL 5 TESTED SUBSETS)</div>
    <table class="mini">
      <thead><tr><th>Filtering Subset</th><th class="num">Row Count</th><th class="num">USD Mean</th><th class="num">USD Med</th><th class="num">USD p95</th><th class="num">Direct %</th><th class="num">Cov MH%</th></tr></thead>
      <tbody>{''.join(d1_rows)}</tbody>
    </table>
  </div>
</div>
<div class="viz-col" style="flex: 1.02;">
  <div class="viz-box" style="flex: 1.25;">
    <img src="{imgs['c02_anomaly_counts_bar.png']}" />
    <div class="viz-cap"><b>Undocumented Findings:</b> Six anomaly labels (7,006 rows) + row-0 DISCLAIMER + 1 exact duplicate row-pair (row_id 3,487,188 &amp; 3,487,190).</div>
  </div>
  <div class="viz-box" style="flex: 0.78; padding: 5px 8px; align-items: stretch; justify-content: flex-start;">
    <div style="font-size:8.4px; font-weight:800; color:#0F2942; margin-bottom:2px;">SCHEMA COMPLIANCE, HIDDEN METADATA &amp; DUPLICATE VERIFICATION</div>
    <table class="mini">
      <thead><tr><th>Audit Check (10,000,000 Rows)</th><th class="num">Observed Result</th><th>Handling Rule</th></tr></thead>
      <tbody>
        <tr><td>Core 19 Analytical Columns Nulls</td><td class="num"><b>0 nulls (0.00%)</b></td><td>No imputation needed</td></tr>
        <tr><td>Row-0 Undocumented DISCLAIMER</td><td class="num"><b>1 populated row</b></td><td>Proves NumPy/Faker synthesis</td></tr>
        <tr><td>Exact Full-Row Duplicates</td><td class="num"><b>1 pair (2 rows)</b></td><td>Rows 3,487,188 &amp; 3,487,190 (DUP)</td></tr>
        <tr><td>No-Timestamp Exact Duplicates</td><td class="num"><b>506 pairs (745 rows)</b></td><td>679 DUP + 44 Normal + 22 Scam</td></tr>
        <tr><td>BIC8 / Amount / Duration Validity</td><td class="num"><b>0 invalid rows</b></td><td>Char-7=='0' 10M; Dur 2–21,649s</td></tr>
      </tbody>
    </table>
  </div>
</div>
"""
pages.append(render_page(
    1, "Data Audit, Integrity & Sensitivity", "high", "Confidence: HIGH (Full 10M Audit)",
    "Tiny Tail Defective Subsets Drive Massive Mean Distortion & Filtering Bias If Mishandled",
    "Scope: All 10,000,000 Rows &bull; 21 Columns",
    [
        ("", "Total Rows & Missingness", "10,000,000", "0 core nulls; 21 cols (4.12 GB CSV)"),
        ("amber", "Undocumented Schema Items", "2 Found", "DISCLAIMER (row 0) + CUSTOMER_RECALL (317)"),
        ("red", "Raw vs Clean Mean USD", f"${raw_s['usd_mean']/1e3:.1f}k vs ${cln_s['usd_mean']/1e3:.1f}k", "+22.1% inflation from 1,940 fat-finger rows"),
        ("teal", "Median USD Stability", f"${raw_s['usd_median']:,.0f} vs ${cln_s['usd_median']:,.0f}", "Unchanged (<$1 shift) across all 5 filters"),
        ("blue", "Collider Selection Risk", f"+{qfy_s['direct_pct'] - cln_s['direct_pct']:.2f} pp Direct", "Naive QFLAG='Y' filter drops multi-hop routes"),
    ],
    [
        "<b>Format Compliance:</b> All 10M timestamps (Apr 30–Jul 31 2023), positive amounts ($0.01–$15.66B), durations (2–21,649s), and BIC8s (char-7 == '0') pass validation.",
        "<b>Exact vs Near Duplicates:</b> 1 exact duplicate row-pair (row 3,487,188 &amp; 3,487,190, both DUPLICATE_PAYMENT); 745 no-TS duplicate rows (679 labeled + 44 normal recurring + 22 scam).",
        "<b>Undocumented Features:</b> DISCLAIMER col on row_id=0 discloses NumPy/Faker rules; CUSTOMER_RECALL (n=317) absent from brief Appendix A.1.",
    ],
    [
        "<b>Sensitivity Check:</b> Removing 7,006 anomalies leaves median USD ($3,663), Direct share (63.23%), Covered multihop share (11.32%) unchanged, while raw mean drops -18.1%.",
        "<b>Rejected Filter Strategy:</b> Dropping QFLAG='N' or path-endpoint mismatches (9.73% of rows) is unnecessary for value/timing laws and introduces length selection bias.",
    ],
    [
        "<b>Why Anomalies Distort Means:</b> In heavy-tailed payment logs, uncleaned operational fat-finger shifts (+2 to +6 orders of magnitude) dominate arithmetic aggregates.",
        "<b>Baseline Rule for Analysts:</b> Use Exclude Labeled Anomalies (9.993M rows) for pricing/size benchmarks, but retain QFLAG='N' &amp; mismatches when studying routing.",
    ],
    d1_viz
))

# D2: Entity Graph & Party-Type Coupling
d2_viz = f"""
<div class="viz-col" style="flex: 0.92;">
  <div class="viz-box" style="flex: 1.32;">
    <img src="{imgs['c03_party_matrix.png']}" />
    <div class="viz-cap"><b>Party Origination Skew:</b> 78.75% Business senders; Receiver type P(Ben=B) stays ~28.8% regardless of sender.</div>
  </div>
  <div class="viz-box" style="flex: 0.72; padding: 5px 8px; align-items: stretch; justify-content: flex-start;">
    <div style="font-size:8.4px; font-weight:800; color:#0F2942; margin-bottom:2px;">10M ENTITY PARTY PAIRS &amp; RECEIVER COUPLING</div>
    <table class="mini">
      <thead><tr><th>Segment</th><th class="num">All 10M Count</th><th class="num">10M Share</th><th class="num">Clean Med USD</th></tr></thead>
      <tbody>
        <tr><td><b>B &rarr; C</b> (Corp to Retail)</td><td class="num">5,603,692</td><td class="num">56.04%</td><td class="num">$3,118</td></tr>
        <tr><td><b>B &rarr; B</b> (Wholesale)</td><td class="num">2,270,945</td><td class="num">22.71%</td><td class="num">$22,883</td></tr>
        <tr><td><b>C &rarr; C</b> (Retail P2P)</td><td class="num">1,514,758</td><td class="num">15.15%</td><td class="num">$464</td></tr>
        <tr><td><b>C &rarr; B</b> (Retail to Corp)</td><td class="num">610,605</td><td class="num">6.11%</td><td class="num">$3,236</td></tr>
      </tbody>
    </table>
  </div>
</div>
<div class="viz-col" style="flex: 1.48;">
  <div class="viz-box" style="flex: 1.32;">
    <img src="{imgs['c04_entity_cardinality_asym.png']}" />
    <div class="viz-cap"><b>Account Cardinality Paradox:</b> Strict 1:1 Ordering Account-to-BIC vs. 2.38% multi-BIC Beneficiaries &amp; 253k Dual-Role accounts switching B/C type.</div>
  </div>
  <div class="viz-box" style="flex: 0.72; padding: 5px 8px; align-items: stretch; justify-content: flex-start;">
    <div style="font-size:8.4px; font-weight:800; color:#0F2942; margin-bottom:2px;">NETWORK NODE CARDINALITY &amp; DUAL-ROLE ARTIFACT PROOF</div>
    <table class="mini">
      <thead><tr><th>Graph Entity Dimension</th><th class="num">Unique Count</th><th class="num">Country / Bank Scope</th><th>Observed Structural Rule</th></tr></thead>
      <tbody>
        <tr><td>Ordering Accounts</td><td class="num"><b>391,132</b></td><td class="num">8,392 BIC8 / 228 Ctry</td><td>100.00% locked 1:1 to single BIC8 &amp; single B/C</td></tr>
        <tr><td>Beneficiary Accounts</td><td class="num"><b>2,590,877</b></td><td class="num">8,266 BIC8 / 229 Ctry</td><td>97.62% 1-BIC; 61,606 (2.38%) map to 2–8 BIC8s (max 8)</td></tr>
        <tr><td>Dual-Role Overlap Accounts</td><td class="num"><b>253,120</b></td><td class="num">8,686 Union BIC8s</td><td>Same BIC8 <b>99.92%</b>, Same Ctry <b>99.92%</b>, Same B/C <b>49.60%</b></td></tr>
        <tr><td>Non-IBAN Faker Locale Check</td><td class="num"><b>100% of 61.6k</b></td><td class="num">0.00% IBAN Share</td><td>Multi-BIC beneficiaries share single Faker text locale</td></tr>
      </tbody>
    </table>
  </div>
</div>
"""
pages.append(render_page(
    2, "Entity Graph & Party-Type Architecture", "high", "Confidence: HIGH (Full 10M Graph)",
    "Accounts Couple Strictly to Banks, Yet Beneficiary Role & Type Sampling Are Decoupled",
    "Scope: 391k Ord Accts &bull; 2.59M Ben Accts &bull; 8,686 BIC8s",
    [
        ("blue", "Largest Segment (B2C)", "56.04% (5.60M)", "Baseline: B2B 22.7%, C2C 15.1%, C2B 6.1%"),
        ("", "Ordering Accounts", f"{eg['unique_ordering_accounts']:,}", "100.00% 1:1 to single BIC8 & single B/C type"),
        ("teal", "Beneficiary Accounts", f"{eg['unique_beneficiary_accounts']:,}", f"6.62x ordering pool; {eg['ben_acct_1_bic_pct']:.2f}% 1-BIC"),
        ("red", "Multi-BIC Ben Accts", "61,606 (2.38%)", "Map to 2–8 BIC8s; 0% IBANs but 1 inferred CC"),
        ("amber", "Dual-Role Type Match", f"{eg['dual_same_type_pct']:.2f}%", "253,120 overlap accts match BIC 99.92%, not B/C"),
    ],
    [
        "<b>Asymmetric Origination:</b> Ordering parties are 78.75% Business (B), while Beneficiary parties are ~28.82% Business across all 10M rows.",
        "<b>Strict Account-Bank Lock (Ord):</b> Every one of 391,132 ordering accounts maps 1-to-1 to a unique ORDERING_BIC8 and fixed ORDERING_TYPE.",
        "<b>Multi-BIC Collisions (Ben):</b> 61,606 beneficiary accounts link to 2–8 BIC8s (all lack IBANs, yet share Faker country/address).",
    ],
    [
        "<b>Independence Test:</b> P(Ben=B | Ord=B) = 28.84% vs P(Ben=B | Ord=C) = 28.73% (near-zero structural coupling outside B2C/B2B labels).",
        "<b>Dual-Role Artifact Test:</b> 253,120 account IDs appear as both sender & receiver; 99.92% keep the same BIC8 and country, yet only 49.60% keep the same B/C flag (coin-flip!).",
    ],
    [
        "<b>Economic Rationale:</b> Corporate treasury payroll/vendor payouts explain B2C (56%) & B2B (23%) dominance over C2C (15%) retail remittances.",
        "<b>Synthetic Generator Artifact:</b> Dual-role accounts (~49.6% B/C agreement) + multi-BIC non-IBAN IDs point to random Faker account-number reuse/re-sampling per role.",
    ],
    d2_viz
))

# D3: Temporal Dynamics & Local-Timezone Proof
d3_viz = f"""
<div class="viz-col" style="flex: 1.0;">
  <div class="viz-box">
    <img src="{imgs['c05_timezone_alignment_hours.png']}" />
    <div class="viz-cap"><b>Timezone Proof:</b> Shifting UTC hours by country integer offsets collapses 16 national curves (r=0.957–1.000).</div>
  </div>
  <div class="viz-box">
    <img src="{imgs['c06_dow_utc_vs_local.png']}" />
    <div class="viz-cap"><b>Weekday Paradox &amp; Synthetic Deviation:</b> AU Sunday / US Saturday spikes vanish locally; AE &amp; SA follow Mon–Fri rather than Sun–Thu.</div>
  </div>
</div>
<div class="viz-col" style="flex: 1.0;">
  <div class="viz-box" style="flex: 1.35;">
    <img src="{imgs['c07_daily_step_waves.png']}" />
    <div class="viz-cap"><b>Weekly Batch Regime:</b> Mon–Sun step-wave volume shifts (91k to 218k/day) do not alter median value ($3.66k) or B2B share (22.7%).</div>
  </div>
  <div class="viz-box" style="flex: 0.75; padding: 6px 8px; align-items: stretch; justify-content: flex-start;">
    <div style="font-size:8.6px; font-weight:800; color:#0F2942; margin-bottom:3px;">SELECTED TIMEZONE ALIGNMENT RESULTS (ANCHOR: GB = UTC+1 SUMMER BST)</div>
    <table class="mini">
      <thead><tr><th>Country</th><th class="num">Best UTC Offset</th><th class="num">Corr r</th><th class="num">UTC Sun%</th><th class="num">Local Sat%</th><th class="num">Local Sun%</th><th class="num">Local Fri%</th></tr></thead>
      <tbody>
        <tr><td>US</td><td class="num">UTC-4 (EDT)</td><td class="num">0.9996</td><td class="num">6.00%</td><td class="num">3.72%</td><td class="num">6.50%</td><td class="num">18.90%</td></tr>
        <tr><td>DE / FR / CH</td><td class="num">UTC+2 (CEST)</td><td class="num">0.9998</td><td class="num">6.61%</td><td class="num">3.79%</td><td class="num">6.40%</td><td class="num">18.92%</td></tr>
        <tr><td>AE / SA</td><td class="num">UTC+5 / +4</td><td class="num">0.997 / .999</td><td class="num">6.65 / 6.74%</td><td class="num">3.76%</td><td class="num">6.32 / 6.51%</td><td class="num">18.88%</td></tr>
        <tr><td>SG / CN</td><td class="num">UTC+9</td><td class="num">0.9996 / .999</td><td class="num">8.44%</td><td class="num">3.96%</td><td class="num">6.36%</td><td class="num">18.99%</td></tr>
        <tr><td>AU</td><td class="num">UTC+12</td><td class="num">0.9992</td><td class="num">10.14%</td><td class="num">4.14%</td><td class="num">6.28%</td><td class="num">19.01%</td></tr>
      </tbody>
    </table>
  </div>
</div>
"""
pages.append(render_page(
    3, "Temporal Dynamics & Local Business Calendar", "high", "Confidence: HIGH (92 Days Verified)",
    "Universal Twin-Peak Local Workday Curve Masked by UTC Timezones & Synthetic Step-Waves",
    "Scope: Apr 30 – Jul 31, 2023 (13 Weeks)",
    [
        ("blue", "Hourly Alignment Corr", "r = 0.994–1.000", "15 major countries after UTC shift (IN=0.957)"),
        ("teal", "Universal Business Peaks", "08:00 & 17–19:00", "Trough 02–05:00 local (~1.1% vs ~7.5% peak)"),
        ("amber", "UTC Weekend Illusion", "AU 10.1% -> 6.3% Sun", "Late AU Mon morning falls on UTC Sun night"),
        ("red", "Middle East Deviation", "AE/SA Fri = 18.9%", "Use Western Mon–Fri, not Sun–Thu calendar"),
        ("", "Weekly Volume Regime", "91k – 218k / day", "Sharp Mon–Sun steps; ticket & B2B invariant"),
    ],
    [
        "<b>Local-Time Clock Rule:</b> Every country's origination follows one identical twin-peak business-hour curve (~7.5% at 08:00 open & 17–19:00 cutoff) shifted by country UTC offset.",
        "<b>Universal Workweek:</b> After local-time shifting, ALL tested nations collapse to Mon 18.8%, Tue–Wed 17.0%, Thu 18.0%, Fri 19.0%, Sat 3.8%, Sun 6.4%.",
        "<b>Weekly Volume Plateaus:</b> Daily traffic stays nearly constant within each Mon–Sun ISO week and jumps only on Monday 00:00 boundaries.",
    ],
    [
        "<b>Weekend Distortion Refuted:</b> Apparent high Sunday traffic in APAC (AU 10.1%, JP 9.1%, SG 8.4%) and Saturday traffic in Americas (US 6.6%) disappears under local alignment.",
        "<b>Coupling Test with Value:</b> Week-to-week 2.4x volume swings (91k to 218k/day) do NOT alter clean median USD ($3,663) or B2B share (22.7%), proving exogenous batch scaling.",
    ],
    [
        "<b>Real-World Mechanism:</b> Treasury payment batches queue overnight and execute at local market open (08:00) and Nostro/RTGS cutoffs (17:00–19:00).",
        "<b>Synthetic Departures (Unexplained):</b> AE/SA/IL/EG follow Mon–Fri instead of real Sun–Thu banking; UTC+9/+12 shifts reflect simplified integer summer-anchor offsets.",
    ],
    d3_viz
))

# D4: Payment Value Architecture (USD_AMOUNT)
d4_viz = f"""
<div class="viz-col" style="flex: 0.95;">
  <div class="viz-box">
    <img src="{imgs['c08_ln_amount_segments.png']}" />
    <div class="viz-cap"><b>Multiplicative Party Law:</b> Clean ln(USD) means jump +2.00 (~e^2=7.39x) per Business party with constant spread (sigma=2.29).</div>
  </div>
  <div class="viz-box" style="flex: 0.55; padding: 6px 8px; justify-content: flex-start; align-items: stretch;">
    <div style="font-size:8.6px; font-weight:800; color:#0F2942; margin-bottom:3px;">CLEAN PAYMENT VALUE BY PARTY SEGMENT (EXCL. 7,006 ANOMALIES)</div>
    <table class="mini">
      <thead><tr><th>Segment</th><th class="num">Tx Count</th><th class="num">ln Mean</th><th class="num">ln Std</th><th class="num">Geom Mean</th><th class="num">Median USD</th><th class="num">Mean USD</th></tr></thead>
      <tbody>
        <tr><td><b>C2C</b> (Retail)</td><td class="num">1,512,382</td><td class="num">6.062</td><td class="num">2.292</td><td class="num">$429</td><td class="num">$464</td><td class="num">$5,018</td></tr>
        <tr><td><b>B2C</b> (Corp&rarr;Ret)</td><td class="num">5,600,965</td><td class="num">7.994</td><td class="num">2.284</td><td class="num">$2,964</td><td class="num">$3,118</td><td class="num">$34,252</td></tr>
        <tr><td><b>C2B</b> (Ret&rarr;Corp)</td><td class="num">609,762</td><td class="num">8.027</td><td class="num">2.285</td><td class="num">$3,063</td><td class="num">$3,236</td><td class="num">$35,766</td></tr>
        <tr><td><b>B2B</b> (Wholesale)</td><td class="num">2,269,885</td><td class="num">9.998</td><td class="num">2.285</td><td class="num">$21,981</td><td class="num">$22,883</td><td class="num">$252,957</td></tr>
      </tbody>
    </table>
  </div>
</div>
<div class="viz-col" style="flex: 1.35;">
  <div class="viz-box">
    <img src="{imgs['c09_amount_r2_and_countries.png']}" />
    <div class="viz-cap"><b>Causal Drivers:</b> Ordering Country accounts for +13.5 pp R2 (1,750x spread MC vs BI); Currency (+0.00001) &amp; Receiver (+0.00015) are causally null.</div>
  </div>
  <div class="viz-box">
    <img src="{imgs['c10_usd_rounding_and_eur_fx.png']}" />
    <div class="viz-cap"><b>Pre-FX Generation Proof:</b> Native USD payments show 38.6% integer amounts vs &lt;1.2% non-USD; daily EUR/USD rates step transparently in 1,000 EUR spikes.</div>
  </div>
</div>
"""
pages.append(render_page(
    4, "Payment Value Mechanics (USD_AMOUNT)", "high", "Confidence: HIGH (9.993M Clean Fit)",
    "Pre-FX Log-Normal Pricing Driven by Party Role (+2.0 ln) & Ordering Country, Converted Daily",
    "Scope: 9,992,994 Clean Transactions",
    [
        ("", "Additive Party Effect", "+2.00 ln (~7.4x)", "Per Business endpoint (C2C $464 to B2B $22.9k)"),
        ("teal", "Party + Ord Country R2", f"{pv['r2_party_ord_cc']:.4f}", f"Up from {pv['r2_party']:.4f} (+13.5 pp via sender GDP/hub)"),
        ("blue", "Currency / Ben R2 Gain", "+0.00001 / +0.00015", "Statistically & causally negligible beyond sender"),
        ("amber", "Ordering Country Range", "7.66x to 0.004x", "MC/LI/LU offshore hubs vs low-GDP corridors"),
        ("red", "USD Exact Integer Share", "38.64% vs ~0.94%", "Principal generated in CURRENCY_CODE then FX'd"),
    ],
    [
        "<b>Multiplicative Party Law:</b> Clean ln(USD) = 6.06 + 2.0*I(Ord=B) + 2.0*I(Ben=B) + Country_Ord + Noise(sigma~2.29). B2C (7.99) and C2B (8.03) are virtually identical.",
        "<b>Ordering Bank Hierarchy:</b> Party + Ordering BIC8 reaches R2=0.3855 (+3.29 pp over country); conditioning on BIC8 &amp; Account shrinks within-cell sigma from 2.29 &rarr; 1.98 &rarr; 1.82.",
        "<b>Pre-FX Principal Generation:</b> Amounts originate as round integers in local currency (~38.6% int / 15.6% %100 in USD) and convert at a daily spot schedule (e.g. 1,000 EUR mode tracks 1.11–1.20).",
    ],
    [
        "<b>Confounder Test (Currency &amp; Receiver):</b> Raw cross-currency medians vary wildly, but controlling for Ordering Country collapses +Currency delta-R2 to +0.00001 and +Ben_Country to +0.00015.",
        "<b>Alternative Hypothesis Rejected:</b> Trade bilateral pull (beneficiary wealth determining invoice size) is false here—value scales strictly on the sender's country &amp; party type.",
    ],
    [
        "<b>Economic Intuition:</b> Wealthy offshore/financial centers (MC, LI, LU, KY) &amp; B2B treasury desks issue larger ticket wires than emerging retail corridors.",
        "<b>Unexplained Residual (61.5%):</b> Pooled party sigma=2.29 decomposes into bank/account wealth heterogeneity plus irreducible within-account invoice noise (sigma=1.82).",
    ],
    d4_viz
))

# D5: Currency Choice & 100% FX Transition Law
d5_viz = f"""
<div class="viz-col" style="flex: 1.32;">
  <div class="viz-box" style="flex: 1.25;">
    <img src="{imgs['c11_currency_choice_rules.png']}" />
    <div class="viz-cap"><b>Settlement Currency Selection:</b> Same-zone flows use local currency 94.45%; cross-zone flows invoice in Ordering Home currency 88.53% (6.46% Vehicle, led by EUR/USD).</div>
  </div>
  <div class="viz-box" style="flex: 0.72; padding: 5px 8px; align-items: stretch; justify-content: flex-start;">
    <div style="font-size:8.4px; font-weight:800; color:#0F2942; margin-bottom:2px;">CURRENCY SELECTION REGIMES &amp; TOP THIRD-CURRENCY VEHICLES (10M ROWS)</div>
    <table class="mini">
      <thead><tr><th>Zone Regime / Vehicle</th><th class="num">Row Count</th><th class="num">Ord Home %</th><th class="num">Ben Home %</th><th class="num">Third Vehicle %</th></tr></thead>
      <tbody>
        <tr><td><b>Same Currency Zone</b> (Ord == Ben Zone)</td><td class="num">1,549,831</td><td class="num" colspan="2"><b>94.45%</b> Shared Home Curr</td><td class="num">5.55% Other</td></tr>
        <tr><td><b>Cross Currency Zone</b> (Ord != Ben Zone)</td><td class="num">8,450,169</td><td class="num"><b>88.53%</b></td><td class="num">5.01%</td><td class="num">6.46% (546k tx)</td></tr>
        <tr><td><b>Top Third-Currency Vehicles</b></td><td class="num" colspan="4">EUR 189,318 (34.7%) &bull; USD 107,833 (19.7%) &bull; GBP 41,887 &bull; CHF 34,991 &bull; CNY 18,374 &bull; SGD 16,420</td></tr>
      </tbody>
    </table>
  </div>
</div>
<div class="viz-col" style="flex: 1.08;">
  <div class="viz-box" style="flex: 1.25;">
    <img src="{imgs['c12_fx_transition_proof.png']}" />
    <div class="viz-cap"><b>Deterministic FX Law Proof:</b> 9,999,999 / 10,000,000 exact matches; sole apparent diff (row 4,871,844) is Eritrea (ER) modal sample vs ERN home currency.</div>
  </div>
  <div class="viz-box" style="flex: 0.72; padding: 5px 8px; align-items: stretch; justify-content: flex-start;">
    <div style="font-size:8.4px; font-weight:800; color:#0F2942; margin-bottom:2px;">FX CONVERSION DISTRIBUTION &amp; BOUNDARY RULE PROOF (ALL 10M ROWS)</div>
    <table class="mini">
      <thead><tr><th>FX Metric / Bucket</th><th class="num">Observed Count</th><th class="num">Share %</th><th>Deterministic Proof Note</th></tr></thead>
      <tbody>
        <tr><td>fx = 0 / fx = 1</td><td class="num">1,086,531 / 5,895,841</td><td class="num">10.87% / 58.96%</td><td>Same-currency hops vs 1 border jump</td></tr>
        <tr><td>fx = 2 / fx &ge; 3 (max 17)</td><td class="num">2,017,992 / 999,636</td><td class="num">20.18% / 10.00%</td><td>Zero dependency on missing '...' hops</td></tr>
        <tr><td>Empirical HomeCurr Formula Fit</td><td class="num"><b>9,999,999 / 10M</b></td><td class="num"><b>99.99999%</b></td><td>Sole diff: row 4,871,844 (0033PTP0;0000ERA0)</td></tr>
        <tr><td>Exact Rule With ER=ERN Fixed</td><td class="num"><b>10,000,000 / 10M</b></td><td class="num"><b>100.0000%</b></td><td>Matches BIC8_PATH endpoints on mismatch rows</td></tr>
      </tbody>
    </table>
  </div>
</div>
"""
pages.append(render_page(
    5, "Currency Choice & FX Conversion Rule", "high", "Confidence: EXACT (99.99999% Rule)",
    "FX Conversion Count Tracks Jurisdictional Hops Along BIC8_PATH, Ignoring Invoice Currency",
    "Scope: All 10,000,000 Rows &bull; 155 Currencies",
    [
        ("", "Dominant Currencies", "EUR 32.0% / USD 17.0%", "Top 4 (EUR, USD, GBP, CHF) = 65.8% of volume"),
        ("blue", "Cross-Zone Ord Currency", f"{fc['cross_zone_ord_home_pct']:.2f}%", f"Ben Home: {fc['cross_zone_ben_home_pct']:.2f}% | Third Vehicle: {fc['cross_zone_third_curr_pct']:.2f}%"),
        ("teal", "Same-Zone Shared Curr", f"{fc['same_zone_shared_curr_pct']:.2f}%", "e.g. Intra-Eurozone EUR or US-US USD"),
        ("teal", "FX Rule Accuracy (10M)", "9,999,999 / 10M", "100.0000% with Eritrea (ER=ERN) home currency"),
        ("amber", "Most Common FX Count", "fx = 1 (58.96%)", "fx=0: 10.87% | fx=2: 20.18% | fx>=3: 10.00%"),
    ],
    [
        "<b>Currency Choice Rule:</b> When Ord &amp; Ben belong to different currency zones (8.45M rows), 88.53% use Ord Home currency, 5.01% Ben Home, and 6.46% a third vehicle currency (EUR 189k, USD 108k).",
        "<b>Exact FX Count Law:</b> FX_COUNT = Sum_{i=0..k-2} I(HomeCurr(VisibleBIC_i) != HomeCurr(VisibleBIC_{i+1})) across ALL non-'...' hops in BIC8_PATH.",
        "<b>Counterintuitive Scope:</b> Holds identically on N=2 Direct (6.32M), N>=3 Indirect/Covered, AND QFLAG='N' gap rows.",
    ],
    [
        "<b>Invoice Currency Hypothesis Rejected:</b> A US&rarr;US transfer denominated in EUR ALWAYS records fx_count=0, while a US&rarr;CH&rarr;US transfer in USD records fx_count=2 (ignores CURRENCY_CODE!).",
        "<b>Missing-Hop ('...') Hypothesis Rejected:</b> FX transitions count ONLY adjacent visible BICs; unseen '...' gaps NEVER trigger extra FX jumps.",
        "<b>Endpoint Priority Test:</b> On the 5% start/end mismatch rows, FX_COUNT matches BIC8_PATH endpoints 100%, NOT ORDERING/BENEFICIARY_BIC8.",
    ],
    [
        "<b>Synthetic Engine Ordering:</b> The generator first fixed BIC8_PATH and derived FX_COUNT mechanically from adjacent visible bank currency zones—before applying endpoint mismatch or invoice currency!",
        "<b>Contrast vs Real SWIFT:</b> Real payments settled in USD abroad involve offshore USD Nostro clearing without mandatory FX at every border crossing; here crossing zones always increments fx_count.",
    ],
    d5_viz
))

# D6: Routing Topology & Covered Mechanics
d6_p_rows = []
for r in rt["participant_dist"]:
    if 2 <= r["participant_count"] <= 7:
        mh = r["indirect"] + r["covered"]
        cov_mh = f"{(r['covered']/mh*100):.2f}%" if mh > 0 else "0.00% (Direct)"
        d6_p_rows.append(
            f"<tr><td><b>N = {r['participant_count']}</b></td>"
            f"<td class='num'>{r['count']:,}</td>"
            f"<td class='num'>{r['pct']:.2f}%</td>"
            f"<td class='num'>{cov_mh}</td>"
            f"<td class='num'>{r['qflag_n_pct']:.2f}%</td></tr>"
        )
d6_viz = f"""
<div class="viz-col" style="flex: 0.96;">
  <div class="viz-box" style="flex: 1.25;">
    <img src="{imgs['c13_participant_geometric_law.png']}" />
    <div class="viz-cap"><b>Geometric Hop Law:</b> (N-2) matches floor(Exp(1)) with survival ratio 1/e = 0.3679 identically from N=2 to N=10+.</div>
  </div>
  <div class="viz-box" style="flex: 0.80; padding: 5px 8px; align-items: stretch; justify-content: flex-start;">
    <div style="font-size:8.4px; font-weight:800; color:#0F2942; margin-bottom:2px;">EXACT PARTICIPANT_COUNT LAW (ALL 10M ROWS, MAX OBSERVED N=22)</div>
    <table class="mini">
      <thead><tr><th>Hops (N)</th><th class="num">Row Count</th><th class="num">Share %</th><th class="num">Covered (MH%)</th><th class="num">QFLAG='N' %</th></tr></thead>
      <tbody>{''.join(d6_p_rows)}</tbody>
    </table>
  </div>
</div>
<div class="viz-col" style="flex: 1.44;">
  <div class="viz-box" style="flex: 1.25;">
    <img src="{imgs['c14_covered_bank_bimodality.png']}" />
    <div class="viz-cap"><b>Covered Control &amp; Hubs:</b> Individual Ordering BIC8 drives Covered choice (R2=0.326, span 0–95%); middle hubs gain +1.9 to +3.4 pp in home currency.</div>
  </div>
  <div class="viz-box" style="flex: 0.80; padding: 5px 8px; align-items: stretch; justify-content: flex-start;">
    <div style="font-size:8.4px; font-weight:800; color:#0F2942; margin-bottom:2px;">COVERED ROUTING R2 &amp; N=3 MIDDLE-BANK HUB CURRENCY LIFT</div>
    <table class="mini">
      <thead><tr><th>Policy / Corridor Dimension</th><th class="num">Ordering Bank / Global</th><th class="num">Beneficiary / Home Curr</th><th>Empirical Finding</th></tr></thead>
      <tbody>
        <tr><td>Covered Variance Explained (R2)</td><td class="num">Ord BIC8: <b>0.3255</b></td><td class="num">Ben: 0.1123 / Ctry: 0.0641</td><td>Bank policy drives Covered, not nation</td></tr>
        <tr><td>Bank Covered Quantiles (p10/p50/p90/p99)</td><td class="num">0.5% / 2.7% / 17.8% / 89.4%</td><td class="num">Ben p50: 7.9% / p99: 94.6%</td><td>Bimodal split (e.g. 0130ESM0 94.9%)</td></tr>
        <tr><td>Top N=3 Intermediary: <b>US</b> / <b>CH</b></td><td class="num">17.98% / 10.51% overall</td><td class="num"><b>21.39%</b> USD / <b>12.52%</b> CHF</td><td>+3.41 pp US / +2.01 pp CH clearing pull</td></tr>
        <tr><td>Other N=3 Hubs: <b>GB</b> / <b>CN</b></td><td class="num">6.31% / 4.12% overall</td><td class="num"><b>8.18%</b> GBP / <b>6.96%</b> CNY</td><td>+1.87 pp GB / +2.85 pp CN currency pull</td></tr>
      </tbody>
    </table>
  </div>
</div>
"""
pages.append(render_page(
    6, "Routing Topology & Covered Mechanics", "high", "Confidence: HIGH (Exact Survival Law)",
    "Memoryless Geometric Hop Length (1/e Ratio) Meets Bank-Specific Covered Routing Policy",
    "Scope: All 10,000,000 Routes &bull; Direct/Indirect/Covered",
    [
        ("", "Direct &hArr; N = 2 Threshold", "6,323,280 (63.23%)", "Exact P = 1 - 1/e = 63.21%; 0 Directs at N>=3"),
        ("blue", "Multi-Hop Survival Ratio", "0.3676–0.3685", "Matches theoretical 1/e = 0.3679 (floor(Exp(1)))"),
        ("teal", "Covered Share (N>=3)", "11.31% (415,962)", "Invariant to N (11.29–11.33%), driven by Ord BIC8"),
        ("amber", "Covered R2: Ord BIC vs Ctry", "0.326 vs 0.064", "Bank p10=0.5%, med=2.7%, p90=17.8%, p99=89.4%"),
        ("", "Top N=3 Clearing Hubs", "US 18.0% / CH 10.5%", "+3.4 pp US USD &amp; +2.8 pp CN CNY home bias"),
    ],
    [
        "<b>Exact TX_TYPE Partition:</b> TRANSACTION_TYPE_CODE == 'Direct' IF AND ONLY IF PARTICIPANT_COUNT == 2 (100% deterministic, max observed N=22).",
        "<b>Geometric Hop Generator:</b> PARTICIPANT_COUNT = 2 + floor(Exp(1)), invariant across B2B/C2C (ratio 36.72–36.81%) and domestic vs cross-border (36.75% vs 36.77%).",
        "<b>Covered Bank Policy:</b> Among N>=3, Covered is decided largely by Ordering BIC8 (e.g. 0130ESM0 94.9%, 0360US30 94.8% vs median bank 2.7%).",
    ],
    [
        "<b>Country-Effect Illusion Resolved:</b> Isle of Man (IM) appeared 50.4% Covered in raw country aggregates ONLY because one dominant bank (0004IMD0, 58.3% Covered) sits in IM.",
        "<b>Hop-Length Invariance:</b> Unlike real networks where longer routes trigger different cover rules, Covered share stays flat at 11.3% across N=3, 4, 5, 6.",
    ],
    [
        "<b>Correspondent Banking Logic:</b> USD/CH/GB/DE clearing hubs dominate N=3 intermediaries, with home-currency liquidity boosting middle-bank selection by +1.9 to +3.4 pp.",
        "<b>Synthetic Simplification:</b> Real SWIFT domestic corridors rarely exhibit 36.8% multi-hop routing—the generator samples hop count memorylessly before routing.",
    ],
    d6_viz
))

# D7: Path Continuity (QFLAG) & Endpoint Mismatches
dots_lu = {(r["participant_count"], r["n_dots"]): r for r in rt["dots_fit"]}
pdist_lu = {r["participant_count"]: r for r in rt["participant_dist"]}
d7_dot_rows = []
for n_val in (2, 3, 4, 5):
    tot_n = pdist_lu[n_val]["count"]
    d0 = dots_lu[(n_val, 0)]
    d1 = dots_lu[(n_val, 1)]
    d2 = dots_lu[(n_val, 2)]
    d7_dot_rows.append(
        f"<tr><td><b>N={n_val}</b> ({n_val-1} edges)</td>"
        f"<td class='num'>{tot_n:,}</td>"
        f"<td class='num'>{d0['obs_pct']:.2f}% ({d0['theo_pct']:.2f}%)</td>"
        f"<td class='num'>{d1['obs_pct']:.2f}% ({d1['theo_pct']:.2f}%)</td>"
        f"<td class='num'>{d2['obs_pct']:.2f}% ({d2['theo_pct']:.2f}%)</td></tr>"
    )
d7_viz = f"""
<div class="viz-col" style="flex: 1.02;">
  <div class="viz-box" style="flex: 1.25;">
    <img src="{imgs['c15_dots_negbin_and_endpoints.png']}" />
    <div class="viz-cap"><b>Two Independent Path Noise Processes:</b> NegBin(N-1, 0.95) missing-intermediary '...' tokens + independent 5.0% endpoint BIC mismatches.</div>
  </div>
  <div class="viz-box" style="flex: 0.72; padding: 5px 8px; align-items: stretch; justify-content: flex-start;">
    <div style="font-size:8.4px; font-weight:800; color:#0F2942; margin-bottom:2px;">NEGBIN(N-1, 0.95) MISSING-HOP FIT: OBSERVED VS THEORETICAL (%)</div>
    <table class="mini">
      <thead><tr><th>Route Length</th><th class="num">Total Tx</th><th class="num">0 Gaps (Theo)</th><th class="num">1 Gap (Theo)</th><th class="num">2 Gaps (Theo)</th></tr></thead>
      <tbody>{''.join(d7_dot_rows)}</tbody>
    </table>
  </div>
</div>
<div class="viz-col" style="flex: 0.98;">
  <div class="viz-box" style="flex: 1.25;">
    <img src="{imgs['c16_qflag_confounder_null.png']}" />
    <div class="viz-cap"><b>Simpson's Confounder Resolved:</b> Apparent +51s lag for QFLAG='N' collapses to &lt;9s noise once stratified by PARTICIPANT_COUNT N.</div>
  </div>
  <div class="viz-box" style="flex: 0.72; padding: 5px 8px; align-items: stretch; justify-content: flex-start;">
    <div style="font-size:8.4px; font-weight:800; color:#0F2942; margin-bottom:2px;">ENDPOINT MISMATCHES &amp; DURATION CONFOUNDING BREAKDOWN</div>
    <table class="mini">
      <thead><tr><th>Path Anomaly / Control Test</th><th class="num">10M Count / Effect</th><th class="num">Baseline / Expected</th><th>Causal Conclusion</th></tr></thead>
      <tbody>
        <tr><td>Start Mismatch (Path[0] != Ord)</td><td class="num">498,399 (<b>4.984%</b>)</td><td class="num">ROM=0.0% | CEO=6.1%</td><td>IBAN, Acct, Details &amp; Value match Ord BIC8 100%</td></tr>
        <tr><td>End Mismatch (Path[-1] != Ben)</td><td class="num">499,265 (<b>4.993%</b>)</td><td class="num"><b>0.00%</b> in ROM/CEO/ARS</td><td>FX_COUNT &amp; Duration match Path[0..-1] 100%</td></tr>
        <tr><td>Duration on Mismatch (Path vs BIC)</td><td class="num"><b>MAE 5.60s</b> (R2=.99997)</td><td class="num">BIC8: MAE 300.3s (R2=.978)</td><td>Proves Path[0..-1] is true settlement layer!</td></tr>
        <tr><td>QFLAG='N' Duration (Pooled vs By-N)</td><td class="num"><b>+51s</b> pooled N=3..6</td><td class="num"><b>&lt;9s</b> per-N stratum</td><td>Zero causal delay; pure N confounding</td></tr>
      </tbody>
    </table>
  </div>
</div>
"""
pages.append(render_page(
    7, "Path Continuity (QFLAG) & Endpoint Mismatches", "high", "Confidence: HIGH (0 Mismatches)",
    "Missing Tracking Hops Follow Edge NegBin(N-1, 0.95) Law With Zero Causal Latency Impact",
    "Scope: 767,204 Gap Paths &bull; ~5% Endpoint Noise",
    [
        ("teal", "QFLAG='N' &hArr; '...' Rule", "767,204 (0 errs)", "100.00% equivalence; non-dot BICs == N (0 errs)"),
        ("blue", "Per-Edge Gap Mechanism", "Geom(p=0.05)", "N=2: 4.99% | N=3: 9.74% | N=4: 14.25% | N=5: 18.50%"),
        ("amber", "Start & End Mismatch", "4.98% &amp; 4.99%", "Joint 0.247% vs 0.249% independent (~95% cross-border)"),
        ("", "Two-Layer Authority Split", "Path MAE = 5.60s", "BIC8 cols rule Acct/IBAN/USD; Path rules FX &amp; Dur"),
        ("red", "Duration Confounder", "+51s -> ~0s", "QFLAG='N' lag is 100% mediated by longer N"),
    ],
    [
        "<b>Exact Visibility Rule:</b> CONTINUITY_QFLAG == 'N' iff '...' in BIC8_PATH (0/10M errors). Between every consecutive pair of the N visible BICs (N-1 edges), n_dots ~ Geom(0.05) -> total dots ~ NegBin(N-1, 0.95).",
        "<b>Additive Non-Dot Participant Rule:</b> Excluding '...', visible BIC token count equals PARTICIPANT_COUNT on 10,000,000 / 10,000,000 rows ('...' sits between visible endpoints, never replaces them).",
        "<b>5% Endpoint Mismatch Injection:</b> BIC8_PATH[0] != ORDERING_BIC8 (4.984%) and BIC8_PATH[-1] != BENEFICIARY_BIC8 (4.993%) occur independently (~95% cross-border).",
    ],
    [
        "<b>Two-Layer Authority Split Proven:</b> Customer fields (Account, IBAN, Details, USD multiplier) match ORDERING/BENEFICIARY_BIC8 100%, whereas interbank settlement (FX_COUNT 100% and TOTAL_DURATION R2=0.99997, MAE=5.60s vs 300.3s) obeys BIC8_PATH[0..-1]!",
        "<b>Spurious Latency Correlation Refuted:</b> QFLAG='N' appears slower globally ONLY because each added edge adds another 5% gap chance AND another bank delay!",
    ],
    [
        "<b>SWIFT gpi &amp; Respondent Banking Analogy:</b> Incomplete UETR tracking causes QFLAG='N', while endpoint mismatches represent indirect respondent banks entering/exiting SWIFT via cross-border correspondents (Path[0]/Path[-1]).",
        "<b>Origin Difference (Scams vs Normal):</b> Social scams have 0.0% end mismatch (and Romance 0.0% start mismatch) because fraud generators write directly to destination mules without respondent aliasing.",
    ],
    d7_viz
))

# D8: Settlement Duration Mechanics
d8_viz = f"""
<div class="viz-col" style="flex: 1.05;">
  <div class="viz-box">
    <img src="{imgs['c17_duration_trimodal_and_pairs.png']}" />
    <div class="viz-cap"><b>Pair Determinism + Gamma(2,5):</b> Global trimodality hides tight within-pair std=7.04s (R2=0.99996) with Gamma(2,5) positive queue noise.</div>
  </div>
</div>
<div class="viz-col" style="flex: 1.05;">
  <div class="viz-box" style="flex: 1.25;">
    <img src="{imgs['c18_duration_n3_and_covered_speedup.png']}" />
    <div class="viz-cap"><b>Multi-Hop &amp; Covered Speedup:</b> N=3 duration tracks A,B,C bank structure (R2=0.992); Covered cuts same-bank latency 29–38%.</div>
  </div>
  <div class="viz-box" style="flex: 0.6; padding: 5px 8px; align-items: stretch; justify-content: flex-start;">
    <div style="font-size:8.4px; font-weight:800; color:#0F2942; margin-bottom:2px;">ASYMMETRIC PAIR JUMP EXAMPLES (DIRECT CLEAN) &amp; NOISE TEST</div>
    <table class="mini">
      <thead><tr><th>Directed Pair Example</th><th class="num">A&rarr;B Mean</th><th class="num">B&rarr;A Mean</th><th class="num">Pair Min</th><th class="num">Excess Mean / Var</th></tr></thead>
      <tbody>
        <tr><td>0001MQM0 &rarr; 0009TNT0 (n=12,640)</td><td class="num">1,595.8s</td><td class="num">&mdash;</td><td class="num">1,586s</td><td class="num">9.81s / 47.20</td></tr>
        <tr><td>0032MYK0 &rarr; 0010ALT0 (n=10,623)</td><td class="num">27.0s</td><td class="num">&mdash;</td><td class="num">17s</td><td class="num">9.99s / 50.52</td></tr>
        <tr><td>0309CHZ0 &harr; 0035AU20 (Asym Jump)</td><td class="num">631.6s</td><td class="num">30.1s</td><td class="num">&mdash;</td><td class="num">~600s Discrete Jump</td></tr>
        <tr><td>0026QAQ0 &harr; 0043BHB0 (Asym Jump)</td><td class="num">27.1s</td><td class="num">629.0s</td><td class="num">&mdash;</td><td class="num">~600s Discrete Jump</td></tr>
      </tbody>
    </table>
  </div>
</div>
"""
pages.append(render_page(
    8, "Settlement Duration Architecture (TOTAL_DURATION)", "high", "Confidence: HIGH (R2=0.99996 Direct)",
    "Deterministic Directed-Pair Base Delays + Shifted Gamma(2,5) Noise; Covered Saves ~35%",
    "Scope: 9.993M Clean Tx (2s to 21,649s)",
    [
        ("", "Direct Pair FE R2 / RMSE", f"{dur['direct_pair_r2_n_ge_20']:.5f} / {dur['direct_pair_rmse_n_ge_20']:.2f}s", "Additive bank FE R2=0.9933 (RMSE~91s)"),
        ("blue", "Noise Law (All Pairs)", "Gamma(k=2, th=5)", "Obs excess mean ~9.9s (10s), var ~49 (50)"),
        ("teal", "Covered Speedup (N=3)", "0.62x – 0.71x", "~35% faster at same Ordering BIC8 &amp; N=3"),
        ("amber", "Reverse-Pair Symmetry", f"{dur['sym_within_10s_pct']:.1f}% <= 10s", "Med |diff|=1.57s; discrete ~600s asym jumps"),
        ("", "N=3 (A,B,C) FE Model R2", f"{dur['r2_p3_ind_A_B_C']:.4f} (vs {dur['r2_p3_ind_A_C_only']:.3f})", "Dropping middle bank B loses 20.5 pp R2!"),
    ],
    [
        ("<b>Direct Law:</b> TOTAL_DURATION(u&rarr;v) = Base(Path[0], Path[-1]) + Gamma(2,5) noise (R2=0.999972, RMSE=6.01s on Path endpoints; median within-pair std = 7.04s vs theoretical sqrt(50) = 7.07s!)."),
        "<b>Multi-Hop Composition:</b> N=3 Indirect tracks D(A,B)+D(B,C) with r=0.982 (median -17.7s due to single 10s noise draw &amp; node sharing); Covered cuts latency ~35%.",
        "<b>Bilateral Symmetry &amp; Directed Jumps:</b> 72.7% of bidirectional pairs match within 10s (med 1.57s), while ~27.3% show discrete ~100s–600s bank-pair directional offsets.",
    ],
    [
        "<b>Time-of-Day Congestion Refuted:</b> Hourly duration within fixed corridors (US&rarr;US 42–44s, US&rarr;SG 33–36s, DE&rarr;FR 40–43s) is completely flat across UTC 0–23.",
        "<b>Multiplicative &amp; Country-Diff Hypotheses Rejected:</b> Log-space FE drops R2 to 0.863; within the same country pair, directional diffs have std=199.4s (proving bank-pair edge weights, not timezone diffs).",
    ],
    [
        "<b>Banking SLA Interpretation:</b> Base latencies reflect bilateral settlement rail tiers (~25s instant/ISO20022, ~400s batch, ~2,000s manual compliance/timezone delay); Covered parallelizes cover vs MT103.",
        "<b>Residual Open Question:</b> Why specific directed bank pairs (e.g. 0309CHZ0&rarr;0035AU20) draw +600s asymmetric edge weights reflects unexposed bilateral SLA tier assignment tables.",
    ],
    d8_viz
))

# D9: Party Details & Tokenization Grammar
d9_viz = f"""
<div class="viz-col" style="flex: 1.05;">
  <div class="viz-box">
    <img src="{imgs['c19_details_country_mismatch.png']}" />
    <div class="viz-cap"><b>Asymmetric Faker Geo-Coupling:</b> Ordering text country is 100% tied to BIC8 country, while Beneficiary text ignores BIC8 country 99.0% of the time.</div>
  </div>
</div>
<div class="viz-col" style="flex: 1.05;">
  <div class="viz-box" style="flex: 1.2;">
    <img src="{imgs['c20_iban_structure_finwords.png']}" />
    <div class="viz-cap"><b>Format Grammar:</b> Strict jurisdiction IBAN flags + 10 uniform &lt;finance_word&gt; keywords injected ONLY inside unstructured text (~9.42%).</div>
  </div>
  <div class="viz-box" style="flex: 0.72; padding: 5px 8px; align-items: stretch; justify-content: flex-start;">
    <div style="font-size:8.4px; font-weight:800; color:#0F2942; margin-bottom:2px;">ORDERING VS. BENEFICIARY TEXT TOKEN RATES (10,000,000 ROWS)</div>
    <table class="mini">
      <thead><tr><th>Feature / Token Indicator</th><th class="num">Ordering Rate</th><th class="num">Beneficiary Rate</th><th>Observed Mechanism</th></tr></thead>
      <tbody>
        <tr><td>Inferred Text Country == BIC8 CC</td><td class="num"><b>100.00%</b></td><td class="num"><b style="color:#DC2626">0.98%</b></td><td>Ben address locale sampled independently</td></tr>
        <tr><td>IBAN Account Format Present</td><td class="num">{pd_m['ord_iban_pct']:.2f}%</td><td class="num">{pd_m['ben_iban_pct']:.2f}%</td><td>EU/UK/CH 90–96% vs US/CN/JP/SG/AU/IM 0%</td></tr>
        <tr><td>35-Char Line Wrap (&lt;line_break:35&gt;)</td><td class="num">{pd_m['ord_lb35_pct']:.2f}%</td><td class="num">{pd_m['ben_lb35_pct']:.2f}%</td><td>SWIFT FIN 4x35x field wrapping rule</td></tr>
        <tr><td>Structured 1/ 2/ 3/ Tags (ISO/MT)</td><td class="num">{pd_m['ord_struct123_pct']:.2f}%</td><td class="num">{pd_m['ben_struct123_pct']:.2f}%</td><td>0.002% &lt;finance_word&gt; in structured lines</td></tr>
        <tr><td>&lt;finance_word&gt; in Unstructured Text</td><td class="num">{pd_m['ord_finword_in_unstruct_pct']:.2f}%</td><td class="num">{pd_m['ben_finword_in_unstruct_pct']:.2f}%</td><td>10 uniform words (~159k total mentions each)</td></tr>
      </tbody>
    </table>
  </div>
</div>
"""
pages.append(render_page(
    9, "Party Details & Tokenization Grammar", "high", "Confidence: HIGH (Full Text Parse)",
    "Faker Locale Disconnects Beneficiary Addresses (0.98% Match) While Preserving IBAN & Wrapping Rules",
    "Scope: 20M Detail/Token Strings",
    [
        ("teal", "Ord Text Geo-Match", "100.00%", "Ordering addresses strictly match ORDERING_BIC8 CC"),
        ("red", "Ben Text Geo-Match", "0.984%", "Skews to CN 14.5%, BR 8.9%, UA 7.2%, ET 5.8%"),
        ("blue", "SWIFT 35-Char Wrap Rate", "44.0% / 44.2%", "Authentic 35-char FIN line-wrap constraint"),
        ("", "Structured 1/ 2/ 3/ Share", "16.0% / 15.0%", "Name/Addr omitted in 11.8% Ord / 10.0% Ben"),
        ("amber", "Finance Words Vocab", "Exactly 10 Words", "9.42% unstructured vs 0.002% structured 1/2/3/"),
    ],
    [
        "<b>Dual Formatting Pipeline:</b> ~15–16% use structured 1/Name 2/Street 3/Country/City lines; ~10–12% omit address (BIC/acct only); ~73% use unstructured multi-line Faker blocks wrapped at 35 chars.",
        "<b>Jurisdiction IBAN Law:</b> IBANs appear in 90.3–96.1% of European/SEPA+CH/GB rows and 0.000% of US/CN/JP/SG/AU/HK/CA/IM rows (100% match BIC country when present).",
        "<b>Closed Keyword Injection:</b> Unstructured free-text details add one of 10 uniform words (gross..funds, ~79k–80k each side) at 9.42% rate.",
    ],
    [
        "<b>Corroboration vs Faker Locale Artifact:</b> When Beneficiary has an IBAN (~49.8%), the IBAN country prefix matches BENEFICIARY_BIC8 100%, yet the Faker street/city/country text aligns only 0.98%—proving Faker initialized beneficiary text locale independently!",
        "<b>Truncation Noise Check:</b> &lt;truncation&gt; appears rarely (0.001% Ord, 0.003% Ben, 387 total tags).",
    ],
    [
        "<b>Realistic Compliance Context:</b> Authentic SWIFT MT50F/59F 1/2/3/ structured tags vs messy free text (and 35-char line wrapping) drive real-world sanctions-screening false positives.",
        "<b>Critical Data Warning:</b> NLP geo-screening features built on BENEFICIARY_DETAILS will fail 99% accuracy relative to BENEFICIARY_BIC8 in this synthetic corpus.",
    ],
    d9_viz
))

# D10: Anomaly Mechanics, Mule Pool & Undocumented Recall
d10_viz = f"""
<div class="viz-col" style="flex: 1.0;">
  <div class="viz-box">
    <img src="{imgs['c21_romance_scam_steps.png']}" />
    <div class="viz-cap"><b>Romance Grooming:</b> 500 C-victims, consecutive bursts (med 215s), median $33 &rarr; $1,925.</div>
  </div>
  <div class="viz-box">
    <img src="{imgs['c23_fat_finger_and_scam_routing.png']}" />
    <div class="viz-cap"><b>Pre-FX Typing Errors &amp; Scam Covered Drop:</b> 10^k shifts + 6–37x Covered drop in scams.</div>
  </div>
</div>
<div class="viz-col" style="flex: 1.0;">
  <div class="viz-box">
    <img src="{imgs['c22_mule_pool_venn_countries.png']}" />
    <div class="viz-cap"><b>Shared Mule Pool:</b> 1,068 accounts cross-shared by CEO/ARS/ROM (+7,828 normal txs), skewed 34–211x to EM hubs.</div>
  </div>
  <div class="viz-box">
    <img src="{imgs['c24_customer_recall_null.png']}" />
    <div class="viz-cap"><b>Undocumented CUSTOMER_RECALL Null Test:</b> KS p=0.23–0.90 &amp; LGBM AUC=0.476 (random recall flag).</div>
  </div>
</div>
"""
pages.append(render_page(
    10, "Anomaly Taxonomy, Mule Graph & Recall Null Test", "mixed", "Confidence: HIGH (5 Known) / NULL (Recall)",
    "Six Distinct Defect Mechanisms: Grooming Chains, Shared Mules, 10^k Typing Errors & Invisible Recalls",
    "Scope: 7,006 Labeled Anomalies (6 Classes)",
    [
        ("red", "ROMANCE_SCAM (2,476)", "1.50x/step | 0% Cov", "500 victims (100% Ord=C), 2–8 consecutive bursts"),
        ("blue", "FAT_FINGER (1,940)", "77.5% USD Ints", "10^k pre-FX shifts (+1.57 log10), max $15.66B"),
        ("red", "CEO (945) &amp; ARS (649)", "Shared 1,068 Mules", "CEO=100% B (3.2x); ARS=normal amt, 0 prior pair"),
        ("amber", "DUPLICATE (679)", "185 Consec. Bursts", "Size 2:37, 3:43, 4:49, 5:56; exact excl. epoch (+44 normal)"),
        ("", "CUSTOMER_RECALL* (317)", f"AUC = {an['customer_recall_lgbm_auc']:.3f}", "Undocumented; indistinguishable (KS p=0.23–0.90)"),
    ],
    [
        "<b>Scam Signature Separation:</b> ROMANCE (0% start/end mis, 0.3% Covered mh); CEO &amp; ARS (normal ~5–6% start_mis, 0.0% end_mis, 1.5–2.0% Covered mh, target shared 1,068-acct EM mule pool).",
        "<b>Operational Defects:</b> FAT_FINGER doubles USD integer rate (77.5% vs 38.6%); DUPLICATE emits 2–5 consecutive exact copies within ~2 hrs (med dt=1,099s).",
    ],
    [
        "<b>Covered Confounder Ruled Out:</b> Social-engineering Covered drops 6–37x even AFTER conditioning on N>=3 (not just higher Direct share!), driven by mule destination &amp; bank selection.",
        "<b>Recall Negative Result:</b> Extensive KS tests &amp; 16-feature 5-fold LightGBM yield OOF AUC=0.476; only marginal QFLAG='N' shift (11.0% vs 7.6%, p=0.023).",
    ],
    [
        "<b>AML / Fraud Insight:</b> Same-mule reuse across CEO/ARS/Romance (+7,828 'Normal' hits to 441 mules) reflects money-mule networks laundering multi-typology proceeds; ARS bypasses amount rules so new-beneficiary graph checks are vital.",
        "<b>Recall Reality:</b> Customer recalls often stem from off-network invoice disputes absent from interbank fields.",
    ],
    d10_viz
))

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>UMDAC Datathon 2026 - Standalone Dashboards (Swift Case Study)</title>
<style>{CSS}</style>
</head>
<body>
{''.join(pages)}
</body>
</html>
"""

html_path = os.path.abspath("work/dashboards.html")
pdf_path = os.path.abspath("Swift_Datathon_2026_Dashboards.pdf")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(full_html)
print(f"Wrote {html_path} ({len(full_html)/1024:.1f} KB)")

# Find Chrome or Edge cleanly
candidates = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]
browser = next((c for c in candidates if os.path.exists(c)), shutil.which("chrome") or shutil.which("msedge"))
if not browser:
    raise RuntimeError("No headless Chromium/Chrome/Edge binary found!")
print("Rendering PDF via:", browser)

url = Path(html_path).as_uri()
cmd = [
    browser,
    "--headless=new",
    "--disable-gpu",
    "--no-sandbox",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    url,
]
res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
if res.returncode != 0:
    print("STDERR:", res.stderr)
    raise RuntimeError(f"Browser PDF print failed with {res.returncode}")
print(f"Created PDF: {pdf_path} ({os.path.getsize(pdf_path)/1024/1024:.2f} MB)")
