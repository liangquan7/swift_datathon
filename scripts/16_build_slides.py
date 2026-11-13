import base64
import json
import os
import shutil
import subprocess
from pathlib import Path

with open("work/master_metrics.json", "r", encoding="utf-8") as f:
    M = json.load(f)
with open("work/audit_summary.json", "r", encoding="utf-8") as f:
    A = json.load(f)


def b64_img(rel_path: str) -> str:
    with open(rel_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


imgs = {p.name: b64_img(str(p)) for p in sorted(Path("work/charts").glob("*.png"))}

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
  size: 13.333in 7.5in;
  margin: 0;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', Arial, Helvetica, sans-serif;
  color: #0F172A;
  background: #CBD5E1;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
.slide {
  width: 13.333in;
  height: 7.5in;
  background: #F8FAFC;
  padding: 0.28in 0.35in 0.22in 0.35in;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  page-break-after: always;
  overflow: hidden;
  position: relative;
}
.slide:last-child { page-break-after: auto; }
.slide.dark {
  background: linear-gradient(135deg, #091B2E 0%, #0F2942 58%, #153B5E 100%);
  color: #F8FAFC;
}

/* Header */
.shdr {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  border-bottom: 2.5px solid #0F2942;
  padding-bottom: 6px;
  margin-bottom: 8px;
  flex-shrink: 0;
}
.slide.dark .shdr { border-bottom-color: rgba(255,255,255,0.22); }
.eyebrow {
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: #2563EB;
  margin-bottom: 2px;
}
.slide.dark .eyebrow { color: #38BDF8; }
.stitle {
  font-size: 18px;
  font-weight: 800;
  color: #0F2942;
  line-height: 1.15;
}
.slide.dark .stitle { color: #FFFFFF; }
.pill {
  font-size: 8.5px;
  font-weight: 700;
  padding: 3px 9px;
  border-radius: 999px;
  border: 1px solid #CBD5E1;
  background: #FFFFFF;
  color: #334155;
  white-space: nowrap;
  margin-left: 12px;
}
.pill.fg { background: #EFF6FF; border-color: #93C5FD; color: #1D4ED8; }
.pill.syn { background: #ECFDF5; border-color: #6EE7B7; color: #047857; }
.pill.warn { background: #FEF2F2; border-color: #FCA5A5; color: #B91C1C; }

/* Main body */
.sbody {
  flex: 1;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.row { display: flex; gap: 10px; min-height: 0; }
.col { display: flex; flex-direction: column; gap: 8px; min-height: 0; }

/* Cards */
.scard {
  background: #FFFFFF;
  border: 1px solid #CBD5E1;
  border-radius: 7px;
  padding: 8px 10px;
}
.scard.navy { border-top: 3px solid #0F2942; }
.scard.blue { border-top: 3px solid #2563EB; }
.scard.teal { border-top: 3px solid #0D9488; }
.scard.amber { border-top: 3px solid #D97706; }
.scard.red { border-top: 3px solid #DC2626; }
.scard.rule {
  background: #F1F5F9;
  border-left: 4px solid #0F2942;
  border-top: 1px solid #CBD5E1;
  padding: 7px 10px;
}
.scard h3 {
  font-size: 10px;
  font-weight: 800;
  color: #0F2942;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 4px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.tag {
  font-size: 7.5px;
  padding: 1.5px 5px;
  border-radius: 3px;
  color: #FFFFFF;
  font-weight: 800;
}
.t-obs { background: #0F2942; }
.t-test { background: #D97706; }
.t-int { background: #0D9488; }
.t-gap { background: #DC2626; }
.t-act { background: #2563EB; }

.scard ul { list-style: none; font-size: 9.6px; line-height: 1.36; color: #1E293B; }
.scard ul li { margin-bottom: 3.5px; padding-left: 10px; position: relative; }
.scard ul li:last-child { margin-bottom: 0; }
.scard ul li::before { content: "▪"; position: absolute; left: 0; color: #2563EB; font-weight: 900; font-size: 9px; }
.scard b { color: #0F2942; }
.slide.dark .scard.dark-panel b { color: #FFFFFF; }
.slide.dark .scard.dark-panel code { color: #93C5FD; font-weight: 700; }
.eq {
  font-family: Consolas, 'Courier New', monospace;
  font-size: 9.4px;
  background: #FFFFFF;
  border: 1px solid #CBD5E1;
  padding: 3px 6px;
  border-radius: 4px;
  color: #0F2942;
  font-weight: 700;
  margin-bottom: 4px;
  display: block;
}

/* Viz box */
.simg-box {
  background: #FFFFFF;
  border: 1px solid #CBD5E1;
  border-radius: 7px;
  padding: 5px 7px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  min-height: 0;
  overflow: hidden;
}
.simg-box img {
  max-width: 100%;
  max-height: 100%;
  width: auto;
  height: auto;
  object-fit: contain;
  display: block;
}
.scap {
  font-size: 8.5px;
  color: #475569;
  margin-top: 3px;
  text-align: center;
  line-height: 1.2;
  flex-shrink: 0;
}

/* Tables */
table.stbl {
  width: 100%;
  border-collapse: collapse;
  font-size: 8.6px;
}
table.stbl th {
  background: #0F2942;
  color: #FFFFFF;
  padding: 4px 6px;
  text-align: left;
  font-weight: 700;
}
table.stbl td {
  padding: 3.5px 6px;
  border-bottom: 1px solid #E2E8F0;
  color: #1E293B;
}
table.stbl tr:nth-child(even) td { background: #F8FAFC; }
.num { text-align: right; font-variant-numeric: tabular-nums; }

/* Dark cover elements */
.hero-kpis {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 10px;
  margin: 10px 0 12px 0;
}
.hkpi {
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.16);
  border-left: 3.5px solid #38BDF8;
  border-radius: 6px;
  padding: 8px 10px;
}
.hkpi-l { font-size: 8px; text-transform: uppercase; letter-spacing: 0.06em; color: #CBD5E1; font-weight: 700; }
.hkpi-v { font-size: 16px; font-weight: 800; color: #FFFFFF; margin: 2px 0; }
.hkpi-s { font-size: 8.3px; color: #94A3B8; }
.law-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 9px;
  flex: 1;
  min-height: 0;
}
.law-card {
  background: rgba(255,255,255,0.96);
  color: #0F172A;
  border-radius: 7px;
  padding: 10px;
  border-top: 4px solid #2563EB;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.law-num {
  font-size: 9px;
  font-weight: 800;
  color: #2563EB;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.law-title { font-size: 11.5px; font-weight: 800; color: #0F2942; margin-bottom: 5px; line-height: 1.2; }
.law-stat { font-size: 17px; font-weight: 900; color: #0D9488; margin: 4px 0; }
.law-body { font-size: 9.2px; line-height: 1.38; color: #334155; flex: 1; }
.law-foot { font-size: 8.2px; color: #64748B; border-top: 1px solid #E2E8F0; padding-top: 4px; margin-top: 6px; font-weight: 600; }

/* Footer */
.sftr {
  display: flex;
  justify-content: space-between;
  font-size: 8.2px;
  color: #64748B;
  border-top: 1px solid #CBD5E1;
  padding-top: 4px;
  margin-top: 6px;
  flex-shrink: 0;
}
.slide.dark .sftr {
  color: #94A3B8;
  border-top-color: rgba(255,255,255,0.18);
}
"""


def slide_wrap(num: int, eyebrow: str, title: str, badge_txt: str, badge_cls: str, body_html: str, dark: bool = False) -> str:
    cls = "slide dark" if dark else "slide"
    return f"""
    <div class="{cls}">
      <div>
        <div class="shdr">
          <div>
            <div class="eyebrow">{eyebrow}</div>
            <div class="stitle">{title}</div>
          </div>
          <span class="pill {badge_cls}">{badge_txt}</span>
        </div>
      </div>
      <div class="sbody">
        {body_html}
      </div>
      <div class="sftr">
        <span>UMDAC Datathon 2026 (Preliminary Round) &bull; Exclusive Analysis of Provided Synthetic Swift Dataset (10,000,000 Rows, 21 Cols)</span>
        <span>Slide {num} / 16</span>
      </div>
    </div>
    """


slides = []

# SLIDE 1: Executive Title & Top 5 Reverse-Engineered System Laws
s1_body = f"""
<div style="background: rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.15); border-radius:7px; padding:10px 14px;">
  <div style="font-size:11px; color:#E2E8F0; line-height:1.42;">
    <b>Executive Objective:</b> Reconstruct the underlying payment, routing, pricing, latency, text-formatting, and fraud-injection rules governing the 4.12 GB fictitious SWIFT dataset&mdash;separating <b>economic behavior</b>, <b>causal confounders</b>, <b>hardcoded generator mechanics</b>, and <b>unexplained variance</b> without external data.
  </div>
</div>
<div class="hero-kpis">
  <div class="hkpi"><div class="hkpi-l">Full Census Analyzed</div><div class="hkpi-v">10,000,000</div><div class="hkpi-s">Rows &bull; 0 core nulls &bull; 92 days (May–Jul 2023)</div></div>
  <div class="hkpi"><div class="hkpi-l">Global Payment Graph</div><div class="hkpi-v">8,686 BIC8s</div><div class="hkpi-s">391k Ord &amp; 2.59M Ben Accts &bull; 229 Nations</div></div>
  <div class="hkpi"><div class="hkpi-l">Currencies Recovered</div><div class="hkpi-v">155 Units</div><div class="hkpi-s">EUR 32.0%, USD 17.0%, GBP 8.7%, CHF 8.0%</div></div>
  <div class="hkpi"><div class="hkpi-l">FX Rule Exactness</div><div class="hkpi-v">9,999,999 / 10M</div><div class="hkpi-s">100.000% with Eritrea (ER=ERN) zone</div></div>
  <div class="hkpi" style="border-left-color:#F59E0B;"><div class="hkpi-l">Anomalies &amp; Hidden</div><div class="hkpi-v">7,006 (6 Types)</div><div class="hkpi-s">+ Undocumented row-0 DISCLAIMER &amp; RECALL</div></div>
</div>
<div class="law-grid">
  <div class="law-card">
    <div>
      <div class="law-num">Law 1 &bull; FX Mechanics</div>
      <div class="law-title">Path Currency Transitions Govern FX Count</div>
      <div class="law-stat">100.000%</div>
      <div class="law-body">FX count equals adjacent <code>HomeCurr(BIC_i) != HomeCurr(BIC_{{i+1}})</code> along visible <code>BIC8_PATH</code> hops, completely ignoring invoice currency and <code>'...'</code> gaps.</div>
    </div>
    <div class="law-foot">Scope: 10M rows &bull; Post-mismatch path truth</div>
  </div>
  <div class="law-card" style="border-top-color:#0D9488;">
    <div>
      <div class="law-num" style="color:#0D9488;">Law 2 &bull; Network Routing</div>
      <div class="law-title">Memoryless 1/e Hops &amp; Edge Continuity</div>
      <div class="law-stat">63.23% Direct</div>
      <div class="law-body"><code>Direct &hArr; N=2</code> exactly; <code>N-2 = floor(Exp(1))</code> (ratio 0.368); <code>QFLAG='N' &hArr; '...'</code> (0 errs) follows <code>NegBin(N-1, 0.95)</code> with zero causal latency lag.</div>
    </div>
    <div class="law-foot">Simpson's +51s lag collapses &rarr; &lt;9s per N</div>
  </div>
  <div class="law-card" style="border-top-color:#7C3AED;">
    <div>
      <div class="law-num" style="color:#7C3AED;">Law 3 &bull; Settlement Latency</div>
      <div class="law-title">Directed-Pair Slabs + Gamma(2,5) Queue</div>
      <div class="law-stat">R&sup2; = 0.99996</div>
      <div class="law-body">Direct duration = deterministic corridor slab + shifted <code>Gamma(2,5)</code> noise (std=7.04s). <code>N=3</code> scales as A,B,C sum (R&sup2;=0.992); <code>Covered</code> speeds up ~35%.</div>
    </div>
    <div class="law-foot">Trimodal rails (~25s / ~400s / ~2,000s)</div>
  </div>
  <div class="law-card" style="border-top-color:#D97706;">
    <div>
      <div class="law-num" style="color:#D97706;">Law 4 &bull; Value &amp; Time</div>
      <div class="law-title">Pre-FX Party+Origin Pricing &amp; Twin Clock</div>
      <div class="law-stat">+2.00 ln / B</div>
      <div class="law-body">Clean <code>ln(USD)</code> jumps +2.00 (~7.4x) per Business party + Ord Country (R&sup2;=0.353); Receiver &amp; Currency add &lt;0.0002 R&sup2;. Local business clocks align at r=0.994–1.000.</div>
    </div>
    <div class="law-foot">38.6% USD ints; 1,000 EUR tracks daily spot</div>
  </div>
  <div class="law-card" style="border-top-color:#DC2626;">
    <div>
      <div class="law-num" style="color:#DC2626;">Law 5 &bull; Risk &amp; Fraud</div>
      <div class="law-title">Shared Mule Ecosystem &amp; Recall Null</div>
      <div class="law-stat">1,068 Mules</div>
      <div class="law-body">CEO/ARS/Romance share 1,068 EM mules (hitting 7,828 normal rows), cut Covered 6–37x, and omit end-mismatch (0%). Undocumented <code>CUSTOMER_RECALL</code> yields AUC=0.476.</div>
    </div>
    <div class="law-foot">BEN_DETAILS Faker country matches BIC 0.98%!</div>
  </div>
</div>
"""
slides.append(slide_wrap(1, "UMDAC Datathon 2026 &bull; Swift Case Study Preliminary Round", "Reverse-Engineering the Synthetic SWIFT Global Payment Engine", "Full 10M-Row Executive Synthesis", "", s1_body, dark=True))


# SLIDE 2: Methodology, Research Hypotheses & Full System Causal DAG
dag_svg = """
<svg viewBox="0 0 700 410" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" style="font-family:'Segoe UI',Arial,sans-serif;">
  <defs>
    <marker id="arr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="#334155"/></marker>
    <marker id="arrR" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="#DC2626"/></marker>
    <marker id="arrT" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 z" fill="#0D9488"/></marker>
  </defs>
  <!-- Stage labels -->
  <text x="12" y="18" font-size="10" font-weight="800" fill="#64748B">1. ENTITIES &amp; TIME REGIME</text>
  <text x="242" y="18" font-size="10" font-weight="800" fill="#64748B">2. VALUE &amp; ROUTING GRAPH</text>
  <text x="485" y="18" font-size="10" font-weight="800" fill="#64748B">3. NOISE &amp; OBSERVED LOGS</text>

  <!-- Node: Entity Pool -->
  <rect x="10" y="32" width="185" height="72" rx="6" fill="#EFF6FF" stroke="#2563EB" stroke-width="1.5"/>
  <text x="102" y="50" text-anchor="middle" font-size="10.5" font-weight="800" fill="#0F2942">Account &amp; Bank Registry</text>
  <text x="102" y="65" text-anchor="middle" font-size="9" fill="#1E293B">391k Ord (100% 1:1 BIC)</text>
  <text x="102" y="78" text-anchor="middle" font-size="9" fill="#1E293B">2.59M Ben (2.38% multi-BIC)</text>
  <text x="102" y="92" text-anchor="middle" font-size="8.5" fill="#475569">78.8% Ord=B; ~28.8% Ben=B indep.</text>

  <!-- Node: Temporal -->
  <rect x="10" y="124" width="185" height="68" rx="6" fill="#F0FDFA" stroke="#0D9488" stroke-width="1.5"/>
  <text x="102" y="142" text-anchor="middle" font-size="10.5" font-weight="800" fill="#0F2942">Local Clock &amp; Week Wave</text>
  <text x="102" y="157" text-anchor="middle" font-size="9" fill="#1E293B">UTC Shift &rarr; 08:00 &amp; 17–19h peaks</text>
  <text x="102" y="170" text-anchor="middle" font-size="9" fill="#1E293B">Mon–Fri plateau (91k–218k/day)</text>
  <text x="102" y="183" text-anchor="middle" font-size="8.5" fill="#0D9488" font-weight="700">&rarr; EPOCH_SECONDS + Daily FX</text>

  <!-- Node: Party Details -->
  <rect x="10" y="212" width="185" height="66" rx="6" fill="#FFFFFF" stroke="#64748B" stroke-width="1.5"/>
  <text x="102" y="230" text-anchor="middle" font-size="10.5" font-weight="800" fill="#0F2942">Details &amp; Token Grammar</text>
  <text x="102" y="245" text-anchor="middle" font-size="9" fill="#1E293B">IBAN: Ord &amp; Ben match BIC CC 100%</text>
  <text x="102" y="258" text-anchor="middle" font-size="9" fill="#DC2626" font-weight="700">Ben Faker Locale: 0.98% Match!</text>
  <text x="102" y="271" text-anchor="middle" font-size="8.5" fill="#475569">1/2/3/ tags (15%) &bull; 35-char wrap (44%)</text>

  <!-- Node: Anomalies -->
  <rect x="10" y="302" width="185" height="92" rx="6" fill="#FEF2F2" stroke="#DC2626" stroke-width="1.8"/>
  <text x="102" y="320" text-anchor="middle" font-size="10.5" font-weight="800" fill="#991B1B">Anomaly Injectors (7,006)</text>
  <text x="102" y="335" text-anchor="middle" font-size="8.8" fill="#1E293B">Fat-Finger: 10^k pre-FX shift</text>
  <text x="102" y="348" text-anchor="middle" font-size="8.8" fill="#1E293B">Romance: 1.50x/step + C-victims</text>
  <text x="102" y="361" text-anchor="middle" font-size="8.8" fill="#1E293B">CEO/ARS: 1,068 EM Mules, 0% Cov~</text>
  <text x="102" y="374" text-anchor="middle" font-size="8.8" fill="#1E293B">Duplicate: 2–5 consec. copies</text>
  <text x="102" y="387" text-anchor="middle" font-size="8.5" fill="#6D28D9" font-weight="700">Customer Recall: Random Null (AUC 0.48)</text>

  <!-- Center Top: Amount -->
  <rect x="235" y="32" width="210" height="82" rx="6" fill="#FFFBEB" stroke="#D97706" stroke-width="1.5"/>
  <text x="340" y="50" text-anchor="middle" font-size="10.5" font-weight="800" fill="#0F2942">Pre-FX Principal Pricing</text>
  <text x="340" y="65" text-anchor="middle" font-size="9" fill="#1E293B">ln(USD) = 6.06 + 2*B_o + 2*B_b + Ctry_o</text>
  <text x="340" y="78" text-anchor="middle" font-size="9" fill="#1E293B">+ Log-Normal(0, 2.29); R&sup2; = 0.353</text>
  <text x="340" y="92" text-anchor="middle" font-size="8.5" fill="#92400E">Curr Chosen (88.5% Ord Home) &rarr; Spot FX</text>
  <text x="340" y="106" text-anchor="middle" font-size="8.5" fill="#475569">USD Ints 38.6% &bull; 1,000 EUR daily mode</text>

  <!-- Center Mid: Routing Graph -->
  <rect x="235" y="136" width="210" height="88" rx="6" fill="#EFF6FF" stroke="#1D4ED8" stroke-width="1.6"/>
  <text x="340" y="154" text-anchor="middle" font-size="10.5" font-weight="800" fill="#0F2942">BIC8_PATH Topology</text>
  <text x="340" y="169" text-anchor="middle" font-size="9" fill="#1E293B">N - 2 = floor(Exp(1)) (63.23% Direct)</text>
  <text x="340" y="182" text-anchor="middle" font-size="9" fill="#1E2942">Covered: Ord BIC8 Policy (R&sup2;=0.326)</text>
  <text x="340" y="196" text-anchor="middle" font-size="9" fill="#1E293B">N=3 Middle Hubs: US 18%, CH 11%</text>
  <text x="340" y="211" text-anchor="middle" font-size="8.5" fill="#1D4ED8" font-weight="700">Clean Path Fixed BEFORE Endpoint Noise!</text>

  <!-- Center Low: FX Rule -->
  <rect x="235" y="246" width="210" height="68" rx="6" fill="#F0FDFA" stroke="#0D9488" stroke-width="1.5"/>
  <text x="340" y="264" text-anchor="middle" font-size="10.5" font-weight="800" fill="#0F2942">Deterministic FX Counter</text>
  <text x="340" y="279" text-anchor="middle" font-size="9" fill="#1E293B">&Sigma; I(HomeCurr(BIC_i) != HomeCurr(BIC_i+1))</text>
  <text x="340" y="293" text-anchor="middle" font-size="9" fill="#0D9488" font-weight="700">10,000,000 / 10M with ER=ERN</text>
  <text x="340" y="306" text-anchor="middle" font-size="8.2" fill="#475569">Ignores Invoice CURR &amp; '...' gaps</text>

  <!-- Right Top: USD_AMOUNT output -->
  <rect x="485" y="32" width="205" height="68" rx="6" fill="#FFFFFF" stroke="#0F2942" stroke-width="1.6"/>
  <text x="587" y="51" text-anchor="middle" font-size="10.5" font-weight="800" fill="#0F2942">USD_AMOUNT &amp; CURR</text>
  <text x="587" y="66" text-anchor="middle" font-size="9" fill="#1E293B">Median $3,663 | Mean $97.2k raw</text>
  <text x="587" y="80" text-anchor="middle" font-size="9" fill="#B91C1C" font-weight="700">Clean Mean $79.6k (-18.1%)</text>
  <text x="587" y="93" text-anchor="middle" font-size="8.2" fill="#475569">Recvr &amp; Curr have ZERO causal effect</text>

  <!-- Right Mid: Path Corruption & Continuity -->
  <rect x="485" y="130" width="205" height="96" rx="6" fill="#FFFFFF" stroke="#0F2942" stroke-width="1.6"/>
  <text x="587" y="148" text-anchor="middle" font-size="10.5" font-weight="800" fill="#0F2942">Path Telemetry &amp; Aliases</text>
  <text x="587" y="164" text-anchor="middle" font-size="9" fill="#1E293B">Per-edge '...' ~ Geom(0.05) &rarr; NegBin</text>
  <text x="587" y="178" text-anchor="middle" font-size="9" fill="#1E293B">QFLAG = 'N' &hArr; '...' (0/10M errors)</text>
  <text x="587" y="192" text-anchor="middle" font-size="9" fill="#D97706" font-weight="700">Indep. 5.0% Start / End Mismatch</text>
  <text x="587" y="206" text-anchor="middle" font-size="8.3" fill="#DC2626">End mismatch=0% in ALL 3 scam types!</text>
  <text x="587" y="219" text-anchor="middle" font-size="8.2" fill="#475569">NO causal link to TOTAL_DURATION</text>

  <!-- Right Bot: TOTAL_DURATION -->
  <rect x="485" y="256" width="205" height="104" rx="6" fill="#FFFFFF" stroke="#0F2942" stroke-width="1.6"/>
  <text x="587" y="274" text-anchor="middle" font-size="10.5" font-weight="800" fill="#0F2942">TOTAL_DURATION (s)</text>
  <text x="587" y="290" text-anchor="middle" font-size="9" fill="#1E293B">Direct: Base(u&rarr;v) + Gamma(2,5)</text>
  <text x="587" y="304" text-anchor="middle" font-size="9" fill="#0D9488" font-weight="800">R&sup2; = 0.99996 (RMSE = 7.02s)</text>
  <text x="587" y="318" text-anchor="middle" font-size="9" fill="#1E293B">N=3 Sum(A,B)+(B,C) r = 0.982</text>
  <text x="587" y="332" text-anchor="middle" font-size="9" fill="#2563EB" font-weight="700">Covered Parallelizes (~0.65x lag)</text>
  <text x="587" y="346" text-anchor="middle" font-size="8.2" fill="#475569">Flat vs UTC hour (no intraday congestion)</text>

  <!-- Arrows (routed strictly through open whitespace gutters) -->
  <line x1="195" y1="62" x2="233" y2="62" stroke="#334155" stroke-width="1.6" marker-end="url(#arr)"/>
  <path d="M195,88 L210,88 L210,152 L233,152" fill="none" stroke="#334155" stroke-width="1.6" marker-end="url(#arr)"/>
  <!-- Temporal -> Top Gutter -> USD_AMOUNT (avoids crossing Pre-FX text) -->
  <path d="M195,135 L204,135 L204,23 L465,23 L465,50 L483,50" fill="none" stroke="#0D9488" stroke-width="1.4" stroke-dasharray="4,2" marker-end="url(#arrT)"/>
  <line x1="445" y1="68" x2="483" y2="68" stroke="#334155" stroke-width="1.6" marker-end="url(#arr)"/>
  <line x1="445" y1="172" x2="483" y2="172" stroke="#334155" stroke-width="1.6" marker-end="url(#arr)"/>
  <line x1="340" y1="224" x2="340" y2="244" stroke="#0D9488" stroke-width="1.6" marker-end="url(#arrT)"/>
  <path d="M445,196 L456,196 L456,282 L483,282" fill="none" stroke="#334155" stroke-width="1.5" marker-end="url(#arr)"/>
  <!-- Anomaly -> Pre-FX via left-center vertical channel x=220 -->
  <path d="M195,318 L220,318 L220,98 L233,98" fill="none" stroke="#DC2626" stroke-width="1.4" stroke-dasharray="3,2" marker-end="url(#arrR)"/>
  <!-- Anomaly -> Path Telemetry via open corridor y=329 below FX Counter then up x=469 -->
  <path d="M195,342 L469,342 L469,212 L483,212" fill="none" stroke="#DC2626" stroke-width="1.4" stroke-dasharray="3,2" marker-end="url(#arrR)"/>
  <!-- Anomaly -> Duration cleanly along y=354 -->
  <path d="M195,368 L450,368 L450,352 L483,352" fill="none" stroke="#DC2626" stroke-width="1.4" stroke-dasharray="3,2" marker-end="url(#arrR)"/>
</svg>
"""

s2_body = f"""
<div class="row" style="flex:1;">
  <div class="col" style="flex:0.95;">
    <div class="scard navy">
      <h3><span class="tag t-obs">METHOD PROGRESSION</span> Simple-First Strategy</h3>
      <ul>
        <li><b>1. Full-Census Streaming ETL:</b> Partitioned 4.12 GB CSV (10M rows) into 40 columnar Parquet shards&mdash;eliminating downsampling blind spots on 0.003%–0.07% rare anomalies.</li>
        <li><b>2. Exact Algebraic Rules First:</b> Tested identity checks (<code>N=2 &hArr; Direct</code>, <code>'...' &hArr; QFLAG='N'</code>, adjacent-hop FX formula) before fitting statistical models.</li>
        <li><b>3. Stratified Confound Controls &amp; Fixed Effects:</b> Sequential OLS on <code>ln(USD)</code> (isolating currency &amp; receiver spurious correlations), within-<code>N</code> stratification (exposing QFLAG Simpson's paradox), and Directed-Pair Fixed Effects on latency.</li>
        <li><b>4. Distributional Fit &amp; ML Falsification Only When Needed:</b> Closed-form <code>Exp(1)</code>, <code>NegBin</code>, <code>Gamma(2,5)</code> fits; reserved 5-fold <b>LightGBM</b> specifically to test non-linear signals in undocumented <code>CUSTOMER_RECALL</code> (AUC=0.476).</li>
      </ul>
    </div>
    <div class="scard teal" style="flex:1;">
      <h3><span class="tag t-test">HYPOTHESIS EVOLUTION</span> Initial &rarr; Falsified &rarr; Replaced (Rubric &sect;1.1)</h3>
      <ul>
        <li><b>H1 (Calendar):</b> <i>Init:</i> APAC/US trade on UTC weekends &rarr; <b style="color:#0D9488;">REVISED:</b> Integer UTC shifts unify all 15 hubs (r = 0.994–1.000), but expose a synthetic Mon–Fri rule in AE/SA/IL/EG.</li>
        <li><b>H2 (Pricing &amp; FX):</b> <i>Init:</i> Currency &amp; receiver wealth drive USD value and FX hops &rarr; <b style="color:#DC2626;">REPLACED:</b> ΔR&sup2; &lt; 0.0002; value is 100% sender-driven, while FX counts adjacent <code>BIC8_PATH</code> borders (100%).</li>
        <li><b>H3 (Path &amp; Authority):</b> <i>Init:</i> <code>QFLAG='N'</code> slows wires (+51s) &amp; 5% endpoint mismatches are typos &rarr; <b style="color:#DC2626;">REPLACED:</b> Lag vanishes inside fixed N (&lt;9s); <code>BIC8_PATH</code> rules FX &amp; Duration (R&sup2;=0.99997, MAE=5.6s) while <code>BIC8</code> cols rule Accounts/IBANs!</li>
        <li><b>H4 (Latency &amp; Recall):</b> <i>Init:</i> Intraday congestion slows corridors &amp; recalls are predictable &rarr; <b style="color:#DC2626;">REPLACED:</b> Directed-Pair Slabs + <code>Gamma(2,5)</code> explain 99.997%; <code>CUSTOMER_RECALL</code> is null (AUC=0.476).</li>
      </ul>
    </div>
  </div>
  <div class="col" style="flex:1.35;">
    <div class="simg-box" style="flex:1; padding:8px; justify-content:space-between;">
      <div style="font-size:10px; font-weight:800; color:#0F2942; align-self:flex-start;">RECONSTRUCTED CAUSAL &amp; GENERATIVE DAG (TWO-LAYER AUTHORITY SPLIT)</div>
      <div style="width:100%; flex:1; display:flex; align-items:center; justify-content:center;">{dag_svg}</div>
      <div class="scap">Two-Layer Split: Customer layer (ORDERING/BENEFICIARY_BIC8) governs Acct/IBAN/Details/USD; Interbank layer (BIC8_PATH) governs FX_COUNT (100%) &amp; TOTAL_DURATION (R&sup2;=0.99997).</div>
    </div>
  </div>
</div>
"""
slides.append(slide_wrap(2, "Methodology, Hypotheses & System Architecture", "Simple-First Verification Reconstructs the Exact Causal DAG Without Model Bloat", "Method & Causal Structure", "syn", s2_body))


# SLIDE 3 (Feature Group 1): Data Audit, Sensitivity Matrix & Mandatory Disclosures
s3_rows = []
for r in M["data_prep_sensitivity"]:
    s3_rows.append(
        f"<tr><td><b>{r['subset']}</b></td>"
        f"<td class='num'>{r['n']:,}</td>"
        f"<td class='num'>${r['usd_mean']:,.0f}</td>"
        f"<td class='num'>${r['usd_median']:,.0f}</td>"
        f"<td class='num'>{r['direct_pct']:.2f}%</td>"
        f"<td class='num'>{r['dur_mean']:.1f}s</td></tr>"
    )

s3_body = f"""
<div class="row" style="flex:1;">
  <div class="col" style="flex:1.18;">
    <div class="simg-box" style="flex:1.15;">
      <img src="{imgs['c01_sensitivity_mean_bias.png']}" />
      <div class="scap"><b>Cleaning Trade-Off:</b> 1,940 fat-fingers inflate mean USD +22.1%; dropping QFLAG='N' introduces +1.84 pp Direct collider bias.</div>
    </div>
    <div class="scard navy" style="padding:7px 9px;">
      <h3><span class="tag t-obs">SENSITIVITY LOG</span> Five Analytical Filtering Regimes</h3>
      <table class="stbl">
        <thead><tr><th>Dataset Subset</th><th class="num">Rows</th><th class="num">Mean USD</th><th class="num">Med USD</th><th class="num">Direct %</th><th class="num">Mean Dur</th></tr></thead>
        <tbody>{''.join(s3_rows)}</tbody>
      </table>
    </div>
  </div>
  <div class="col" style="flex:0.92;">
    <div class="scard amber">
      <h3><span class="tag t-test">PREP ISSUES LOG</span> Quality &amp; Schema Audit</h3>
      <ul>
        <li><b>Zero Core Nulls:</b> All 19 analytical columns are 100% populated across 10M rows; no synthetic imputation was needed or applied.</li>
        <li><b>Undocumented Schema Discoveries:</b> <code>DISCLAIMER</code> (row_id=0 verbatim NumPy/Faker synthetic notice; empty on 9,999,999 rows) &amp; <code>CUSTOMER_RECALL</code> (n=317, omitted from brief Appendix A.1).</li>
        <li><b>Duplicate Taxonomy:</b> Exactly <b>1 full-row duplicate pair</b> (rows 3,487,188 &amp; 3,487,190, both <code>DUPLICATE_PAYMENT</code>) plus <b>745 no-timestamp duplicates</b> (679 labeled + 44 normal recurring + 22 scam). Retained clean rows; isolated anomalies for pricing/duration fits.</li>
        <li><b>Collider Protection Decision:</b> Never globally drop <code>QFLAG='N'</code> (7.67%) or path-endpoint mismatches (9.73%)&mdash;retaining them avoids route-length selection bias.</li>
      </ul>
    </div>
    <div class="scard teal">
      <h3><span class="tag t-act">TOOLS, AI DISCLOSURE &amp; COMPLIANCE</span> Rubric &sect;2.4 / &sect;3.4</h3>
      <ul>
        <li><b>Software Stack:</b> Python 3.13, PyArrow (40-part Parquet ETL), Pandas, NumPy, SciPy (KS/Chi2 tests), Scikit-Learn (OLS/FE), LightGBM (5-fold OOF check), Matplotlib, PyMuPDF, headless Chromium.</li>
        <li><b>AI Assistance Disclosure:</b> AI coding assistance helped orchestrate the reproducible ETL pipeline, diagnostic verification scripts, chart layout tuning, and HTML/PDF publishing. All quantitative claims &amp; statistics were computed deterministically on the raw CSV via verified Python scripts.</li>
        <li><b>External Data &amp; Ethics Statement:</b> Strictly <b>zero external datasets</b> (no real FX feeds, GDP series, or sanction lists). Corpus is 100% fictitious (per row-0 disclaimer); contains no real PII/customer risk, but synthetic quirks limit direct production deployment.</li>
      </ul>
    </div>
    <div class="scard navy" style="flex:1; padding:6px 9px;">
      <h3><span class="tag t-obs">UNDOCUMENTED &amp; LABELED ANOMALY COUNTS</span> Total = 7,006 Rows (0.070%)</h3>
      <table class="stbl" style="font-size:8.3px;">
        <thead><tr><th>Found Label / Schema Item</th><th class="num">Row Count</th><th>Status &amp; Analytical Handling</th></tr></thead>
        <tbody>
          <tr><td><b>ROMANCE_SCAM / CEO / ARS</b></td><td class="num">2,476 / 945 / 649</td><td>Documented fraud (excluded from clean baselines)</td></tr>
          <tr><td><b>FAT_FINGER / DUPLICATE</b></td><td class="num">1,940 / 679</td><td>Documented op defects (+44 normal / +22 scam no-TS dups)</td></tr>
          <tr><td><b>CUSTOMER_RECALL*</b></td><td class="num"><b>317</b></td><td><b>Undocumented</b> in brief Appendix A.1 (AUC=0.476)</td></tr>
          <tr><td><b>DISCLAIMER (row_id=0)*</b></td><td class="num"><b>1 populated / 10M</b></td><td><b>Undocumented</b>; proves NumPy/Faker synthetic generation</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>
"""
slides.append(slide_wrap(3, "Feature Group 1 of 10 &bull; Data Integrity, Preparation & Governance", "1,940 Tail Errors Inflate Means +22.1%; Naive Continuity Filtering Biases Topology +1.84 pp", "Ref: D1 (c01–c02) &bull; Conf: HIGH", "fg", s3_body))


# SLIDE 4 (Feature Group 2): Entity Graph & Party-Type Coupling (Layout C: Top Cards + Bottom Aligned Charts)
s4_body = f"""
<div class="scard rule">
  <span class="eq">Observed Entity Law: Ord Accts (391,132) &mdash;100% 1:1&rarr; Ordering BIC8 &amp; Type (78.75% B) | Ben Accts (2,590,877) &mdash;97.62%&rarr; 1 BIC (2.38% span 2–8 BICs); P(Ben=B|Ord) = 28.8% independent</span>
  <div style="font-size:9.3px; color:#334155;"><b>Scope &amp; Limits:</b> Exact across all 10,000,000 rows &bull; 8,392 ordering / 8,266 beneficiary / 8,686 total BIC8s across 229 countries. Multi-BIC accounts (61,606) have <b>0% IBAN</b> strings.</div>
</div>
<div class="row" style="gap:9px;">
  <div class="scard blue" style="flex:1;">
    <h3><span class="tag t-obs">OBSERVED STRUCTURE</span> Cardinality &amp; Flow</h3>
    <ul>
      <li><b>Origination Concentration:</b> Business accounts send 78.75% of rows (<b>B&rarr;C 56.04%</b>, B&rarr;B 22.71%), yet receivers skew Retail (71.18% C; C&rarr;C 15.15%, C&rarr;B 6.11%).</li>
      <li><b>6.62x Pool Asymmetry:</b> 2.59M Beneficiary IDs vs 391k Ordering IDs across ~8.3k active BIC8s each.</li>
    </ul>
  </div>
  <div class="scard amber" style="flex:1;">
    <h3><span class="tag t-test">COUPLING &amp; ARTIFACT TESTS</span> Falsifications</h3>
    <ul>
      <li><b>Role Independence:</b> <code>P(Ben=B | Ord=B) = 28.84%</code> vs <code>P(Ben=B | Ord=C) = 28.73%</code>&mdash;sender role does NOT induce B2B vs B2C recipient affinity beyond random pairing.</li>
      <li><b>Dual-Role Paradox:</b> 253,120 account IDs appear as both sender &amp; receiver: <b>99.92%</b> keep same BIC8 &amp; country, yet only <b>49.60%</b> keep same B/C type!</li>
    </ul>
  </div>
  <div class="scard teal" style="flex:1;">
    <h3><span class="tag t-int">ECONOMIC VS SYNTHETIC</span> Interpretation</h3>
    <ul>
      <li><b>Economic Logic:</b> High B2C (5.60M) &amp; B2B (2.27M) shares reflect payroll, vendor settlements, and consumer payouts originating from fewer corporate treasury accounts.</li>
      <li><b>Generator Deviation:</b> 49.60% B/C coin-flip on overlap accounts &amp; 2–8 BIC non-IBAN IDs arise from independent per-role B/C sampling and Faker account collisions.</li>
    </ul>
  </div>
</div>
<div class="row" style="flex:1; min-height:0; gap:10px;">
  <div class="simg-box" style="flex:0.94;">
    <img src="{imgs['c03_party_matrix.png']}" />
    <div class="scap">B2C dominates (56.0%); P(Ben=B) invariant at ~28.8% regardless of sender.</div>
  </div>
  <div class="simg-box" style="flex:1.50;">
    <img src="{imgs['c04_entity_cardinality_asym.png']}" />
    <div class="scap">Strict 1:1 Ordering account lock vs 61,606 multi-BIC Beneficiaries &amp; 49.6% dual-role type flips.</div>
  </div>
</div>
"""
slides.append(slide_wrap(4, "Feature Group 2 of 10 &bull; Entity Graph & Party-Type Coupling", "Ordering Accounts Lock 1:1 to Banks, While Beneficiary Type & Dual-Role Flags Decouple", "Ref: D2 (c03–c04) &bull; Conf: HIGH", "fg", s4_body))


# SLIDE 5 (Feature Group 3): Temporal Dynamics & Calendar (Layout B: Left Narrative + Right Dual Stacked Charts)
s5_body = f"""
<div class="row" style="flex:1;">
  <div class="col" style="flex:0.86;">
    <div class="scard rule">
      <span class="eq">Clock Law: Traffic(c, h_UTC) &equiv; Profile_Universal((h_UTC + Offset_c) mod 24)</span>
      <div style="font-size:9px; color:#334155;"><b>Scope:</b> Apr 30 – Jul 31, 2023 (13 weeks); daily volume plateaus jump strictly on Mondays (<b>91k – 218k tx/day</b>).</div>
    </div>
    <div class="scard navy">
      <h3><span class="tag t-obs">OBSERVED DYNAMICS</span> Hour &amp; Day Structure</h3>
      <ul>
        <li><b>Universal Twin-Peak Day:</b> Local hours peak sharply at <b>08:00 (~7.5%)</b> market open and <b>17:00–19:00 (~7.5%)</b> RTGS/Nostro cutoff; overnight trough at 02:00–05:00 (~1.1%).</li>
        <li><b>Weekly Calendar:</b> Aligned weekdays share Mon 18.8%, Tue–Wed 17.0%, Thu 18.0%, Fri 19.0%, Sat 3.8%, Sun 6.4%.</li>
        <li><b>Zero Macro Coupling:</b> Weekly 2.4x step-waves leave clean median USD ($3,663) &amp; B2B share (22.7%) untouched.</li>
      </ul>
    </div>
    <div class="scard amber">
      <h3><span class="tag t-test">CONFOUNDERS &amp; DEVIATIONS</span></h3>
      <ul>
        <li><b>UTC Weekend Paradox Debunked:</b> High raw UTC Sunday share in AU (<b>10.14%</b>), JP (9.14%), SG (8.44%) &amp; US Saturday (6.58%) drops to uniform local weekends (AU Sun <b>6.28%</b>, US Sat <b>3.72%</b>).</li>
        <li><b>Middle East Deviation:</b> AE, SA, IL &amp; EG all peak Fri (<b>18.8–19.3%</b>) and dip Sat/Sun&mdash;using Western Mon–Fri instead of real Sun–Thu banking weeks!</li>
      </ul>
    </div>
    <div class="scard teal" style="flex:1;">
      <h3><span class="tag t-int">ECONOMIC LOGIC &amp; OPEN QUESTION</span></h3>
      <ul>
        <li><b>Real Economics:</b> Twin peaks reflect corporate batch release at local open and end-of-day cutoff deadlines.</li>
        <li><b>Open Artifacts:</b> Why integer summer offsets (e.g. UTC+9 CN/SG, UTC+12 AU) are applied globally and what drives Monday volume regime shifts remain external generator parameters.</li>
      </ul>
    </div>
  </div>
  <div class="col" style="flex:1.18;">
    <div class="simg-box" style="flex:1;">
      <img src="{imgs['c05_timezone_alignment_hours.png']}" />
      <div class="scap"><b>Timezone Alignment Proof:</b> Shifting UTC curves by national offsets aligns 15 hubs at r = 0.994–1.000 (IN r = 0.957).</div>
    </div>
    <div class="simg-box" style="flex:1;">
      <img src="{imgs['c06_dow_utc_vs_local.png']}" />
      <div class="scap"><b>DOW Collapse &amp; Synthetic Deviation:</b> APAC Sunday &amp; US Saturday spikes vanish locally; AE &amp; SA track Mon–Fri.</div>
    </div>
  </div>
</div>
"""
slides.append(slide_wrap(5, "Feature Group 3 of 10 &bull; Temporal Dynamics & Business Calendar", "Integer Timezones Unify 15 Markets (r = 0.994–1.000), Exposing UTC Weekend Illusions", "Ref: D3 (c05–c07) &bull; Conf: HIGH", "fg", s5_body))


# SLIDE 6 (Feature Group 4): Payment Value Mechanics (Layout B with Left Table + Right Stacked c09 & c10)
s6_body = f"""
<div class="row" style="flex:1;">
  <div class="col" style="flex:0.90;">
    <div class="scard rule">
      <span class="eq">ln(USD) = 6.06 + 2.00*I(Ord=B) + 2.00*I(Ben=B) + Country_Ord + N(0, 2.29&sup2;)</span>
      <div style="font-size:9px; color:#334155;"><b>Scope:</b> 9,992,994 Clean Rows (Excl. 7,006 anomalies). Total Ordering-side R&sup2; reaches <b>0.3526</b> (country) / <b>0.3855</b> (BIC8).</div>
    </div>
    <div class="scard navy" style="padding:6px 8px;">
      <h3><span class="tag t-obs">ADDITIVE PARTY MULTIPLIERS</span> Clean Segment Fit (<span style="text-transform:none;">~e&sup2; = 7.39x</span> / Business)</h3>
      <table class="stbl">
        <thead><tr><th>Segment</th><th class="num">Clean Count</th><th class="num">ln Mean</th><th class="num">ln Std</th><th class="num">Geom Mean</th><th class="num">Median USD</th></tr></thead>
        <tbody>
          <tr><td><b>C2C</b> (0 B)</td><td class="num">1,512,382</td><td class="num">6.062</td><td class="num">2.292</td><td class="num">$429</td><td class="num">$464</td></tr>
          <tr><td><b>B2C</b> (1 B)</td><td class="num">5,600,965</td><td class="num">7.994</td><td class="num">2.284</td><td class="num">$2,964</td><td class="num">$3,118</td></tr>
          <tr><td><b>C2B</b> (1 B)</td><td class="num">609,762</td><td class="num">8.027</td><td class="num">2.285</td><td class="num">$3,063</td><td class="num">$3,236</td></tr>
          <tr><td><b>B2B</b> (2 B)</td><td class="num">2,269,885</td><td class="num">9.998</td><td class="num">2.285</td><td class="num">$21,981</td><td class="num">$22,883</td></tr>
        </tbody>
      </table>
    </div>
    <div class="scard amber">
      <h3><span class="tag t-test">CONFOUNDERS DEBUNKED</span> Currency &amp; Receiver Null</h3>
      <ul>
        <li><b>Zero Bilateral Pull:</b> Controlling for Party + Ordering Country (R&sup2;=0.35258), adding Currency yields <b>+0.00001</b> R&sup2; &amp; Beneficiary Country yields <b>+0.00015</b> R&sup2;!</li>
        <li><b>1,750x Origin Spread:</b> Party-adjusted multipliers span <b>MC 7.66x</b> ($34.7k med), LI 5.40x, LU 3.30x down to <b>BI 0.0044x</b> ($14), MZ 0.0079x, YE 0.0097x.</li>
      </ul>
    </div>
    <div class="scard teal" style="flex:1;">
      <h3><span class="tag t-int">PRE-FX PRINCIPAL LAW &amp; VARIANCE DECOMPOSITION</span></h3>
      <ul>
        <li><b>Local-Currency Rounding:</b> Native USD payments exhibit <b>38.64%</b> exact integer amounts (15.55% %100) vs ~0.94–1.02% for EUR/GBP/CHF, which convert via daily FX schedules (92-day 1,000 EUR mode recovers $1,100.72–$1,217.69).</li>
        <li><b>Variance Decomposition (61.5% Residual):</b> Pooled party &sigma; = 2.29 shrinks to <b>&sigma; = 1.98</b> within (BIC8, Party) and <b>&sigma; = 1.82</b> within (Account, Party)&mdash;separating bank/client tiers from invoice noise.</li>
      </ul>
    </div>
  </div>
  <div class="col" style="flex:1.14;">
    <div class="simg-box" style="flex:1;">
      <img src="{imgs['c09_amount_r2_and_countries.png']}" />
      <div class="scap">Ordering Country adds +13.5 pp R&sup2; (1,750x spread); Currency &amp; Receiver contribute zero incremental explanatory power.</div>
    </div>
    <div class="simg-box" style="flex:1;">
      <img src="{imgs['c10_usd_rounding_and_eur_fx.png']}" />
      <div class="scap">USD retains 38.64% round numbers while non-USD currencies dilute decimals across 92 daily FX spot rates.</div>
    </div>
  </div>
</div>
"""
slides.append(slide_wrap(6, "Feature Group 4 of 10 &bull; Payment Value Architecture", "Pre-FX Log-Normal Pricing Scales Multiplicatively (+2.0 ln) on Sender Role & Origin Only", "Ref: D4 (c08–c10) &bull; Conf: HIGH", "fg", s6_body))


# SLIDE 7 (Feature Group 5): Currency Choice & 100% FX Rule (Layout C: Top Cards + Bottom Aligned c11 & c12)
s7_body = f"""
<div class="scard rule">
  <span class="eq">Exact FX Law: CURRENCY_CONVERSION_COUNT = &Sigma;_{{i=0..k-2}} I(HomeCurr(VisibleBIC_i) != HomeCurr(VisibleBIC_{{i+1}}))</span>
  <div style="font-size:9.3px; color:#334155;"><b>Scope:</b> Verified across ALL <b>10,000,000</b> rows (Direct, Indirect, Covered, QFLAG='N', and Mismatch subsets) &mdash; <b>9,999,999 / 10M</b> empirical match, <b>10,000,000 / 10M</b> with Eritrea ER=ERN!</div>
</div>
<div class="row" style="gap:9px;">
  <div class="scard blue" style="flex:1;">
    <h3><span class="tag t-obs">INVOICE CURRENCY REGIMES</span> 155 Currencies</h3>
    <ul>
      <li><b>Same Currency Zone (1.55M):</b> Uses shared local currency <b>94.45%</b> of the time (e.g. intra-Eurozone EUR).</li>
      <li><b>Cross-Zone (8.45M):</b> <b>88.53%</b> invoice in Ordering Home currency, 5.01% Ben Home, and <b>6.46%</b> Third Vehicle (EUR 189k, USD 108k, GBP 42k).</li>
    </ul>
  </div>
  <div class="scard amber" style="flex:1;">
    <h3><span class="tag t-test">REJECTED FX HYPOTHESES</span> Counterintuitive Proofs</h3>
    <ul>
      <li><b>Invoice Currency Irrelevant:</b> US&rarr;US in EUR records <code>fx=0</code>; US&rarr;CH&rarr;US in USD records <code>fx=2</code>! <code>CURRENCY_CODE</code> plays zero role in <code>FX_COUNT</code>.</li>
      <li><b>Missing-Hop &amp; Endpoint Order:</b> Unseen <code>'...'</code> gaps add ZERO FX hops; on 5% endpoint mismatches, FX follows <code>BIC8_PATH</code> endpoints 100%, proving pre-alias calculation!</li>
    </ul>
  </div>
  <div class="scard teal" style="flex:1;">
    <h3><span class="tag t-int">ENGINE PIPELINE &amp; REALISM</span> Deviations</h3>
    <ul>
      <li><b>Eritrea Stress-Test Confirmation:</b> Only 1 apparent mismatch out of 10M (<code>row_id 4871844</code>, <code>0033PTP0;0000ERA0</code>) vanished when tiny ER (0 ord, 2 ben rows) mapped to <code>ERN</code> instead of modal <code>EUR</code>.</li>
      <li><b>Real-World Departure:</b> Real offshore USD clearing avoids FX conversion across borders; this generator triggers FX mechanically at every bank currency boundary.</li>
    </ul>
  </div>
</div>
<div class="row" style="flex:1; min-height:0; gap:10px;">
  <div class="simg-box" style="flex:1.52;">
    <img src="{imgs['c11_currency_choice_rules.png']}" />
    <div class="scap">Cross-zone payments favor Ordering Home currency (88.53%); EUR &amp; USD lead third-currency clearing (6.46%).</div>
  </div>
  <div class="simg-box" style="flex:0.99;">
    <img src="{imgs['c12_fx_transition_proof.png']}" />
    <div class="scap">Adjacent visible-hop zone jumps explain 9,999,999 / 10M rows (10M / 10M with ER=ERN).</div>
  </div>
</div>
"""
slides.append(slide_wrap(7, "Feature Group 5 of 10 &bull; Currency Selection & FX Conversions", "Adjacent Jurisdictional Borders Along BIC8_PATH Explain 100% of FX Conversions", "Ref: D5 (c11–c12) &bull; Conf: EXACT", "fg", s7_body))


# SLIDE 8 (Feature Group 6): Routing Topology & Covered Mechanics (Layout C: Top Cards + Bottom c13 & c14)
s8_body = f"""
<div class="scard rule">
  <span class="eq">Hop Generator: TRANSACTION_TYPE == 'Direct' &hArr; N=2 (0 errs) | (N - 2) = floor(Exp(1)), P(N=k) = (1 - 1/e)(1/e)^(k-2)</span>
  <div style="font-size:9.3px; color:#334155;"><b>Scope:</b> All 10M routes (N=2..22): Direct <b>63.233%</b> vs <code>1-1/e = 63.212%</code>; N&ge;3 Covered share = <b>11.313%</b> (415,962 / 3.68M multi-hop rows).</div>
</div>
<div class="row" style="gap:9px;">
  <div class="scard blue" style="flex:1;">
    <h3><span class="tag t-obs">GEOMETRIC HOPS &amp; CLEARING HUBS</span></h3>
    <ul>
      <li><b>Memoryless Survival (1/e = 0.3679):</b> Empiric ratios stay flat at 0.3676–0.3685 across N=2..10, invariant across B2B/C2C &amp; domestic vs cross-border!</li>
      <li><b>N=3 Hub Hierarchy:</b> <b>US 17.98%</b>, <b>CH 10.51%</b>, GB 6.31%, DE 6.08% middle banks&mdash;with <b>+1.87 to +3.41 pp</b> lift when transaction uses the hub's home currency!</li>
    </ul>
  </div>
  <div class="scard amber" style="flex:1;">
    <h3><span class="tag t-test">CONFOUNDER &amp; POLICY TESTS</span></h3>
    <ul>
      <li><b>Isle of Man Country Illusion:</b> Raw IM shows 50.4% Covered multi-hop ONLY because 1 large bank (<code>0004IMD0</code>, 58.3% Cov) sits there! Bank FE explains <b>R&sup2;=0.326</b> vs Country <b>0.064</b>.</li>
      <li><b>Hop-Length Invariance Rejected:</b> Covered share does NOT rise on longer paths&mdash;stays flat at 11.29–11.33% across N=3, 4, 5, 6 (set purely at bank level, median 2.7%, p99 89.4%).</li>
    </ul>
  </div>
  <div class="scard teal" style="flex:1;">
    <h3><span class="tag t-int">CORRESPONDENT BANKING VS SYNTHETIC</span></h3>
    <ul>
      <li><b>Economic Plausibility:</b> USD/CHF/GBP clearing dominance reflects real nostro liquidity concentration; bank-level Covered preference captures institutional MT202COV vs serial MT103 policies.</li>
      <li><b>Synthetic Quirk:</b> True domestic wires use RTGS directly rather than 36.8% multi-hop chains&mdash;showing hop count N is sampled before corridor geography.</li>
    </ul>
  </div>
</div>
<div class="row" style="flex:1; min-height:0; gap:10px;">
  <div class="simg-box" style="flex:0.94;">
    <img src="{imgs['c13_participant_geometric_law.png']}" />
    <div class="scap">Log-linear decay matches floor(Exp(1)) with exact 1/e = 0.3679 slope from N=2 to N=10+.</div>
  </div>
  <div class="simg-box" style="flex:1.56;">
    <img src="{imgs['c14_covered_bank_bimodality.png']}" />
    <div class="scap">Ordering BIC8 sets Covered share (0–95%, R&sup2;=0.326); currency settlement boosts US/CH/GB/CN middle hubs.</div>
  </div>
</div>
"""
slides.append(slide_wrap(8, "Feature Group 6 of 10 &bull; Network Routing & Covered Preference", "Memoryless 1/e Hop Decay Meets Bank-Owned Covered Policy & Currency Clearing Hubs", "Ref: D6 (c13–c14) &bull; Conf: HIGH", "fg", s8_body))


# SLIDE 9 (Feature Group 7): Path Continuity & Endpoint Mismatches (Layout B: Left Narrative + Right c15 & c16)
s9_body = f"""
<div class="row" style="flex:1;">
  <div class="col" style="flex:0.86;">
    <div class="scard rule">
      <span class="eq">QFLAG='N' &hArr; '...' (0/10M errs) | #NonDot == N (0 errs) | Dots ~ NegBin(N-1, 0.95)</span>
      <div style="font-size:9px; color:#334155;"><b>Scope:</b> 767,204 QFLAG='N' paths (7.67%) + 498,399 start (4.98%) &amp; 499,265 end (4.99%) BIC mismatches.</div>
    </div>
    <div class="scard navy">
      <h3><span class="tag t-obs">EXACT PATH NOISE GRAMMAR</span></h3>
      <ul>
        <li><b>Per-Edge Missing Telemetry:</b> Between each of the <code>N-1</code> adjacent visible pairs, unseen gaps inject with <code>Geom(p=0.05)</code> &rarr; <b>N=2: 4.99%, N=3: 9.74%, N=4: 14.25%, N=5: 18.50%</b>.</li>
        <li><b>Additive Representation:</b> Missing hops appear between visible BICs (never replacing endpoints), so visible non-dot count ALWAYS equals <code>PARTICIPANT_COUNT</code>.</li>
        <li><b>Independent 5% Endpoint Aliases:</b> Joint both-end mismatch = <b>0.247%</b> vs theoretical independent <code>0.0498 * 0.0499 = 0.249%</code>.</li>
      </ul>
    </div>
    <div class="scard amber">
      <h3><span class="tag t-test">CONFOUNDING &amp; TWO-LAYER AUTHORITY SPLIT</span></h3>
      <ul>
        <li><b>Simpson's Latency Paradox Debunked:</b> Pooled Indirect N=3..6 <code>QFLAG='N'</code> averages <b>991s vs 941s (+50s)</b>. Conditioning on N eliminates the gap (&lt;9s noise)&mdash;longer routes simply cause both more delays and more 5% edge drops!</li>
        <li><b>Two-Layer Authority Law Proven:</b> On the 9.73% mismatch rows, <code>ORDERING/BENEFICIARY_BIC8</code> govern Account/IBAN/Details/USD <b>100%</b>, whereas <code>BIC8_PATH[0..-1]</code> governs <code>FX_COUNT</code> (<b>100%</b>) AND <code>TOTAL_DURATION</code> (<b>R&sup2;=0.999972, MAE=5.60s</b> vs 300.3s)!</li>
      </ul>
    </div>
    <div class="scard teal" style="flex:1;">
      <h3><span class="tag t-int">RESPONDENT BANKING VS SCAM BYPASS</span></h3>
      <ul>
        <li><b>Real Economics:</b> Mismatches (~95% cross-border) model indirect respondent banks whose customers hold local IBANs while settling SWIFT messages via correspondent gateways (<code>BIC8_PATH[0/-1]</code>).</li>
        <li><b>Scam Difference:</b> Normal rows carry 5.0% end mismatches, whereas <b>all 3 social-engineering scams have 0.0% end mismatch</b> (direct mule targeting without respondent aliasing).</li>
      </ul>
    </div>
  </div>
  <div class="col" style="flex:1.18;">
    <div class="simg-box" style="flex:1;">
      <img src="{imgs['c15_dots_negbin_and_endpoints.png']}" />
      <div class="scap"><b>Exact Stochastic Fit:</b> Observed '...' gap counts match NegBin(N-1, 0.95) across N=2..5; 5% endpoint noise is strictly independent.</div>
    </div>
    <div class="simg-box" style="flex:1;">
      <img src="{imgs['c16_qflag_confounder_null.png']}" />
      <div class="scap"><b>Simpson's Confounder Crushed:</b> Global +50s lag for QFLAG='N' shrinks to &lt;9s random noise within every fixed N stratum.</div>
    </div>
  </div>
</div>
"""
slides.append(slide_wrap(9, "Feature Group 7 of 10 &bull; Path Continuity & Endpoint Mismatches", "Missing Hops Obey NegBin(N-1, 0.95) & Independent 5% Aliases With Zero Causal Delay", "Ref: D7 (c15–c16) &bull; Conf: HIGH", "fg", s9_body))


# SLIDE 10 (Feature Group 8): Settlement Duration Architecture (Layout B: Left Narrative + Right c17 & c18)
s10_body = f"""
<div class="row" style="flex:1;">
  <div class="col" style="flex:0.88;">
    <div class="scard rule">
      <span class="eq">Direct: D(u&rarr;v) = Base(Path[0], Path[-1]) + Gamma(k=2, &theta;=5) &nbsp;|&nbsp; Path FE R&sup2; = 0.999972 (RMSE = 6.01s)</span>
      <div style="font-size:9px; color:#334155;"><b>Scope:</b> 9.993M Clean Rows (2–21,649s); (Ord,Ben) Clean Match R&sup2; = 0.999959 (RMSE = 7.02s); Indirect additive FE R&sup2; = 0.9933 vs 0.8634 log-space.</div>
    </div>
    <div class="scard navy">
      <h3><span class="tag t-obs">DETERMINISTIC SLAB &amp; GAMMA(2,5)</span></h3>
      <ul>
        <li><b>Exact Noise Signature:</b> Across top 10 large corridors (e.g. <code>0001MQM0&rarr;0009TNT0</code>, n=12,640), excess above pair min averages <b>9.81–10.09s</b> with variance <b>46.3–52.7</b>&mdash;exact match to <code>Gamma(2,5)</code> (mean 10, var 50, std <b>7.07s</b> vs obs median <b>7.04s</b>)!</li>
        <li><b>N=3 Composition:</b> Indirect 3-bank latency tracks <code>D(A,B)+D(B,C)</code> at <b>r = 0.982</b> (med -17.7s due to shared queue draw &amp; 3-bank FE <b>R&sup2;=0.9924</b> vs A+C only <b>0.7875</b>).</li>
        <li><b>Covered Acceleration:</b> Holding Ordering BIC8 &amp; N=3 fixed, Covered routes run at <b>0.62x–0.71x</b> (~35% faster).</li>
      </ul>
    </div>
    <div class="scard amber">
      <h3><span class="tag t-test">FALSIFICATIONS &amp; DIRECTED EDGE PROOF</span></h3>
      <ul>
        <li><b>Intraday Congestion Refuted:</b> Hourly corridor durations stay completely flat across UTC 00–23 (US&rarr;US 42–44s, US&rarr;SG 33–36s, DE&rarr;FR 40–43s).</li>
        <li><b>Country/Timezone Diff Refuted:</b> Within the same country pair, directional diffs <code>Base(u&rarr;v) - Base(v&rarr;u)</code> have <b>std = 199.4s</b>, proving bank-pair directed SLA weights!</li>
      </ul>
    </div>
    <div class="scard teal" style="flex:1;">
      <h3><span class="tag t-int">SLA TIERS &amp; UNEXPLAINED ASYMMETRY</span></h3>
      <ul>
        <li><b>Banking Intuition:</b> Trimodal peaks (~25s instant rails, ~400s batch clearing, ~2,000s compliance/timezone hold); Covered parallelizes cover &amp; announcement legs.</li>
        <li><b>Open Gap (~600s Jumps):</b> 72.7% of bidirectional pairs match &le;10s (med 1.57s), yet ~27% show discrete <b>~600s</b> directional shifts (e.g. <code>0309CHZ0&harr;0035AU20</code>: 631.6s vs 30.1s).</li>
      </ul>
    </div>
  </div>
  <div class="col" style="flex:1.16;">
    <div class="simg-box" style="flex:1;">
      <img src="{imgs['c17_duration_trimodal_and_pairs.png']}" />
      <div class="scap"><b>Micro Determinism Under Macro Trimodality:</b> Pair fixed effects hit R&sup2;=0.99996 with textbook Gamma(2,5) queue noise.</div>
    </div>
    <div class="simg-box" style="flex:1;">
      <img src="{imgs['c18_duration_n3_and_covered_speedup.png']}" />
      <div class="scap"><b>Edge Sum Composition &amp; Covered Speedup:</b> N=3 matches D(A,B)+D(B,C) at r=0.982; Covered reduces latency ~35% at same bank.</div>
    </div>
  </div>
</div>
"""
slides.append(slide_wrap(10, "Feature Group 8 of 10 &bull; Settlement Duration Architecture", "Directed-Pair Base Tables + Gamma(2,5) Noise Explain 99.997% of Direct Latency", "Ref: D8 (c17–c18) &bull; Conf: HIGH", "fg", s10_body))


# SLIDE 11 (Feature Group 9): Party Details & Tokenization Grammar (Layout B: Left Narrative + Right c19 & c20)
s11_body = f"""
<div class="row" style="flex:1;">
  <div class="col" style="flex:0.86;">
    <div class="scard rule">
      <span class="eq">Ord Address &hArr; Ord BIC8 CC (100.00%) &nbsp;|&nbsp; Ben Address &hArr; Ben BIC8 CC (0.984% Random Locale!)</span>
      <div style="font-size:9px; color:#334155;"><b>Scope:</b> 20M Party Detail &amp; Token records across 10M rows; 19 observed parser token types.</div>
    </div>
    <div class="scard navy">
      <h3><span class="tag t-obs">TEXT FORMAT &amp; JURISDICTION RULES</span></h3>
      <ul>
        <li><b>3-Tier Format Split:</b> ~15–16% use structured MT50F/59F <code>1/2/3/</code> lines; ~10–12% omit address (account/BIC only); ~73% emit unstructured Faker text wrapped at 35 chars (<code>&lt;line_break:35&gt;</code> = 44.0%/44.2%).</li>
        <li><b>Jurisdictional IBAN Boundary:</b> European/SEPA+GB+CH rows carry valid BIC-matched IBANs in <b>90.3–96.1%</b> of rows vs exact <b>0.000%</b> in US/CN/JP/SG/AU/HK/CA/IM!</li>
        <li><b>Closed Keyword Vocabulary:</b> Exactly <b>10 words</b> (<code>gross..funds</code>, ~159k each total) inject into unstructured text at <b>9.42%</b> vs <b>0.002%</b> inside structured <code>1/2/3/</code> tags!</li>
      </ul>
    </div>
    <div class="scard amber">
      <h3><span class="tag t-test">FAKER LOCALE CONTRADICTION PROVEN</span></h3>
      <ul>
        <li><b>IBAN vs Address Split:</b> When a Beneficiary holds an IBAN (49.77%), its prefix agrees with <code>BENEFICIARY_BIC8</code> <b>100.0%</b>, yet the street/city/country text matches only <b>0.984%</b>&mdash;skewing globally to Faker's largest locale tables (<b>CN 14.5%, BR 8.9%, UA 7.2%, ET 5.8%</b>)!</li>
      </ul>
    </div>
    <div class="scard teal" style="flex:1;">
      <h3><span class="tag t-int">REALISM VS COMPLIANCE HAZARD</span></h3>
      <ul>
        <li><b>Authentic FIN Constraints:</b> 35-char field wrapping &amp; rare truncations (387 tags) realistically mirror legacy MT103 message friction that triggers compliance hits.</li>
        <li><b>Pipeline Alert:</b> Any ML/NLP feature extracting geo-risk from <code>BENEFICIARY_DETAILS</code> instead of <code>BENEFICIARY_BIC8</code> or IBAN prefix will misclassify <b>99%</b> of entities!</li>
      </ul>
    </div>
  </div>
  <div class="col" style="flex:1.18;">
    <div class="simg-box" style="flex:1;">
      <img src="{imgs['c19_details_country_mismatch.png']}" />
      <div class="scap"><b>Asymmetric Locale Coupling:</b> Ordering details align 100.00% with BIC country; Beneficiary details drop to 0.98% (populous-locale bias).</div>
    </div>
    <div class="simg-box" style="flex:1;">
      <img src="{imgs['c20_iban_structure_finwords.png']}" />
      <div class="scap"><b>IBAN &amp; Closed Lexicon Laws:</b> Strict SEPA/UK/CH vs non-IBAN regimes; 10 uniform &lt;finance_word&gt; keywords restricted to free text.</div>
    </div>
  </div>
</div>
"""
slides.append(slide_wrap(11, "Feature Group 9 of 10 &bull; Party Details & Token Grammar", "Faker Locale Disconnects Beneficiary Addresses (0.98%) While IBANs & 35-Char Wrapping Hold 100%", "Ref: D9 (c19–c20) &bull; Conf: HIGH", "fg", s11_body))


# SLIDE 12 (Feature Group 10): Anomaly Mechanics Overview (Layout C+: Top Compact Table + Dual Charts c21 & c23)
a_rows = []
for key, lbl in [
    ("NORMAL", "Normal Baseline (99.93%)"),
    ("ROMANCE_SCAM", "ROMANCE_SCAM (2,476)"),
    ("FAT_FINGER_MISTAKE", "FAT_FINGER_MISTAKE (1,940)"),
    ("CEO_SCAM", "CEO_SCAM (945)"),
    ("DUPLICATE_PAYMENT", "DUPLICATE_PAYMENT (679)"),
    ("ACCOUNT_REPLACEMENT_SCAM", "ACCT_REPLACEMENT_SCAM (649)"),
    ("CUSTOMER_RECALL", "CUSTOMER_RECALL* (317, Undoc.)"),
]:
    r = anom_map[key]
    a_rows.append(
        f"<tr><td><b>{lbl}</b></td>"
        f"<td class='num'>{r['ord_B_pct']:.1f}%</td>"
        f"<td class='num'>${r['usd_median']:,.0f}</td>"
        f"<td class='num'>${r['usd_mean']:,.0f}</td>"
        f"<td class='num'>{r['dur_median']:.0f}s</td>"
        f"<td class='num'>{r['covered_among_multihop_pct']:.2f}%</td>"
        f"<td class='num'>{r['start_mis_pct']:.1f}% / {r['end_mis_pct']:.1f}%</td></tr>"
    )

s12_body = f"""
<div class="scard navy" style="padding:6px 9px;">
  <h3><span class="tag t-obs">SIX ANOMALY SIGNATURES VS 9.993M NORMAL BASELINE</span> Quantifying Separate Defect &amp; Fraud Generators</h3>
  <table class="stbl">
    <thead><tr><th>Class Label (Count)</th><th class="num">Ord=B %</th><th class="num">Med USD</th><th class="num">Mean USD</th><th class="num">Med Dur</th><th class="num">Covered (MH%)</th><th class="num">Start / End Mis</th></tr></thead>
    <tbody>{''.join(a_rows)}</tbody>
  </table>
</div>
<div class="row" style="gap:9px;">
  <div class="scard blue" style="flex:1; padding:6px 9px;">
    <h3><span class="tag t-obs">FRAUD VS OP DEFECT MECHANISMS</span></h3>
    <ul>
      <li><b>ROMANCE_SCAM:</b> 500 Retail victims (100% Ord=C) hit 378 mules over 2–8 consecutive tx (med gap 215s), escalating <b>1.50x/step</b> ($33 &rarr; $1,925 med).</li>
      <li><b>FAT_FINGER_MISTAKE:</b> Multiplies pre-FX integer principal by <b>10^k</b> (+1.57 log10 mean; <b>77.5%</b> USD integers vs 38.6% normal; max <b>$15.66B</b>).</li>
    </ul>
  </div>
  <div class="scard amber" style="flex:1; padding:6px 9px;">
    <h3><span class="tag t-test">CONTROLLED TESTS &amp; NULL RESULT</span></h3>
    <ul>
      <li><b>CEO &amp; ARS Separation:</b> CEO = 100% Business (3.2x ticket); ARS matches normal amounts ($3,977) with <b>0 prior history</b>&mdash;BOTH collapse Covered (1.5–2.0%) even controlling for N&ge;3!</li>
      <li><b>DUPLICATE (185 bursts, 2–5 copies) vs RECALL* (Null):</b> Undocumented <code>CUSTOMER_RECALL</code> (317) fails 7 KS tests (<b>p=0.23–0.90</b>) &amp; LGBM (<b>AUC=0.476</b>).</li>
    </ul>
  </div>
</div>
<div class="row" style="flex:1; min-height:0; gap:10px;">
  <div class="simg-box" style="flex:1.05;">
    <img src="{imgs['c21_romance_scam_steps.png']}" />
    <div class="scap"><b>1.50x Grooming Progression:</b> Geometric step scaling from $33; step 8 jumps to $1,925 via higher-tier survivors.</div>
  </div>
  <div class="simg-box" style="flex:1.48;">
    <img src="{imgs['c23_fat_finger_and_scam_routing.png']}" />
    <div class="scap"><b>Magnitude Shift vs Covered Collapse:</b> Fat-finger injects +10^k errors; ALL 3 scam classes cut Covered 6–37x vs normal &amp; op errors.</div>
  </div>
</div>
"""
slides.append(slide_wrap(12, "Feature Group 10 of 10 &bull; Anomaly Taxonomy & Signatures", "Distinct Defect Engines: 1.50x Grooming, 10^k Shifts, Consecutive Clones & Unpredictable Recalls", "Ref: D10 (c21–c24) &bull; Conf: HIGH/NULL", "fg", s12_body))


# SLIDE 13: Cross-Cutting Synthesis — Shared 1,068-Account Mule Network & Contagion
mv = an["mule_venn"]
s13_svg = """
<svg viewBox="0 0 510 315" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" style="font-family:'Segoe UI',Arial,sans-serif;">
  <rect x="8" y="8" width="150" height="58" rx="6" fill="#FEE2E2" stroke="#DC2626" stroke-width="1.5"/>
  <text x="83" y="27" text-anchor="middle" font-size="10.5" font-weight="800" fill="#991B1B">3 Social Scams</text>
  <text x="83" y="42" text-anchor="middle" font-size="9" fill="#1E293B">ROM (2,476) + CEO (945)</text>
  <text x="83" y="55" text-anchor="middle" font-size="9" fill="#1E293B">+ ARS (649) = 4,070 tx</text>

  <rect x="8" y="88" width="150" height="58" rx="6" fill="#DBEAFE" stroke="#2563EB" stroke-width="1.5"/>
  <text x="83" y="107" text-anchor="middle" font-size="10.5" font-weight="800" fill="#1E40AF">2 Operational Defects</text>
  <text x="83" y="122" text-anchor="middle" font-size="9" fill="#1E293B">Fat-Finger (1,940)</text>
  <text x="83" y="135" text-anchor="middle" font-size="9" fill="#1E293B">+ Duplicate (679) = 2,619 tx</text>

  <!-- Center Box: Mule Ecosystem -->
  <rect x="195" y="16" width="305" height="125" rx="7" fill="#FFFBEB" stroke="#D97706" stroke-width="1.6"/>
  <text x="347" y="35" text-anchor="middle" font-size="11" font-weight="800" fill="#92400E">CROSS-CUTTING TELEMETRY SPLIT</text>
  <text x="208" y="54" font-size="9.2" fill="#0F2942"><tspan font-weight="800">1. Destination Account Graph:</tspan> Scams share 1,068 EM mules</text>
  <text x="208" y="69" font-size="9.2" fill="#DC2626" font-weight="700">   (PK 210.7x, EG 154.7x, ID 121.2x, PH 40.5x lift!)</text>
  <text x="208" y="86" font-size="9.2" fill="#0F2942"><tspan font-weight="800">2. Path End-Mismatch Signature:</tspan> ALL 3 scams = <tspan font-weight="800" fill="#0D9488">0.00%</tspan></text>
  <text x="208" y="101" font-size="9.2" fill="#0F2942">   vs Op Errors / Recall / Normal = <tspan font-weight="800">4.42–5.52%</tspan>!</text>
  <text x="208" y="118" font-size="9.2" fill="#0F2942"><tspan font-weight="800">3. Covered Routing (Cond. N&ge;3):</tspan> Scams = 0.3–2.0%</text>
  <text x="208" y="132" font-size="9.2" fill="#0F2942">   vs Op Errors / Recall / Normal = <tspan font-weight="800">11.3–13.2%</tspan>!</text>

  <!-- Normal contagion bottom box -->
  <rect x="8" y="168" width="492" height="138" rx="7" fill="#FEF2F2" stroke="#DC2626" stroke-width="1.8"/>
  <text x="254" y="189" text-anchor="middle" font-size="11.5" font-weight="900" fill="#991B1B">HIGH-RISK FINDING: MULE CONTAGION IN UNLABELED "NORMAL" TRAFFIC</text>
  <text x="24" y="210" font-size="9.5" fill="#1E293B">&#9656; <tspan font-weight="800">441 of the 1,068</tspan> known CEO/ARS/Romance mule accounts (41.3%) also receive</text>
  <text x="24" y="226" font-size="9.5" fill="#DC2626" font-weight="800">   7,828 unlabeled 'Normal' transactions (~1.92x the 4,070 labeled scam transactions!)</text>
  <text x="24" y="246" font-size="9.5" fill="#1E293B">&#9656; Overlap Venn Proof: <tspan font-weight="800">54 mules</tspan> launder all 3 scam typologies simultaneously; <tspan font-weight="800">171</tspan></text>
  <text x="24" y="261" font-size="9.5" fill="#1E293B">   shared across CEO &amp; ARS; <tspan font-weight="800">82</tspan> across CEO &amp; Romance; <tspan font-weight="800">45</tspan> across ARS &amp; Romance.</text>
  <text x="24" y="281" font-size="9.3" fill="#0F2942">&#9656; <tspan font-weight="800">Why Covered Collapses 6–37x:</tspan> Conditional on N&ge;3, EM mule destinations &amp; victim</text>
  <text x="24" y="295" font-size="9.3" fill="#0F2942">   banks route serial MT103 chains rather than institutional MT202COV cover corridors.</text>

  <line x1="158" y1="37" x2="193" y2="55" stroke="#DC2626" stroke-width="2"/>
  <line x1="158" y1="117" x2="193" y2="105" stroke="#2563EB" stroke-width="2"/>
</svg>
"""

s13_body = f"""
<div class="row" style="flex:1;">
  <div class="col" style="flex:0.95;">
    <div class="simg-box" style="flex:1.25; padding:8px;">
      {s13_svg}
    </div>
    <div class="scard amber" style="padding:7px 9px;">
      <h3><span class="tag t-obs">EMPIRICAL VENN &amp; CONTAGION COUNTS</span></h3>
      <ul>
        <li><b>Overlap Breakdown (1,068 Total):</b> CEO-only 327, Romance-only 197, ARS-only 192, CEO&amp;ARS 171, CEO&amp;ROM 82, All 3 Scams 54, ARS&amp;ROM 45.</li>
        <li><b>Extreme EM Skew:</b> PK <b>210.7x</b> (3.0% vs 0.014%), EG <b>154.7x</b> (13.3% vs 0.086%), ID <b>121.2x</b> (8.85% vs 0.073%), PH <b>40.5x</b> vs normal baselines.</li>
      </ul>
    </div>
  </div>
  <div class="col" style="flex:1.10;">
    <div class="simg-box" style="flex:1.35;">
      <img src="{imgs['c22_mule_pool_venn_countries.png']}" />
      <div class="scap"><b>1,068 Shared EM Mules:</b> Left: Venn intersection across CEO/ARS/Romance (441 hit normal traffic). Right: Extreme EM concentration lifts.</div>
    </div>
    <div class="scard teal" style="padding:7px 9px;">
      <h3><span class="tag t-int">SYNTHESIS TAKEAWAY</span> Two Distinct Engineering Pipelines</h3>
      <ul>
        <li><b>Pipeline A (Pre-Corruption Fraud Graph):</b> Generates clean scam chains targeting a shared 1,068 EM-skewed mule ring, explains slow corridor duration (571–1,148s med), skips endpoint corruption (<b>0.0%</b> end mismatch), and routes serially without Covered (0.3–2.0%).</li>
        <li><b>Pipeline B (Post-Generation Operational Perturbation):</b> Copies or shifts normal transactions (<code>DUPLICATE</code>, <code>FAT_FINGER</code>), inheriting ~5% end mismatch, ~11–13% Covered, and standard corridor speeds!</li>
      </ul>
    </div>
  </div>
</div>
"""
slides.append(slide_wrap(13, "Cross-Cutting Graph & Routing Synthesis", "1,068 Shared EM Mules Connect All 3 Scams & Contaminate 7,828 'Normal' Wires", "Cross-Cutting Synthesis", "syn", s13_body))


# SLIDE 14: Synthetic Deviations, Rejected Hypotheses & Honest Unexplained Residuals
s14_body = f"""
<div class="row" style="flex:1;">
  <div class="col" style="flex:1.0;">
    <div class="scard amber">
      <h3><span class="tag t-test">SYNTHETIC DEVIATIONS VS REAL SWIFT</span> Known Simulator Artifacts</h3>
      <ul>
        <li><b>Beneficiary Text Locale (0.98% Match):</b> Faker street/city/country ignores Beneficiary BIC8 99% of the time (skewing to CN 14.5%, BR 8.9%), even while IBAN prefix is 100% accurate.</li>
        <li><b>Middle East Western Workweek:</b> AE, SA, IL, EG follow Mon–Fri peaks (Fri ~18.9%) instead of real-world Sun–Thu banking days.</li>
        <li><b>Over-Clean Corridor Physics:</b> Additive <code>Gamma(2,5)</code> noise (std=7.04s) with zero intraday congestion; domestic wires take 36.8% multi-hop paths; offshore USD triggers mandatory FX; dual-role accounts flip B/C type ~50/50.</li>
      </ul>
    </div>
    <div class="scard red" style="flex:1;">
      <h3><span class="tag t-gap">HONEST UNEXPLAINED RESIDUALS</span> Transparent Open Questions</h3>
      <ul>
        <li><b>1. Undocumented CUSTOMER_RECALL Mechanism (n=317):</b> Indistinguishable across all 7 KS tests (<b>p=0.234–0.902</b>) and 16-feature 5-fold LightGBM (<b>AUC=0.4764</b>); sole marginal flag is QFLAG='N' (11.04% vs 7.64%, Chi2 p=0.023)&mdash;likely keyed off external/unlogged dispute events.</li>
        <li><b>2. ~600s Asymmetric Reverse-Pair Jumps:</b> 72.7% of reverse corridors agree &le;10s, yet ~27% show discrete ~600s one-way offsets (<code>0309CHZ0&harr;0035AU20</code>: 632s vs 30s).</li>
        <li><b>3. Why 5% Endpoint Mismatch Only Hits Normal/Op Rows:</b> Absent (0.0%) on end endpoints of ALL 3 scams, yet ~5.0% everywhere else without an explicit schema trigger.</li>
        <li><b>4. Exogenous Monday–Sunday Volume Plateaus:</b> Weekly swings from 91k to 218k/day occur independently of median ticket size ($3,663) or B2B share (22.7%).</li>
        <li><b>5. Within-Corridor Value Noise &amp; +3.4% C2B Edge:</b> Remaining 61.5% log-normal variance (&sigma;=2.29) &amp; slight C2B vs B2C median edge ($3,236 vs $3,118) reflect missing commercial metadata.</li>
      </ul>
    </div>
    <div class="scard navy" style="padding:6px 9px;">
      <h3><span class="tag t-obs">CUSTOMER_RECALL KS NULL TABLE</span> n = 317 vs 200k Normal (Plus Epoch KS p = 0.8406)</h3>
      <table class="stbl" style="font-size:8.2px;">
        <thead><tr><th>Tested Feature</th><th class="num">Recall Mean</th><th class="num">Normal Mean</th><th class="num">KS Stat</th><th class="num">KS p-value</th></tr></thead>
        <tbody>
          {''.join(f"<tr><td><b>{k}</b></td><td class='num'>{v['cr_mean']:,.2f}</td><td class='num'>{v['norm_mean']:,.2f}</td><td class='num'>{v['ks_stat']:.4f}</td><td class='num'><b>{v['ks_p']:.4f}</b></td></tr>" for k, v in an['customer_recall_ks'].items() if k != 'epoch_seconds')}
        </tbody>
      </table>
    </div>
  </div>
  <div class="col" style="flex:1.05;">
    <div class="simg-box" style="flex:1.12;">
      <img src="{imgs['c24_customer_recall_null.png']}" />
      <div class="scap"><b>Rigorous Null Verification on Undocumented Label:</b> Every core numeric feature &amp; 5-fold GBDT model confirms CUSTOMER_RECALL is orthogonal to logged wire attributes.</div>
    </div>
    <div class="scard navy" style="padding:7px 9px;">
      <h3><span class="tag t-obs">FORMALLY REJECTED NAIVE HYPOTHESES</span></h3>
      <ul>
        <li><b>"Higher duration in QFLAG='N' means telemetry outages delay funds"</b> &rarr; Debunked by N stratification (+51s drops to &lt;9s).</li>
        <li><b>"Invoice currency (e.g. USD vs EUR) alters USD ticket size or FX count"</b> &rarr; Debunked (+0.00001 R&sup2; on value; 100% BIC_PATH control on FX).</li>
        <li><b>"Isle of Man banks cover half their wires for jurisdiction privacy"</b> &rarr; Debunked by 1 bank (<code>0004IMD0</code>) bank FE dominance.</li>
      </ul>
    </div>
  </div>
</div>
"""
slides.append(slide_wrap(14, "Limits, Synthetic Artifacts & Honest Negative Results", "Strict Falsification Separates Simulator Shortcuts From Five Explicit Unexplained Patterns", "Deviations & Open Gaps", "warn", s14_body))


# SLIDE 15: Practical & Economic Recommendations for Risk, Fraud & Operations
s15_body = f"""
<div class="scard rule">
  <span class="eq">Translating Reverse-Engineered Invariances Into Immediate Payment-Ops &amp; AML Guardrails</span>
  <div style="font-size:9.3px; color:#334155;"><b>Scope Principle:</b> Recommendations respect synthetic dataset limits (benchmarking analytics pipelines rather than setting production KYC thresholds directly).</div>
</div>
<div class="row" style="flex:1; gap:10px;">
  <div class="col" style="flex:1;">
    <div class="scard red" style="flex:1; display:flex; flex-direction:column; justify-content:space-between; padding:10px 12px;">
      <div>
        <h3 style="font-size:10.5px;"><span class="tag t-act">PILLAR 1 &bull; GRAPH AML CONTAGION</span> Screen Mule Union (+7,828 Unlabeled Hits)</h3>
        <ul style="font-size:10.2px; line-height:1.42;">
          <li><b>Gap in Single-Rule Alerts:</b> ARS steals normal ticket sizes ($3,977 med vs $3,663) and Romance starts tiny ($33 step 1)&mdash;evading static USD amount thresholds entirely.</li>
          <li><b>High-ROI Fix:</b> Deploy <b>first-time corridor (Ord, Ben) velocity checks + mule-hop graph contagion</b> across the 1,068 EM-concentrated destinations (capturing 7,828 high-risk normal-labeled transfers across 441 reused accounts).</li>
          <li><b>Expected Impact &amp; Limit:</b> Links <b>54 triple-typology</b> &amp; <b>298 dual-typology</b> mule accounts in real time; requires daily graph refresh to avoid blocking legitimate EM family remittances.</li>
        </ul>
      </div>
      <div style="font-size:8.6px; color:#64748B; border-top:1px solid #E2E8F0; padding-top:4px;"><b>Target Scope:</b> 4,070 labeled scam rows + 7,828 normal hits &bull; <b>Rule Basis:</b> 0 prior history (ARS) &amp; 34–211x EM destination lift</div>
    </div>
    <div class="scard amber" style="flex:1; display:flex; flex-direction:column; justify-content:space-between; padding:10px 12px;">
      <div>
        <h3 style="font-size:10.5px;"><span class="tag t-act">PILLAR 2 &bull; MULTI-SIGNAL ROUTING</span> Covered Drop + 0% End-Mismatch</h3>
        <ul style="font-size:10.2px; line-height:1.42;">
          <li><b>Unexpected Behavioral Indicator:</b> Holding <code>N&ge;3</code> fixed, social-engineering flows drop Covered usage <b>6–37x</b> (0.3–2.0% vs 11.3%) and never exhibit post-path end aliasing (<b>0.0% vs 5.0%</b>).</li>
          <li><b>Operational Triage:</b> Combine <b>new retail-to-EM corridor + absent Covered leg + step-up burst timing</b> (med 215s between Romance steps) for automated confirmation-of-payee delays.</li>
          <li><b>Execution Headroom:</b> Scam corridors settle slowly (median <b>571–1,148s</b> vs 56s global baseline)&mdash;providing a 10–19 minute pre-settlement hold window before irreversibility.</li>
        </ul>
      </div>
      <div style="font-size:8.6px; color:#64748B; border-top:1px solid #E2E8F0; padding-top:4px;"><b>Target Scope:</b> Multi-hop retail/CEO/ARS wires &bull; <b>Caveat:</b> Low Covered alone also reflects 50% of small ordering banks</div>
    </div>
  </div>
  <div class="col" style="flex:1;">
    <div class="scard blue" style="flex:1; display:flex; flex-direction:column; justify-content:space-between; padding:10px 12px;">
      <div>
        <h3 style="font-size:10.5px;"><span class="tag t-act">PILLAR 3 &bull; SANCTIONS &amp; DATA QUALITY</span> Fix NLP Geo &amp; Fat-Finger Controls</h3>
        <ul style="font-size:10.2px; line-height:1.42;">
          <li><b>Critical Model Hazard:</b> Never train NLP geo-screening on raw <code>BENEFICIARY_DETAILS</code> in synthetic/noisy feeds (only <b>0.984%</b> matches <code>BENEFICIARY_BIC8</code> vs 100.0% on Ordering &amp; IBAN prefixes)! Enforce <b>BIC8 + IBAN prefix precedence</b>.</li>
          <li><b>Pre-FX Typing Guardrail:</b> 1,940 fat-finger shifts inflate raw volume mean <b>+22.1%</b> via exact powers of 10 in native integers (<b>77.5%</b> in USD vs 38.6% normal; max $15.66B).</li>
          <li><b>Parser &amp; Pre-Release Check:</b> Reassemble <code>&lt;line_break:35&gt;</code> (44.0%/44.2%) before screening, and pause orders &ge;100x Ordering-Account historical p95 prior to FX conversion.</li>
        </ul>
      </div>
      <div style="font-size:8.6px; color:#64748B; border-top:1px solid #E2E8F0; padding-top:4px;"><b>Target Scope:</b> 20M Party Detail strings &amp; 1,940 fat-finger rows &bull; <b>Impact:</b> Eliminates 99% Ben NLP geo-error</div>
    </div>
    <div class="scard teal" style="flex:1; display:flex; flex-direction:column; justify-content:space-between; padding:10px 12px;">
      <div>
        <h3 style="font-size:10.5px;"><span class="tag t-act">PILLAR 4 &bull; UNCONFOUNDED ANALYTICS</span> Separate SLA Monitoring From Telemetry</h3>
        <ul style="font-size:10.2px; line-height:1.42;">
          <li><b>Eliminate Collider Filters:</b> Retain <code>QFLAG='N'</code> (767,204 rows) when auditing network SLAs&mdash;dropping incomplete UETR traces falsely inflates Direct shares <b>+1.84 pp</b> and invents a spurious +50s delay.</li>
          <li><b>Direct-Corridor SLA Benchmarking:</b> Grade settlement desks against <b>Directed-Pair Fixed Effects</b> (R&sup2;=0.99996, std=7.0s) &amp; monitor Covered speedup (0.62–0.71x at N=3) rather than pooled trimodal means.</li>
          <li><b>Dispute Readiness (CUSTOMER_RECALL):</b> Since recalls (n=317) yield chance <b>AUC=0.476</b> on wire fields, integrate external ERP/invoice dispute APIs instead of overfitting interbank models.</li>
        </ul>
      </div>
      <div style="font-size:8.6px; color:#64748B; border-top:1px solid #E2E8F0; padding-top:4px;"><b>Target Scope:</b> 10M SLA telemetry &amp; 317 recall tickets &bull; <b>Rule Basis:</b> Stratified N=2..6 invariance &amp; KS nulls</div>
    </div>
  </div>
</div>
"""
slides.append(slide_wrap(15, "Operational & Economic Decision Implications", "Four Evidence-Backed Controls for Fraud Triage, Sanctions NLP, Pre-FX Safeguards & SLA Ops", "Recommendations", "syn", s15_body))


# SLIDE 16: Conclusion & Full Reproducibility Archive
s16_rows = [
    ("Group 1 &bull; Audit & Prep", "6 anomaly labels (7,006), row-0 DISCLAIMER, 1 exact dup pair", "+22.1% raw USD mean; +1.84 pp QFLAG Direct collider bias", "HIGH (10M Census)"),
    ("Group 2 &bull; Entity Graph", "391k Ord (100% 1:1 BIC), 2.59M Ben (2.38% multi-BIC)", "P(Ben=B) ~28.8% indep.; 253k overlap match BIC 99.9%, B/C 49.6%", "HIGH (10M Census)"),
    ("Group 3 &bull; Temporal", "Uniform 08:00 & 17–19h local curve + Mon–Sun waves", "r = 0.994–1.000 (15 hubs); ME Mon–Fri deviation; flat value", "HIGH (92 Days)"),
    ("Group 4 &bull; Value (USD)", "Pre-FX ln(USD) = 6.06 + 2*B_o + 2*B_b + Ctry_o + N(0,2.29²)", "R²=0.353; Curr/Recvr ΔR² < 0.0002; 38.6% USD ints; daily EUR", "HIGH (9.993M Clean)"),
    ("Group 5 &bull; FX & Currency", "Adjacent HomeCurr diffs along visible BIC8_PATH", "9,999,999 / 10M exact (100% w/ ER=ERN); 88.5% Ord Home cross-zone", "EXACT (100.0%)"),
    ("Group 6 &bull; Routing", "Direct <=> N=2; N-2 = floor(Exp(1)); Bank Covered policy", "1/e=0.368 ratio; Ord BIC Covered R²=0.326 (IM 1-bank debunked)", "HIGH (Exact Law)"),
    ("Group 7 &bull; Path & QFLAG", "QFLAG='N' <=> '...' (0 errs); dots ~ NegBin(N-1, 0.95)", "Two-Layer Split: BIC8=Acct/IBAN, Path=FX & Dur (MAE 5.6s); +51s nullified", "HIGH (0 Mismatches)"),
    ("Group 8 &bull; Duration", "Directed-Pair Base(Path[0]->Path[-1]) + Gamma(2,5) noise", "Path FE R²=0.99997 (RMSE 6.0s); N=3 r=0.982; Covered ~35% speedup", "HIGH (R²=0.99997)"),
    ("Group 9 &bull; Text & Tokens", "Ord 100% geo-match vs Ben 0.984% Faker locale mismatch", "IBAN SEPA/GB/CH 90–96% vs US/APAC 0%; 10 fin-words free text only", "HIGH (20M Strings)"),
    ("Group 10 &bull; Anomalies", "5 distinct defect/scam rules + 1,068 EM mules (+7.8k normal hits)", "1.50x Romance; 10^k Fat-Finger; CUSTOMER_RECALL* AUC=0.476 NULL", "HIGH / NULL (Recall)"),
]
s16_table_html = "".join(
    f"<tr><td><b>{a}</b></td><td>{b}</td><td>{c}</td><td class='num'><b>{d}</b></td></tr>"
    for a, b, c, d in s16_rows
)

s16_body = f"""
<div class="scard" style="background:rgba(255,255,255,0.97); padding:7px 9px;">
  <h3 style="color:#0F2942;">COMPLETE 10-GROUP REVERSE-ENGINEERED SPECIFICATION SUMMARY</h3>
  <table class="stbl" style="font-size:8.35px;">
    <thead><tr><th style="width:16%">Feature Group</th><th style="width:31%">Reconstructed Exact Mechanism / Equation</th><th style="width:39%">Quantitative Size, Controls &amp; Limits</th><th class="num" style="width:14%">Evidence Grade</th></tr></thead>
    <tbody>{s16_table_html}</tbody>
  </table>
</div>
<div class="row" style="gap:10px;">
  <div class="scard dark-panel" style="flex:1; background:rgba(255,255,255,0.10); border:1px solid rgba(255,255,255,0.22); color:#F8FAFC; padding:10px 12px;">
    <h3 style="color:#38BDF8; font-size:10.8px;">REPRODUCIBILITY ARCHIVE &amp; DELIVERABLE PACKAGE</h3>
    <ul style="color:#E2E8F0; font-size:9.7px; line-height:1.45;">
      <li><b>ETL &amp; Audits (10M Census):</b> <code>scripts/01</code>–<code>13</code> partitioned all 10M CSV rows into 40 Parquet shards and computed <code>work/audit_summary.json</code> &amp; <code>work/master_metrics.json</code> (full audit runs cleanly from raw CSV).</li>
      <li><b>Visualizations &amp; PDFs:</b> <code>scripts/14_generate_charts.py</code> (24 PNGs at 220 DPI), <code>scripts/15_build_dashboards.py</code> &rarr; <b>10 Standalone Dashboards</b> (<code>Swift_Datathon_2026_Dashboards.pdf</code>), <code>scripts/16_build_slides.py</code> &rarr; <b>16 Executive Slides</b> (<code>Swift_Datathon_2026_Presentation.pdf</code>).</li>
      <li><b>Methodological Discipline:</b> Simple closed-form algebraic &amp; regression baselines resolved 9 of 10 groups; complex GBDT reserved strictly to validate the <code>CUSTOMER_RECALL</code> null result.</li>
    </ul>
  </div>
  <div class="scard dark-panel" style="flex:0.85; background:rgba(13,148,136,0.22); border:1px solid rgba(45,212,191,0.40); color:#F8FAFC; padding:10px 12px;">
    <h3 style="color:#5EEAD4; font-size:10.8px;">STRICT COMPETITION &amp; INTEGRITY GUARANTEE</h3>
    <ul style="color:#E2E8F0; font-size:9.7px; line-height:1.45;">
      <li><b>Zero External Data:</b> Verified 100% self-contained analysis of <code>swift_umdac_datathon_2026_fictitious_payment_data.csv</code> (4.12 GB, 10,000,000 rows, 21 columns).</li>
      <li><b>Visual &amp; Numeric QA:</b> Every quantitative claim verified against script outputs and every page of both PDFs inspected visually via PNG rasterization prior to delivery.</li>
      <li><b>Transparency Standard:</b> Observed statistics are kept strictly distinct from economic hypotheses, synthetic simulator quirks, and five explicitly documented unresolved patterns.</li>
    </ul>
  </div>
</div>
"""
slides.append(slide_wrap(16, "Conclusion & Verification Ledger", "Complete Equation Stack Verified Across 10 Million Synthetic Transactions", "Conclusion & Archive", "", s16_body, dark=True))


full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>UMDAC Datathon 2026 - Executive Presentation (Swift Case Study)</title>
<style>{CSS}</style>
</head>
<body>
{''.join(slides)}
</body>
</html>
"""

html_path = os.path.abspath("work/slides.html")
pdf_path = os.path.abspath("Swift_Datathon_2026_Presentation.pdf")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(full_html)
print(f"Wrote {html_path} ({len(full_html)/1024:.1f} KB)")

candidates = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
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
