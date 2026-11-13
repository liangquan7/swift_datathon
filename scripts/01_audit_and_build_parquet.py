import json
import os
import re
import time
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

CSV_PATH = "swift_umdac_datathon_2026_fictitious_payment_data.csv"
WORK_DIR = "work"
DETAILS_DIR = os.path.join(WORK_DIR, "details_parquet")
CORE_DIR = os.path.join(WORK_DIR, "core_parquet")
os.makedirs(DETAILS_DIR, exist_ok=True)
os.makedirs(CORE_DIR, exist_ok=True)

CHUNK_SIZE = 250_000

RE_CC = re.compile(r"<country_code:([A-Z]{2})>")
RE_CNAME = re.compile(r"<country_name:([A-Z]{2}):([^>]+)>")
RE_CITY = re.compile(r"<city:([A-Z]{2}):")
RE_POST = re.compile(r"<post_code:([A-Z]{2}):")
RE_FIN = re.compile(r"<finance_word:([^>]+)>")
RE_TAG = re.compile(r"<([a-z_]+):")
RE_BIC8 = re.compile(r"^[0-9]{4}[A-Z]{2}[A-Z0-9]{2}$")


def extract_token_features(df: pd.DataFrame, prefix: str, det_col: str, tok_col: str) -> dict:
    det = df[det_col]
    tok = df[tok_col]

    cc = tok.str.extract(RE_CC, expand=False).fillna("")
    cname_df = tok.str.extract(RE_CNAME)
    cname_cc = cname_df[0].fillna("")
    cname_lang = cname_df[1].fillna("")
    city_cc = tok.str.extract(RE_CITY, expand=False).fillna("")
    post_cc = tok.str.extract(RE_POST, expand=False).fillna("")
    fin_word = tok.str.extract(RE_FIN, expand=False).fillna("")

    # Best inferred party country from tokens (priority: explicit country_code -> country_name -> city -> post_code)
    inferred_cc = cc.where(cc != "", cname_cc)
    inferred_cc = inferred_cc.where(inferred_cc != "", city_cc)
    inferred_cc = inferred_cc.where(inferred_cc != "", post_cc)

    has_iban = tok.str.contains("<iban:", regex=False)
    has_lb35 = tok.str.contains("<line_break:35>", regex=False)
    has_trunc = tok.str.contains("<truncation:", regex=False)
    has_biz = tok.str.contains("<business:", regex=False) | tok.str.contains("<company_keyword:", regex=False)
    has_person = tok.str.contains("<first_name:", regex=False) | tok.str.contains("<last_name:", regex=False)
    has_street = tok.str.contains("<street:", regex=False)
    has_acct_line = det.str.startswith("/")
    has_double_slash = det.str.startswith("//")
    is_structured_123 = det.str.contains(r"(?:^|\n)1/", regex=True)

    return {
        f"{prefix}_country_code": cc,
        f"{prefix}_country_name_cc": cname_cc,
        f"{prefix}_country_lang": cname_lang,
        f"{prefix}_city_cc": city_cc,
        f"{prefix}_post_cc": post_cc,
        f"{prefix}_inferred_cc": inferred_cc,
        f"{prefix}_fin_word": fin_word,
        f"{prefix}_has_iban": has_iban,
        f"{prefix}_has_lb35": has_lb35,
        f"{prefix}_has_trunc": has_trunc,
        f"{prefix}_has_biz": has_biz,
        f"{prefix}_has_person": has_person,
        f"{prefix}_has_street": has_street,
        f"{prefix}_has_acct_line": has_acct_line,
        f"{prefix}_has_double_slash": has_double_slash,
        f"{prefix}_is_struct_123": is_structured_123,
        f"{prefix}_det_len": det.str.len().astype(np.int16),
        f"{prefix}_n_lines": (det.str.count("\n") + 1).astype(np.int8),
    }


def main():
    t0 = time.time()
    empty_counts = Counter()
    anomaly_counts = Counter()
    currency_counts = Counter()
    tx_type_counts = Counter()
    qflag_counts = Counter()
    ord_type_counts = Counter()
    ben_type_counts = Counter()
    part_counts = Counter()
    fx_counts = Counter()
    disclaimer_counts = Counter()
    fin_word_counts = Counter()
    token_tag_counts = Counter()
    bic_bad_counts = Counter()
    ord_bic_char7 = Counter()
    ben_bic_char7 = Counter()
    ord_acct_nonfake = 0
    ben_acct_nonfake = 0

    full_hashes = set()
    n_exact_full_dups = 0
    no_ts_hashes = set()
    n_no_ts_dups = 0

    epoch_min = 10**18
    epoch_max = -1
    amt_min = float("inf")
    amt_max = float("-inf")
    amt_nonpos = 0
    dur_min = 10**18
    dur_max = -1
    dur_neg = 0

    total_rows = 0
    reader = pd.read_csv(
        CSV_PATH,
        chunksize=CHUNK_SIZE,
        dtype=str,
        keep_default_na=False,
    )

    for idx, ch in enumerate(reader):
        n = len(ch)
        for c in ch.columns:
            empty_counts[c] += int((ch[c] == "").sum())

        disclaimer_counts.update(ch["DISCLAIMER"].value_counts().to_dict())
        anomaly_counts.update(ch["ANOMALY_TYPE"].value_counts().to_dict())
        currency_counts.update(ch["CURRENCY_CODE"].value_counts().to_dict())
        tx_type_counts.update(ch["TRANSACTION_TYPE_CODE"].value_counts().to_dict())
        qflag_counts.update(ch["CONTINUITY_QFLAG"].value_counts().to_dict())
        ord_type_counts.update(ch["ORDERING_TYPE"].value_counts().to_dict())
        ben_type_counts.update(ch["BENEFICIARY_TYPE"].value_counts().to_dict())
        part_counts.update(ch["PARTICIPANT_COUNT"].value_counts().to_dict())
        fx_counts.update(ch["CURRENCY_CONVERSION_COUNT"].value_counts().to_dict())

        # Numeric parsing & sanity checks
        epoch = pd.to_numeric(ch["EPOCH_SECONDS"], errors="raise").astype(np.int64)
        amt = pd.to_numeric(ch["USD_AMOUNT"], errors="raise").astype(np.float64)
        dur = pd.to_numeric(ch["TOTAL_DURATION"], errors="raise").astype(np.int32)
        fx = pd.to_numeric(ch["CURRENCY_CONVERSION_COUNT"], errors="raise").astype(np.int16)
        part = pd.to_numeric(ch["PARTICIPANT_COUNT"], errors="raise").astype(np.int16)

        epoch_min = min(epoch_min, int(epoch.min()))
        epoch_max = max(epoch_max, int(epoch.max()))
        amt_min = min(amt_min, float(amt.min()))
        amt_max = max(amt_max, float(amt.max()))
        amt_nonpos += int((amt <= 0).sum())
        dur_min = min(dur_min, int(dur.min()))
        dur_max = max(dur_max, int(dur.max()))
        dur_neg += int((dur < 0).sum())

        # Account prefix checks
        ord_acct_nonfake += int((~ch["ORDERING_ACCOUNT"].str.startswith("_FAKE_")).sum())
        ben_acct_nonfake += int((~ch["BENEFICIARY_ACCOUNT"].str.startswith("_FAKE_")).sum())

        # BIC8 checks
        bic_bad_counts["ORDERING_BIC8"] += int((~ch["ORDERING_BIC8"].str.match(RE_BIC8)).sum())
        bic_bad_counts["BENEFICIARY_BIC8"] += int((~ch["BENEFICIARY_BIC8"].str.match(RE_BIC8)).sum())
        ord_bic_char7.update(ch["ORDERING_BIC8"].str[7].value_counts().to_dict())
        ben_bic_char7.update(ch["BENEFICIARY_BIC8"].str[7].value_counts().to_dict())

        # Exact duplicate tracking (excluding DISCLAIMER)
        cols_no_disc = [c for c in ch.columns if c != "DISCLAIMER"]
        row_h = pd.util.hash_pandas_object(ch[cols_no_disc], index=False).to_numpy()
        for h in row_h:
            hi = int(h)
            if hi in full_hashes:
                n_exact_full_dups += 1
            else:
                full_hashes.add(hi)

        cols_no_ts = [c for c in cols_no_disc if c != "EPOCH_SECONDS"]
        nots_h = pd.util.hash_pandas_object(ch[cols_no_ts], index=False).to_numpy()
        for h in nots_h:
            hi = int(h)
            if hi in no_ts_hashes:
                n_no_ts_dups += 1
            else:
                no_ts_hashes.add(hi)

        # Save details parquet
        row_ids = np.arange(total_rows, total_rows + n, dtype=np.int32)
        det_df = pd.DataFrame({
            "row_id": row_ids,
            "ORDERING_DETAILS": ch["ORDERING_DETAILS"],
            "ORDERING_TOKENS": ch["ORDERING_TOKENS"],
            "BENEFICIARY_DETAILS": ch["BENEFICIARY_DETAILS"],
            "BENEFICIARY_TOKENS": ch["BENEFICIARY_TOKENS"],
        })
        det_df.to_parquet(
            os.path.join(DETAILS_DIR, f"part_{idx:02d}.parquet"),
            index=False,
            compression="snappy",
        )

        # Extract token tags inventory on sample or all
        for tcol in ["ORDERING_TOKENS", "BENEFICIARY_TOKENS"]:
            tags = ch[tcol].str.findall(RE_TAG).explode().dropna()
            token_tag_counts.update(tags.value_counts().to_dict())

        ord_feat = extract_token_features(ch, "ord", "ORDERING_DETAILS", "ORDERING_TOKENS")
        ben_feat = extract_token_features(ch, "ben", "BENEFICIARY_DETAILS", "BENEFICIARY_TOKENS")
        fin_word_counts.update(ord_feat["ord_fin_word"][ord_feat["ord_fin_word"] != ""].value_counts().to_dict())
        fin_word_counts.update(ben_feat["ben_fin_word"][ben_feat["ben_fin_word"] != ""].value_counts().to_dict())

        # Path features
        has_gap = ch["BIC8_PATH"].str.contains("...", regex=False)
        path_bics_cnt = ch["BIC8_PATH"].str.count(";") + 1 - has_gap.astype(np.int16)

        core_data = {
            "row_id": row_ids,
            "epoch_seconds": epoch,
            "currency_code": ch["CURRENCY_CODE"],
            "usd_amount": amt,
            "ordering_bic8": ch["ORDERING_BIC8"],
            "beneficiary_bic8": ch["BENEFICIARY_BIC8"],
            "bic8_path": ch["BIC8_PATH"],
            "ord_bic_bank": ch["ORDERING_BIC8"].str[:4],
            "ord_bic_country": ch["ORDERING_BIC8"].str[4:6],
            "ord_bic_loc": ch["ORDERING_BIC8"].str[6:7],
            "ord_bic_branch": ch["ORDERING_BIC8"].str[7:8],
            "ben_bic_bank": ch["BENEFICIARY_BIC8"].str[:4],
            "ben_bic_country": ch["BENEFICIARY_BIC8"].str[4:6],
            "ben_bic_loc": ch["BENEFICIARY_BIC8"].str[6:7],
            "ben_bic_branch": ch["BENEFICIARY_BIC8"].str[7:8],
            "path_has_gap": has_gap,
            "path_visible_bics": path_bics_cnt.astype(np.int16),
            "total_duration": dur,
            "continuity_qflag": ch["CONTINUITY_QFLAG"],
            "fx_count": fx,
            "participant_count": part,
            "tx_type": ch["TRANSACTION_TYPE_CODE"],
            "ordering_account": ch["ORDERING_ACCOUNT"],
            "beneficiary_account": ch["BENEFICIARY_ACCOUNT"],
            "ordering_type": ch["ORDERING_TYPE"],
            "beneficiary_type": ch["BENEFICIARY_TYPE"],
            "anomaly_type": ch["ANOMALY_TYPE"],
        }
        core_data.update(ord_feat)
        core_data.update(ben_feat)
        core_df = pd.DataFrame(core_data)
        core_df.to_parquet(
            os.path.join(CORE_DIR, f"part_{idx:02d}.parquet"),
            index=False,
            compression="snappy",
        )

        total_rows += n
        if (idx + 1) % 5 == 0:
            print(f"Chunk {idx+1}: {total_rows:,} rows processed ({time.time()-t0:.1f}s)")

    summary = {
        "total_rows": total_rows,
        "elapsed_seconds": round(time.time() - t0, 2),
        "empty_counts": dict(empty_counts),
        "disclaimer_nonempty_count": total_rows - empty_counts["DISCLAIMER"],
        "anomaly_counts": dict(anomaly_counts),
        "n_currencies": len(currency_counts),
        "currency_counts": dict(currency_counts.most_common()),
        "tx_type_counts": dict(tx_type_counts),
        "qflag_counts": dict(qflag_counts),
        "ord_type_counts": dict(ord_type_counts),
        "ben_type_counts": dict(ben_type_counts),
        "part_counts": dict(part_counts),
        "fx_counts": dict(fx_counts),
        "epoch_min": epoch_min,
        "epoch_max": epoch_max,
        "amt_min": amt_min,
        "amt_max": amt_max,
        "amt_nonpos": amt_nonpos,
        "dur_min": dur_min,
        "dur_max": dur_max,
        "dur_neg": dur_neg,
        "ord_acct_nonfake": ord_acct_nonfake,
        "ben_acct_nonfake": ben_acct_nonfake,
        "bic_bad_counts": dict(bic_bad_counts),
        "ord_bic_char7": dict(ord_bic_char7),
        "ben_bic_char7": dict(ben_bic_char7),
        "n_exact_full_dups_64bit_hash": n_exact_full_dups,
        "n_no_ts_dups_64bit_hash": n_no_ts_dups,
        "token_tag_counts": dict(token_tag_counts),
        "fin_word_counts": dict(fin_word_counts.most_common()),
    }
    with open(os.path.join(WORK_DIR, "audit_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("DONE!", json.dumps({k: v for k, v in summary.items() if k not in ("currency_counts", "fin_word_counts")}, indent=2))
    print("Top 30 currencies:", list(summary["currency_counts"].items())[:30])
    print("Top 30 fin_words:", list(summary["fin_word_counts"].items())[:30])


if __name__ == "__main__":
    main()
