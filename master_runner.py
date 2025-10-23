#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from scrapers_all_providers import scrape_all_providers
from index_calculator import calculate_hcpi, format_hcpi_report, save_hcpi_results
from database_integration import save_to_database
from roi_calculator import (
    calculate_total_cost,
    calculate_basic_roi,
    calculate_utilisation_adjusted_roi,
    print_roi_summary,
)

# ──────────────────────────────────────────────────────────────────────────────
# Paths & Retention
DERIVED_DIR = Path("data/derived")
HISTORY_DIR = Path("data/history")
HCPI_DIR = Path("hcpi")
PUBLIC_DIR = Path("public")

# Keep N hourly points in hcpi_history.json (0 = keep all). Default 1 year.
HISTORY_RETENTION_HOURS = int(os.getenv("HISTORY_RETENTION_HOURS", "8760"))

def _ts_tag(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")

# ──────────────────────────────────────────────────────────────────────────────
# Normalise provider rows (canonical CSV for latest + history)
def _normalize_scrape_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical provider schema used for latest + history CSVs.
    Ensures per-GPU price lives in `effective_price_usd_per_gpu_hr`.
    """
    c = {
        "gpu": "gpu_model",
        "gpu_type": "gpu_model",
        "price_hourly_usd": "effective_price_usd_per_gpu_hr",
        "price_usd_per_gpu_hr": "effective_price_usd_per_gpu_hr",
        "price": "effective_price_usd_per_gpu_hr",
    }
    out = df.rename(columns=c).copy()
    needed = [
        "provider",
        "region",
        "gpu_model",
        "type",
        "duration",
        "gpu_count",
        "effective_price_usd_per_gpu_hr",
        "timestamp",
        "priceiq_score",
    ]
    for k in needed:
        if k not in out.columns:
            out[k] = None

    # Unify H100 SXM to H100 (display only; does not change price)
    out["gpu_model"] = out["gpu_model"].astype(str).str.replace(r"\bH100\s*SXM\b", "H100", regex=True)

    # Types
    out["gpu_count"] = pd.to_numeric(out["gpu_count"], errors="coerce").fillna(1).astype(int)
    out["effective_price_usd_per_gpu_hr"] = pd.to_numeric(out["effective_price_usd_per_gpu_hr"], errors="coerce")

    # Keep valid price rows
    out = out[out["effective_price_usd_per_gpu_hr"].notna() & (out["effective_price_usd_per_gpu_hr"] > 0)]
    return out[needed]

# Adapter for index_calculator (do NOT modify index_calculator itself)
def _prepare_index_input(df_scraped: pd.DataFrame) -> pd.DataFrame:
    """
    Schema for calculate_hcpi(): [provider, region, price_hourly_usd, gpu_count]
    """
    df = df_scraped.copy()

    if "price_hourly_usd" not in df.columns:
        if "effective_price_usd_per_gpu_hr" in df.columns:
            df["price_hourly_usd"] = pd.to_numeric(df["effective_price_usd_per_gpu_hr"], errors="coerce")
        else:
            df["price_hourly_usd"] = pd.to_numeric(df.get("price"), errors="coerce")

    for col in ["provider", "region", "gpu_count"]:
        if col not in df.columns:
            df[col] = None

    df["gpu_count"] = pd.to_numeric(df["gpu_count"], errors="coerce").fillna(1).astype(int)

    out = df[["provider", "region", "price_hourly_usd", "gpu_count"]].copy()
    out = out.dropna(subset=["provider", "region", "price_hourly_usd"])
    out = out[out["price_hourly_usd"] > 0]
    return out

# ──────────────────────────────────────────────────────────────────────────────
def _ensure_dirs():
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    HCPI_DIR.mkdir(parents=True, exist_ok=True)

def _write_latest_artifacts(df_scraped: pd.DataFrame, hcpi_result, ts: str, verbose: bool):
    _ensure_dirs()

    # 1) Provider latest
    df_out = _normalize_scrape_columns(df_scraped)
    (DERIVED_DIR / "provider_scores_latest.csv").write_text(df_out.to_csv(index=False))

    # 2) HCPI JSONs (full + summary) + latest pointers
    full_path, summary_path = save_hcpi_results(hcpi_result, prefix=f"{HCPI_DIR}/hcpi_{ts}")
    Path(HCPI_DIR / "hcpi_latest_full.json").write_bytes(Path(full_path).read_bytes())
    Path(HCPI_DIR / "hcpi_latest_summary.json").write_bytes(Path(summary_path).read_bytes())

    # 3) ROI scaffolding (unchanged)
    roi_csv = DERIVED_DIR / "roi_comparison.csv"
    snap_csv = DERIVED_DIR / "compute_calculator_snapshot.csv"
    if not roi_csv.exists():
        pd.DataFrame(
            columns=["gpu_model","gpu_count","duration","best_provider","best_region",
                     "total_cost_usd","price_per_gpu_hr","timestamp"]
        ).to_csv(roi_csv, index=False)
    if not snap_csv.exists():
        pd.DataFrame(
            columns=["gpu_model","provider_type","$/GPU-hr","TOTAL",
                     "price_lo_usd","price_md_usd","price_hi_usd","term_hours",
                     "gpu_count","region","n_quotes","source"]
        ).to_csv(snap_csv, index=False)

    if verbose:
        print(f"wrote {DERIVED_DIR/'provider_scores_latest.csv'}")
        print(f"wrote {HCPI_DIR/'hcpi_latest_full.json'}")
        print(f"wrote {HCPI_DIR/'hcpi_latest_summary.json'}")

# ──────────────────────────────────────────────────────────────────────────────
# History appenders (PROVIDER + INDEX)
def _append_history_quotes(df_scraped: pd.DataFrame, run_iso_ts: str):
    """
    Append all provider rows to data/history/provider_scores_history.csv
    Columns:
      run_timestamp, quote_timestamp, provider, region, gpu_model, type, duration,
      gpu_count, effective_price_usd_per_gpu_hr, priceiq_score
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = HISTORY_DIR / "provider_scores_history.csv"

    norm = _normalize_scrape_columns(df_scraped).copy()

    # Preserve individual quote time if scrapers provided one
    if "timestamp" in norm.columns:
        norm.rename(columns={"timestamp": "quote_timestamp"}, inplace=True)
    else:
        norm["quote_timestamp"] = pd.NaT

    # Put run timestamp first
    norm.insert(0, "run_timestamp", run_iso_ts)

    cols_order = [
        "run_timestamp", "quote_timestamp",
        "provider", "region", "gpu_model", "type", "duration", "gpu_count",
        "effective_price_usd_per_gpu_hr", "priceiq_score",
    ]
    for c in cols_order:
        if c not in norm.columns:
            norm[c] = None
    norm = norm[cols_order]

    write_header = not hist_path.exists()
    norm.to_csv(hist_path, mode="a", header=write_header, index=False)

def _append_history_indices(hcpi_result: dict):
    """
    Append index values to:
      - data/history/index_history.csv  (long CSV)
      - hcpi/hcpi_history.csv           (site CSV)
      - hcpi/hcpi_history.json          (rolling JSON with retention)
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    HCPI_DIR.mkdir(parents=True, exist_ok=True)

    ts_iso = hcpi_result.get("metadata", {}).get("timestamp") or datetime.now(timezone.utc).isoformat()
    reg = hcpi_result.get("regional_indices", {}) or {}
    row = {
        "timestamp": ts_iso,
        "us_index": hcpi_result.get("us_index"),
        "US-East": reg.get("US-East"),
        "US-Central": reg.get("US-Central"),
        "US-West": reg.get("US-West"),
    }

    # (A) long CSV under data/history/
    idx_hist_csv = HISTORY_DIR / "index_history.csv"
    pd.DataFrame([row]).to_csv(idx_hist_csv, mode="a", header=not idx_hist_csv.exists(), index=False)

    # (B) site CSV
    hcpi_hist_csv = HCPI_DIR / "hcpi_history.csv"
    pd.DataFrame([row]).to_csv(hcpi_hist_csv, mode="a", header=not hcpi_hist_csv.exists(), index=False)

    # (C) rolling JSON with retention
    hcpi_hist_json = HCPI_DIR / "hcpi_history.json"
    try:
        arr = json.loads(hcpi_hist_json.read_text())
        if not isinstance(arr, list):
            arr = []
    except Exception:
        arr = []
    arr.append(row)

    # Sort by timestamp for stability (handles backfills)
    arr = sorted(arr, key=lambda r: r.get("timestamp", ""))

    if HISTORY_RETENTION_HOURS > 0:
        # Approx: one point per hour
        arr = arr[-HISTORY_RETENTION_HOURS:]

    hcpi_hist_json.write_text(json.dumps(arr, indent=2))

# ──────────────────────────────────────────────────────────────────────────────
def run_single_scrape_and_calculate(database_url: str, verbose: bool = True):
    start_dt = datetime.now(timezone.utc)
    ts = _ts_tag(start_dt)

    if verbose:
        print("\n" + "=" * 70)
        print(f"START {start_dt.isoformat()}")
        print("=" * 70 + "\n")
        print("STEP 1: scrape providers")
        print("-" * 70 + "\n")

    df_scraped = scrape_all_providers()
    if df_scraped is None or df_scraped.empty:
        print("no data scraped")
        return None, None

    # Keep a raw snapshot (debug)
    try:
        raw_csv = f"scraped_data_{ts}.csv"
        df_scraped.to_csv(raw_csv, index=False)
        if verbose:
            print(f"saved raw scrape {raw_csv}")
    except Exception as e:
        print(f"raw csv save failed: {e}")

    # Optional ROI preview (non-blocking)
    try:
        df_cost = calculate_total_cost(df_scraped, duration_hours=1)
        if "perf_score" not in df_cost.columns:
            df_cost["perf_score"] = 1.0
        if "utilisation" not in df_cost.columns:
            df_cost["utilisation"] = 1.0
        df_roi = calculate_basic_roi(df_cost, perf_col="perf_score")
        df_u_roi = calculate_utilisation_adjusted_roi(df_roi, perf_col="perf_score", utilisation_col="utilisation")
        if verbose:
            print("\n" + "=" * 70)
            print("STEP 1b: ROI preview")
            print("-" * 70 + "\n")
            print_roi_summary(df_u_roi.head(10))
    except Exception as e:
        if verbose:
            print(f"roi step skipped: {e}")

    # ── Index calc (using adapter; calculator unchanged)
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: calculate HCPI")
        print("-" * 70 + "\n")

    df_idx = _prepare_index_input(df_scraped)
    hcpi_result = calculate_hcpi(df_idx, verbose=verbose)

    # ── Write artifacts
    try:
        _write_latest_artifacts(df_scraped, hcpi_result, ts, verbose)
    except Exception as e:
        print(f"artifact write failed: {e}")

    # ── Append histories (providers + index) — fail loudly if broken
    try:
        run_iso_ts = hcpi_result.get("metadata", {}).get("timestamp") or start_dt.isoformat()
        _append_history_quotes(df_scraped, run_iso_ts)
        _append_history_indices(hcpi_result)
    except Exception as e:
        import traceback
        print(f"history append failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise

    # ── Report
    if verbose:
        print("\n" + "=" * 70)
        print("FINAL REPORT")
        print("=" * 70 + "\n")
        print(format_hcpi_report(hcpi_result))

    # ── Persist to DB (non-blocking)
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3: persist to DB")
        print("-" * 70 + "\n")
    try:
        save_to_database(df_scraped, hcpi_result, database_url=database_url)
    except Exception as e:
        print(f"db persistence failed: {e}")

    end_dt = datetime.now(timezone.utc)
    if verbose:
        print(f"\nDONE in {(end_dt - start_dt).total_seconds():.1f}s")
        print("=" * 70 + "\n")

    return df_scraped, hcpi_result

# ──────────────────────────────────────────────────────────────────────────────
def run_continuous(interval_seconds: int, database_url: str, verbose: bool = True):
    cycle = 1
    print(f"\nHCPI continuous runner every {interval_seconds}s\n")
    while True:
        try:
            print(f"\n{'='*70}\nCYCLE #{cycle}\n{'='*70}\n")
            df_scraped, hcpi_result = run_single_scrape_and_calculate(database_url=database_url, verbose=verbose)
            if df_scraped is None or hcpi_result is None:
                print("cycle produced no results")
            cycle += 1
            next_run = datetime.now(timezone.utc).timestamp() + interval_seconds
            nxt = datetime.fromtimestamp(next_run, tz=timezone.utc)
            print(f"\nnext cycle at {nxt.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nstopped by user")
            break
        except Exception as e:
            print(f"\ncycle error: {e}")
            time.sleep(interval_seconds)

def run_scheduled_cron(database_url: str, verbose: bool = True) -> int:
    try:
        df_scraped, hcpi_result = run_single_scrape_and_calculate(database_url=database_url, verbose=verbose)
        if df_scraped is None or hcpi_result is None:
            return 1
        return 0
    except Exception as e:
        print(f"fatal error: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(description="HCPI Master Runner")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--interval", type=int, default=3600)
    parser.add_argument("--cron", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--db", type=str, default=None)
    args = parser.parse_args()

    verbose = not args.quiet
    database_url = args.db or os.getenv("HCPI_DATABASE_URL") or "sqlite:///hcpi.db"
    if verbose:
        print(f"DB URL: {database_url}")
        print(f"HISTORY_RETENTION_HOURS={HISTORY_RETENTION_HOURS}")

    if args.cron:
        sys.exit(run_scheduled_cron(database_url=database_url, verbose=verbose))
    elif args.continuous:
        run_continuous(interval_seconds=args.interval, database_url=database_url, verbose=verbose)
    else:
        run_single_scrape_and_calculate(database_url=database_url, verbose=verbose)

if __name__ == "__main__":
    main()



