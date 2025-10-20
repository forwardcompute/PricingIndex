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

DERIVED_DIR = Path("data/derived")
HISTORY_DIR = Path("data/history")
HCPI_DIR = Path("hcpi")
PUBLIC_DIR = Path("public")

def _ts_tag(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")

def _normalize_scrape_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    # unify H100 SXM under H100 label for display consistency (doesn't change price)
    out["gpu_model"] = out["gpu_model"].astype(str).str.replace(r"\bH100\s*SXM\b", "H100", regex=True)
    return out[needed]

def _ensure_dirs():
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    HCPI_DIR.mkdir(parents=True, exist_ok=True)

def _write_latest_artifacts(df_scraped: pd.DataFrame, hcpi_result, ts: str, verbose: bool):
    _ensure_dirs()
    df_out = _normalize_scrape_columns(df_scraped)
    (DERIVED_DIR / "provider_scores_latest.csv").write_text(df_out.to_csv(index=False))
    full_path, summary_path = save_hcpi_results(hcpi_result, prefix=f"{HCPI_DIR}/hcpi_{ts}")
    Path(HCPI_DIR / "hcpi_latest_full.json").write_bytes(Path(full_path).read_bytes())
    Path(HCPI_DIR / "hcpi_latest_summary.json").write_bytes(Path(summary_path).read_bytes())
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

def _append_history_quotes(df_scraped: pd.DataFrame, run_iso_ts: str):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    hist_path = HISTORY_DIR / "provider_scores_history.csv"
    norm = _normalize_scrape_columns(df_scraped).copy()
    norm.insert(0, "timestamp", run_iso_ts)
    write_header = not hist_path.exists()
    norm.to_csv(hist_path, mode="a", header=write_header, index=False)

def _append_history_indices(hcpi_result: dict):
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
    hist_csv = HCPI_DIR / "hcpi_history.csv"
    write_header = not hist_csv.exists()
    pd.DataFrame([row]).to_csv(hist_csv, mode="a", header=write_header, index=False)

    hist_json = HCPI_DIR / "hcpi_history.json"
    try:
        arr = json.loads(hist_json.read_text())
        if not isinstance(arr, list):
            arr = []
    except Exception:
        arr = []
    arr.append(row)
    arr = arr[-720:]  # keep ~30 days of hourly points
    hist_json.write_text(json.dumps(arr, indent=2))

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

    try:
        raw_csv = f"scraped_data_{ts}.csv"
        df_scraped.to_csv(raw_csv, index=False)
        if verbose:
            print(f"saved raw scrape {raw_csv}")
    except Exception as e:
        print(f"raw csv save failed: {e}")

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

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: calculate HCPI")
        print("-" * 70 + "\n")

    hcpi_result = calculate_hcpi(df_scraped, verbose=verbose)

    try:
        _write_latest_artifacts(df_scraped, hcpi_result, ts, verbose)
    except Exception as e:
        print(f"artifact write failed: {e}")

    try:
        run_iso_ts = hcpi_result.get("metadata", {}).get("timestamp") or start_dt.isoformat()
        _append_history_quotes(df_scraped, run_iso_ts)
        _append_history_indices(hcpi_result)
    except Exception as e:
        print(f"history append failed: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("FINAL REPORT")
        print("=" * 70 + "\n")
        print(format_hcpi_report(hcpi_result))

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
    if args.cron:
        sys.exit(run_scheduled_cron(database_url=database_url, verbose=verbose))
    elif args.continuous:
        run_continuous(interval_seconds=args.interval, database_url=database_url, verbose=verbose)
    else:
        run_single_scrape_and_calculate(database_url=database_url, verbose=verbose)

if __name__ == "__main__":
    main()


