#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HCPI Master Runner - Scrape → Calculate → Persist → Export Dashboard
====================================================================

This orchestrates the full Forward Compute pipeline:

1. Scrape all GPU providers
2. Calculate the ORNN-based HCPI index
3. Compute ROI and total cost summaries
4. Save raw + derived data and dashboard artifacts
5. Export hcpi/hcpi_dashboard.json for the new UI
6. Persist to database (optional)

Usage:
    python master_runner.py
    python master_runner.py --continuous
    python master_runner.py --interval 900
    python master_runner.py --cron
    python master_runner.py --db sqlite:///hcpi.db
"""

import os
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

# === Pipeline imports ===
from scrapers_all_providers import scrape_all_providers
from index_calculator import (
    calculate_hcpi,
    format_hcpi_report,
    save_hcpi_results,
    export_dashboard_json,  # new addition
)
from database_integration import save_to_database
from roi_calculator import (
    calculate_total_cost,
    calculate_basic_roi,
    calculate_utilisation_adjusted_roi,
    print_roi_summary,
)

# === Directories ===
DERIVED_DIR = Path("data/derived")
HCPI_DIR = Path("hcpi")
PUBLIC_DIR = Path("public")


# ============================================================
# Helpers
# ============================================================

def _normalize_scrape_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names for uniform downstream use.
    """
    rename_map = {
        "gpu": "gpu_model",
        "gpu_type": "gpu_model",
        "price_hourly_usd": "effective_price_usd_per_gpu_hr",
        "price_usd_per_gpu_hr": "effective_price_usd_per_gpu_hr",
    }
    df = df.rename(columns=rename_map).copy()
    required = [
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
    for c in required:
        if c not in df.columns:
            df[c] = None
    return df[required]


def _write_dashboard_artifacts(df_scraped: pd.DataFrame, hcpi_result, ts_tag: str, verbose: bool):
    """
    Write all derived artifacts needed for the dashboard & analytics.
    """
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    HCPI_DIR.mkdir(parents=True, exist_ok=True)

    # Save scrape
    df_norm = _normalize_scrape_columns(df_scraped)
    df_norm.to_csv(DERIVED_DIR / "provider_scores_latest.csv", index=False)

    # Save HCPI outputs
    full_path, summary_path = save_hcpi_results(hcpi_result, prefix=f"{HCPI_DIR}/hcpi_{ts_tag}")
    Path(HCPI_DIR / "hcpi_latest_full.json").write_bytes(Path(full_path).read_bytes())
    Path(HCPI_DIR / "hcpi_latest_summary.json").write_bytes(Path(summary_path).read_bytes())

    # NEW — Export dashboard JSON
    try:
        export_dashboard_json(hcpi_result, out_path=HCPI_DIR / "hcpi_dashboard.json")
    except Exception as e:
        print(f"[WARN] dashboard JSON export failed: {e}")

    # Prepare empty CSV placeholders (ROI + snapshot)
    roi_csv = DERIVED_DIR / "roi_comparison.csv"
    snap_csv = DERIVED_DIR / "compute_calculator_snapshot.csv"

    if not roi_csv.exists():
        pd.DataFrame(
            columns=[
                "gpu_model",
                "gpu_count",
                "duration",
                "best_provider",
                "best_region",
                "total_cost_usd",
                "price_per_gpu_hr",
                "timestamp",
            ]
        ).to_csv(roi_csv, index=False)

    if not snap_csv.exists():
        pd.DataFrame(
            columns=[
                "gpu_model",
                "provider_type",
                "$ / GPU-hr",
                "TOTAL",
                "price_lo_usd",
                "price_md_usd",
                "price_hi_usd",
                "term_hours",
                "gpu_count",
                "region",
                "n_quotes",
                "source",
            ]
        ).to_csv(snap_csv, index=False)

    if verbose:
        print(f"✓ wrote {DERIVED_DIR/'provider_scores_latest.csv'}")
        print(f"✓ wrote {HCPI_DIR/'hcpi_latest_full.json'}")
        print(f"✓ wrote {HCPI_DIR/'hcpi_latest_summary.json'}")
        print(f"✓ wrote {HCPI_DIR/'hcpi_dashboard.json'}")


# ============================================================
# Core runner
# ============================================================

def run_single_scrape_and_calculate(database_url: str, verbose: bool = True):
    """
    Execute a single full HCPI run (scrape → calculate → save → DB).
    """
    start_time = datetime.now(timezone.utc)
    ts_tag = start_time.strftime("%Y%m%d_%H%M%S")

    if verbose:
        print("\n" + "=" * 70)
        print(f"START {start_time.isoformat()}")
        print("=" * 70 + "\n")
        print("STEP 1: Scrape all providers")
        print("-" * 70 + "\n")

    df_scraped = scrape_all_providers()
    if df_scraped is None or df_scraped.empty:
        print("[ERROR] No data scraped — aborting.")
        return None, None

    try:
        df_scraped.to_csv(f"scraped_data_{ts_tag}.csv", index=False)
        if verbose:
            print(f"✓ Saved raw scrape → scraped_data_{ts_tag}.csv")
    except Exception as e:
        print(f"[WARN] Could not save raw scrape: {e}")

    # ROI calculation
    try:
        df_cost = calculate_total_cost(df_scraped, duration_hours=1)
        df_cost["perf_score"] = df_cost.get("perf_score", 1.0)
        df_cost["utilisation"] = df_cost.get("utilisation", 1.0)
        df_roi = calculate_basic_roi(df_cost, perf_col="perf_score")
        df_u_roi = calculate_utilisation_adjusted_roi(
            df_roi, perf_col="perf_score", utilisation_col="utilisation"
        )

        if verbose:
            print("\n" + "=" * 70)
            print("STEP 1B: ROI preview")
            print("-" * 70 + "\n")
            print_roi_summary(df_u_roi.head(10))
    except Exception as e:
        if verbose:
            print(f"[WARN] ROI step skipped: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: Calculate HCPI")
        print("-" * 70 + "\n")

    # HCPI calculation
    hcpi_result = calculate_hcpi(df_scraped, verbose=verbose)

    try:
        _write_dashboard_artifacts(df_scraped, hcpi_result, ts_tag, verbose)
    except Exception as e:
        print(f"[WARN] Artifact write failed: {e}")

    if verbose:
        print("\n" + "=" * 70)
        print("FINAL REPORT")
        print("=" * 70 + "\n")
        print(format_hcpi_report(hcpi_result))

    # Save to DB
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3: Persist to Database")
        print("-" * 70 + "\n")

    try:
        save_to_database(df_scraped, hcpi_result, database_url=database_url)
    except Exception as e:
        print(f"[WARN] DB persistence failed: {e}")

    # Done
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    if verbose:
        print(f"\n✓ DONE in {duration:.1f}s")
        print("=" * 70 + "\n")

    return df_scraped, hcpi_result


# ============================================================
# Continuous / Cron runners
# ============================================================

def run_continuous(interval_seconds: int, database_url: str, verbose: bool = True):
    """
    Run the full scrape + index process on a repeating timer.
    """
    cycle_number = 1
    print(f"\nHCPI Continuous Runner — interval: {interval_seconds}s\n")
    while True:
        try:
            print(f"\n{'='*70}\nCYCLE #{cycle_number}\n{'='*70}\n")
            df_scraped, hcpi_result = run_single_scrape_and_calculate(
                database_url=database_url, verbose=verbose
            )
            if df_scraped is None or hcpi_result is None:
                print("[WARN] Cycle produced no results.")
            cycle_number += 1
            next_run = datetime.now(timezone.utc).timestamp() + interval_seconds
            next_time = datetime.fromtimestamp(next_run, tz=timezone.utc)
            print(f"\nNext cycle at {next_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break
        except Exception as e:
            print(f"[ERROR] Cycle error: {e}")
            time.sleep(interval_seconds)


def run_scheduled_cron(database_url: str, verbose: bool = True) -> int:
    """
    Run once in cron-like scheduled mode.
    """
    try:
        df_scraped, hcpi_result = run_single_scrape_and_calculate(
            database_url=database_url, verbose=verbose
        )
        if df_scraped is None or hcpi_result is None:
            return 1
        return 0
    except Exception as e:
        print(f"[FATAL] Cron job error: {e}")
        return 1


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="HCPI Master Runner")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=3600, help="Interval between runs (s)")
    parser.add_argument("--cron", action="store_true", help="Run once for GitHub cron")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--db", type=str, default=None, help="Database URL")
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

