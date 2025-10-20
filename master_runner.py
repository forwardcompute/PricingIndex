#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HCPI Master Runner - Scrape → Calculate → Persist
=================================================

This script:
1) Scrapes all H100 providers
2) Calculates the HCPI index (ORNN methodology)
3) (Optional) Computes Total Cost and basic ROI summaries
4) Saves raw data & HCPI artifacts to disk
5) Persists everything to the database

Usage:
    python master_runner.py
    python master_runner.py --continuous
    python master_runner.py --interval 900
    python master_runner.py --cron
    python master_runner.py --db sqlite:///hcpi.db
    HCPI_DATABASE_URL=postgresql://user:pass@host:5432/hcpi python master_runner.py
"""

import os
import sys
import time
import argparse
from datetime import datetime, timezone

import pandas as pd

from scrapers_all_providers import scrape_all_providers
from index_calculator import calculate_hcpi, format_hcpi_report, save_hcpi_results
from database_integration import save_to_database

# ROI utilities are imported, but only used inside the run function (after we have data)
from roi_calculator import (
    calculate_total_cost,
    calculate_basic_roi,
    calculate_utilisation_adjusted_roi,
    print_roi_summary,
)


def run_single_scrape_and_calculate(database_url: str, verbose: bool = True):
    """
    Run one complete scrape + HCPI calculation + persistence cycle.

    Returns:
        (df_scraped, hcpi_result) or (None, None) on failure
    """
    start_time = datetime.now(timezone.utc)
    ts_tag = start_time.strftime("%Y%m%d_%H%M%S")

    if verbose:
        print("\n" + "=" * 70)
        print(f"STARTING SCRAPE CYCLE - {start_time.isoformat()}")
        print("=" * 70 + "\n")
        print("STEP 1: Scraping all providers...")
        print("-" * 70 + "\n")

    # 1) Scrape
    df_scraped = scrape_all_providers()

    if df_scraped is None or df_scraped.empty:
        print("⚠ WARNING: No data scraped from any provider!")
        return None, None

    # Save raw scrape to CSV for auditing
    raw_csv = f"scraped_data_{ts_tag}.csv"
    try:
        df_scraped.to_csv(raw_csv, index=False)
        if verbose:
            print(f"✓ Saved raw data to {raw_csv}")
    except Exception as e:
        print(f"✗ Could not save raw CSV: {e}")

    # 1b) Optional: Total Cost + ROI (does not affect HCPI)
    # Keep this lightweight and non-failing.
    try:
        # Basic 1-hour total cost per listing
        df_cost = calculate_total_cost(df_scraped, duration_hours=1)

        # Placeholder perf/util if not present; safe defaults keep tests/usage simple
        if "perf_score" not in df_cost.columns:
            df_cost["perf_score"] = 1.0
        if "utilisation" not in df_cost.columns:
            df_cost["utilisation"] = 1.0

        df_roi = calculate_basic_roi(df_cost, perf_col="perf_score")
        df_u_roi = calculate_utilisation_adjusted_roi(
            df_roi, perf_col="perf_score", utilisation_col="utilisation"
        )

        if verbose:
            print("\n" + "=" * 70)
            print("STEP 1b: Total Cost & ROI (informational)")
            print("-" * 70 + "\n")
            print_roi_summary(df_u_roi.head(10))
    except Exception as e:
        # Never fail the main pipeline because of ROI helpers
        if verbose:
            print(f"(ROI step skipped: {e})")

    # 2) Calculate HCPI
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: Calculating HCPI...")
        print("-" * 70 + "\n")

    hcpi_result = calculate_hcpi(df_scraped, verbose=verbose)

    # Save HCPI JSON artifacts
    try:
        full_path, summary_path = save_hcpi_results(hcpi_result, prefix=f"hcpi_{ts_tag}")
        if verbose:
            print(f"✓ Saved HCPI artifacts:\n  - {full_path}\n  - {summary_path}")
    except Exception as e:
        print(f"✗ Could not save HCPI JSON artifacts: {e}")

    # Print terminal report
    if verbose:
        print("\n" + "=" * 70)
        print("FINAL REPORT")
        print("=" * 70 + "\n")
        print(format_hcpi_report(hcpi_result))

    # 3) Persist to DB
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 3: Persisting to database...")
        print("-" * 70 + "\n")

    try:
        save_to_database(df_scraped, hcpi_result, database_url=database_url)
    except Exception as e:
        print(f"✗ Database persistence failed: {e}")

    # Done
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    if verbose:
        print(f"\n✓ Cycle completed in {duration:.1f} seconds")
        print("=" * 70 + "\n")

    return df_scraped, hcpi_result


def run_continuous(interval_seconds: int, database_url: str, verbose: bool = True):
    """
    Run scrape → calculate → persist cycles continuously.
    """
    cycle_number = 1

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  HCPI CONTINUOUS RUNNER                                     ║
║  Every {interval_seconds} seconds ({interval_seconds/3600:.2f} hours)                         ║
╚══════════════════════════════════════════════════════════════╝
""")

    while True:
        try:
            print(f"\n{'='*70}")
            print(f"CYCLE #{cycle_number}")
            print(f"{'='*70}\n")

            df_scraped, hcpi_result = run_single_scrape_and_calculate(
                database_url=database_url,
                verbose=verbose
            )

            if df_scraped is None or hcpi_result is None:
                print("⚠ Cycle produced no results - will retry next interval")

            cycle_number += 1

            next_run = datetime.now(timezone.utc).timestamp() + interval_seconds
            next_run_time = datetime.fromtimestamp(next_run, tz=timezone.utc)
            print(f"\n⏰ Next cycle at {next_run_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            print(f"   Sleeping for {interval_seconds} seconds...")
            time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user. Shutting down...")
            break
        except Exception as e:
            print(f"\n✗ ERROR in cycle: {e}")
            print(f"   Will retry in {interval_seconds} seconds...")
            time.sleep(interval_seconds)


def run_scheduled_cron(database_url: str, verbose: bool = True) -> int:
    """
    Run once - designed to be called by cron.

    Add to crontab:
        0 * * * * cd /path/to/project && /usr/bin/env python master_runner.py --cron
    """
    try:
        df_scraped, hcpi_result = run_single_scrape_and_calculate(
            database_url=database_url,
            verbose=verbose
        )
        if df_scraped is None or hcpi_result is None:
            return 1
        return 0
    except Exception as e:
        print(f"✗ FATAL ERROR: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='HCPI Master Runner - Scrape providers, calculate ORNN index, persist to DB'
    )

    parser.add_argument('--continuous', action='store_true',
                        help='Run continuously at specified interval')

    parser.add_argument('--interval', type=int, default=3600,
                        help='Interval in seconds between runs (default: 3600 = 1 hour)')

    parser.add_argument('--cron', action='store_true',
                        help='Run once for cron scheduling (exit code reflects success)')

    parser.add_argument('--quiet', action='store_true',
                        help='Suppress verbose output')

    parser.add_argument('--db', type=str, default=None,
                        help='Database URL (overrides HCPI_DATABASE_URL env var)')

    args = parser.parse_args()
    verbose = not args.quiet

    # Resolve DB URL
    database_url = args.db or os.getenv('HCPI_DATABASE_URL') or "sqlite:///hcpi.db"
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
