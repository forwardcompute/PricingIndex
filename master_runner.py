#!/usr/bin/env python3
import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import numpy as np

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
# Smoothing window for history (hours)
SMOOTH_INDEX_WINDOW_HOURS = int(os.getenv("SMOOTH_INDEX_WINDOW_HOURS", "48"))

# Outlier detection parameters
OUTLIER_IQR_MULTIPLIER = float(os.getenv("OUTLIER_IQR_MULTIPLIER", "3.0"))  # 3.0 = aggressive, 1.5 = conservative
OUTLIER_MIN_SAMPLES = int(os.getenv("OUTLIER_MIN_SAMPLES", "5"))  # need at least this many samples to detect outliers
OUTLIER_PRICE_FLOOR = float(os.getenv("OUTLIER_PRICE_FLOOR", "0.10"))  # absolute minimum price threshold
OUTLIER_PRICE_CEILING = float(os.getenv("OUTLIER_PRICE_CEILING", "15.0"))  # absolute maximum price threshold

def _ts_tag(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S")

# ──────────────────────────────────────────────────────────────────────────────
# NEW: Outlier Detection & Removal
def _remove_outliers(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Remove outliers from scraped data using IQR method per provider + region combination.
    This prevents bad data from corrupting the index.
    
    Steps:
    1. Apply absolute floor/ceiling thresholds
    2. Group by provider+region
    3. Calculate IQR for each group
    4. Remove prices outside [Q1 - multiplier*IQR, Q3 + multiplier*IQR]
    5. Log what was removed
    """
    if df.empty or 'effective_price_usd_per_gpu_hr' not in df.columns:
        return df
    
    initial_count = len(df)
    price_col = 'effective_price_usd_per_gpu_hr'
    
    # Step 1: Apply absolute thresholds (catch obviously bad data)
    absolute_mask = (
        (df[price_col] >= OUTLIER_PRICE_FLOOR) & 
        (df[price_col] <= OUTLIER_PRICE_CEILING)
    )
    
    removed_absolute = df[~absolute_mask].copy()
    df = df[absolute_mask].copy()
    
    if verbose and len(removed_absolute) > 0:
        print(f"\n🚫 Removed {len(removed_absolute)} quotes outside absolute bounds [${OUTLIER_PRICE_FLOOR:.2f}, ${OUTLIER_PRICE_CEILING:.2f}]:")
        for _, row in removed_absolute.iterrows():
            print(f"   - {row.get('provider', 'unknown')} / {row.get('region', 'unknown')}: ${row[price_col]:.2f}/hr")
    
    # Step 2: IQR-based outlier detection per provider+region group
    if len(df) < OUTLIER_MIN_SAMPLES:
        if verbose:
            print(f"⚠️  Only {len(df)} samples after absolute filtering - skipping IQR outlier detection")
        return df
    
    # Create grouping columns
    df['_group'] = df['provider'].astype(str) + '|' + df['region'].astype(str)
    
    removed_iqr = []
    kept_indices = []
    
    for group_name, group_df in df.groupby('_group'):
        if len(group_df) < OUTLIER_MIN_SAMPLES:
            # Not enough samples in this group - keep all
            kept_indices.extend(group_df.index.tolist())
            continue
        
        # Calculate IQR
        Q1 = group_df[price_col].quantile(0.25)
        Q3 = group_df[price_col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - (OUTLIER_IQR_MULTIPLIER * IQR)
        upper_bound = Q3 + (OUTLIER_IQR_MULTIPLIER * IQR)
        
        # Identify outliers
        outlier_mask = (group_df[price_col] < lower_bound) | (group_df[price_col] > upper_bound)
        
        outliers = group_df[outlier_mask]
        inliers = group_df[~outlier_mask]
        
        if len(outliers) > 0:
            provider, region = group_name.split('|')
            median_price = inliers[price_col].median() if len(inliers) > 0 else group_df[price_col].median()
            
            for _, row in outliers.iterrows():
                removed_iqr.append({
                    'provider': provider,
                    'region': region,
                    'price': row[price_col],
                    'median': median_price,
                    'deviation': abs(row[price_col] - median_price) / median_price * 100 if median_price > 0 else 0,
                    'bounds': f"[{lower_bound:.2f}, {upper_bound:.2f}]"
                })
        
        kept_indices.extend(inliers.index.tolist())
    
    # Filter to kept indices
    df_clean = df.loc[kept_indices].copy()
    df_clean = df_clean.drop(columns=['_group'])
    
    # Report
    if verbose and len(removed_iqr) > 0:
        print(f"\n🔍 IQR Outlier Detection (multiplier={OUTLIER_IQR_MULTIPLIER}):")
        print(f"   Removed {len(removed_iqr)} statistical outliers:")
        for item in removed_iqr:
            print(f"   - {item['provider']} / {item['region']}: ${item['price']:.2f}/hr "
                  f"(median: ${item['median']:.2f}, deviation: {item['deviation']:.0f}%, bounds: {item['bounds']})")
    
    removed_total = initial_count - len(df_clean)
    if verbose and removed_total > 0:
        removal_pct = (removed_total / initial_count) * 100
        print(f"\n✅ Total outliers removed: {removed_total}/{initial_count} ({removal_pct:.1f}%)")
        print(f"   Remaining clean quotes: {len(df_clean)}")
    
    return df_clean

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
      - data/history/index_history.csv  (long CSV - RAW values)
      - hcpi/hcpi_history.csv           (site CSV - RAW values)
      - hcpi/hcpi_history.json          (rolling JSON with retention and smoothing applied)
    
    Process:
    1. Append new raw value to CSVs
    2. Load existing JSON history
    3. Append new raw value
    4. Apply centered rolling average smoothing to entire dataset
    5. Save smoothed data back to JSON
    
    Returns (smoothed_us_index, timestamp_iso) for the new point.
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

    # (A) long CSV under data/history/ (raw, append-only)
    idx_hist_csv = HISTORY_DIR / "index_history.csv"
    pd.DataFrame([row]).to_csv(idx_hist_csv, mode="a", header=not idx_hist_csv.exists(), index=False)

    # (B) site CSV (raw, append-only)
    hcpi_hist_csv = HCPI_DIR / "hcpi_history.csv"
    pd.DataFrame([row]).to_csv(hcpi_hist_csv, mode="a", header=not hcpi_hist_csv.exists(), index=False)

    # (C) rolling JSON - smooth NEW point only, don't re-smooth history
    hcpi_hist_json = HCPI_DIR / "hcpi_history.json"
    hcpi_hist_csv = HCPI_DIR / "hcpi_history.csv"
    
    # Load RAW values from CSV for smoothing calculation (NOT from JSON which is already smoothed!)
    if hcpi_hist_csv.exists():
        df_raw = pd.read_csv(hcpi_hist_csv)
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], utc=True, errors="coerce")
        df_raw = df_raw.dropna(subset=["timestamp"]).sort_values("timestamp")
        
        # Convert to list of dicts for consistency
        raw_history = df_raw.to_dict('records')
    else:
        raw_history = []
    
    # Add new raw point
    raw_with_new = raw_history + [row]
    
    # Convert to DataFrame for smoothing calculation
    df = pd.DataFrame(raw_with_new)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Apply retention (remove old data)
    if HISTORY_RETENTION_HOURS > 0:
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(hours=HISTORY_RETENTION_HOURS)
        df = df[df["timestamp"] >= cutoff]

    # Convert to numeric for smoothing calculation
    numeric_cols = ["us_index", "US-East", "US-Central", "US-West"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Calculate smoothed values (to get the smoothed value for the NEW point)
    smooth_window = int(os.getenv("SMOOTH_INDEX_WINDOW_HOURS", "48"))
    df_smoothed = df.copy()
    if smooth_window > 0:
        for col in numeric_cols:
            if col in df_smoothed.columns:
                # Use center=False so new point is smoothed ONCE when added
                # and never changes afterward
                df_smoothed[col] = df_smoothed[col].rolling(
                    window=smooth_window,
                    min_periods=1,
                    center=False  # Backward-looking - smooth once and done
                ).mean()

    # Extract ONLY the smoothed value for the NEW point (last row)
    # IMPORTANT: Use the original raw timestamp, not from df_smoothed (which may have duplicates)
    if len(df_smoothed) > 0:
        last_smoothed = df_smoothed.iloc[-1]
        smoothed_row = {"timestamp": row["timestamp"]}  # Use NEW raw timestamp
        for col in numeric_cols:
            val = last_smoothed[col]
            if pd.notna(val):
                smoothed_row[col] = round(float(val), 4)
    else:
        smoothed_row = row

    # Keep existing history unchanged, just append the NEW smoothed point
    # Re-load existing to avoid any modifications
    if hcpi_hist_json.exists():
        existing = json.loads(hcpi_hist_json.read_text())
    else:
        existing = []
    
    # Apply retention to existing history
    if HISTORY_RETENTION_HOURS > 0:
        cutoff = datetime.now(timezone.utc) - pd.Timedelta(hours=HISTORY_RETENTION_HOURS)
        existing = [
            rec for rec in existing 
            if pd.to_datetime(rec["timestamp"], utc=True, errors="coerce") >= cutoff
        ]
    
    # Append ONLY the new smoothed point
    existing.append(smoothed_row)

    # Write back
    hcpi_hist_json.write_text(json.dumps(existing, indent=2))

    # Return last smoothed value and timestamp
    return float(smoothed_row.get("us_index")) if smoothed_row.get("us_index") is not None else None, smoothed_row["timestamp"]

def _sync_latest_to_smoothed(us_index_smoothed: float, ts_iso: str):
    """
    Update latest summary/full with the smoothed US index value.
    If a dashboard JSON exists, update it too.
    """
    latest_summary = HCPI_DIR / "hcpi_latest_summary.json"
    latest_full = HCPI_DIR / "hcpi_latest_full.json"
    dash = HCPI_DIR / "hcpi_dashboard.json"  # optional, if your UI uses this

    def _patch(path: Path):
        if not path.exists():
            return
        try:
            obj = json.loads(path.read_text())
            # Summary file has keys: us_index, timestamp, regional_indices, ...
            if isinstance(obj, dict):
                if "us_index" in obj:
                    obj["us_index"] = round(float(us_index_smoothed), 4) if us_index_smoothed is not None else None
                # normalise timestamp to history's last ts
                obj["timestamp"] = ts_iso
                path.write_text(json.dumps(obj, indent=2))
        except Exception as e:
            print(f"warning: failed to patch {path.name}: {e}")

    if us_index_smoothed is not None:
        _patch(latest_summary)
        _patch(latest_full)
        _patch(dash)

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

    # NEW: Remove outliers before processing
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 1a: outlier detection & removal")
        print("-" * 70 + "\n")
    
    df_clean = _remove_outliers(df_scraped, verbose=verbose)
    
    if df_clean.empty:
        print("❌ All data removed as outliers - aborting")
        return None, None

    # Optional ROI preview (non-blocking)
    try:
        df_cost = calculate_total_cost(df_clean, duration_hours=1)
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

    # ── Index calc (using adapter; calculator unchanged) - NOW USING CLEAN DATA
    if verbose:
        print("\n" + "=" * 70)
        print("STEP 2: calculate HCPI")
        print("-" * 70 + "\n")

    df_idx = _prepare_index_input(df_clean)
    hcpi_result = calculate_hcpi(
        df_idx,
        verbose=verbose
    )

    # ── Write artifacts (raw from calculator; will be synced to smoothed later)
    try:
        _write_latest_artifacts(df_clean, hcpi_result, ts, verbose)
    except Exception as e:
        print(f"artifact write failed: {e}")

    # ── Append histories (providers + index) with smoothing applied
    try:
        run_iso_ts = hcpi_result.get("metadata", {}).get("timestamp") or start_dt.isoformat()
        _append_history_quotes(df_clean, run_iso_ts)
        us_index_smoothed, ts_hist = _append_history_indices(hcpi_result)
        # Sync latest JSONs to match the smoothed data point
        _sync_latest_to_smoothed(us_index_smoothed, ts_hist)
        if verbose and us_index_smoothed is not None:
            print(f"Added new data point (smoothed) - US index: {us_index_smoothed:.4f}")
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
        save_to_database(df_clean, hcpi_result, database_url=database_url)
    except Exception as e:
        print(f"db persistence failed: {e}")

    end_dt = datetime.now(timezone.utc)
    if verbose:
        print(f"\nDONE in {(end_dt - start_dt).total_seconds():.1f}s")
        print("=" * 70 + "\n")

    return df_clean, hcpi_result

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
        print(f"SMOOTH_INDEX_WINDOW_HOURS={SMOOTH_INDEX_WINDOW_HOURS}")
        print(f"OUTLIER_IQR_MULTIPLIER={OUTLIER_IQR_MULTIPLIER}")
        print(f"OUTLIER_PRICE_FLOOR=${OUTLIER_PRICE_FLOOR:.2f}")
        print(f"OUTLIER_PRICE_CEILING=${OUTLIER_PRICE_CEILING:.2f}")

    if args.cron:
        sys.exit(run_scheduled_cron(database_url=database_url, verbose=verbose))
    elif args.continuous:
        run_continuous(interval_seconds=args.interval, database_url=database_url, verbose=verbose)
    else:
        run_single_scrape_and_calculate(database_url=database_url, verbose=verbose)

if __name__ == "__main__":
    main()

