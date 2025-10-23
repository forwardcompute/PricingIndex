#!/usr/bin/env python3
import json, sys, argparse
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone
from index_calculator import calculate_hcpi  # reuse your existing calc

def load_legacy_rows(src_dir: Path) -> pd.DataFrame:
    # Read all *_history.csv in data/history/
    files = sorted(src_dir.glob("*_history.csv"))
    parts = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df.columns = [c.strip() for c in df.columns]
            parts.append(df)
        except Exception as e:
            print(f"skip {f}: {e}", file=sys.stderr)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)

    # Normalise columns across legacy files
    rename = {
        "price": "price_hourly_usd",
        "price_usd_per_gpu_hr": "price_hourly_usd",
        "effective_price_usd_per_gpu_hr": "price_hourly_usd",
        "fetched_at_utc": "timestamp",
        "quote_timestamp": "timestamp",
        "gpu": "gpu_model",
        "gpu_type": "gpu_model",
    }
    for k,v in rename.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]

    # Required columns populate/clean
    for col in ["provider","region","gpu_model","type","duration","price_hourly_usd","timestamp"]:
        if col not in df.columns: df[col] = None

    # Filter: only H100 + On-Demand
    df = df[df["gpu_model"].astype(str).str.contains("H100", case=False, na=False)]
    df = df[df["type"].astype(str).str.contains("on-?demand", case=False, na=False)]

    # Hour bucket timestamps
    def to_hour(ts):
        try:
            dt = pd.to_datetime(ts, utc=True)
        except Exception:
            return pd.NaT
        return dt.floor("H")
    df["ts_hour"] = df["timestamp"].apply(to_hour)
    df = df.dropna(subset=["ts_hour"])

    # Clean price
    df["price_hourly_usd"] = pd.to_numeric(df["price_hourly_usd"], errors="coerce")
    df = df[df["price_hourly_usd"] > 0]

    # Default region for "Global"/"US"/"US (All)" → expand handled inside index_calculator
    df["region"] = df["region"].fillna("US (All)")
    return df

def compute_index_timeseries(df: pd.DataFrame) -> list[dict]:
    rows = []
    # Group per hour and compute index with existing methodology
    for ts, grp in df.groupby("ts_hour"):
        # Prepare minimal schema expected by calculate_hcpi
        tmp = pd.DataFrame({
            "provider": grp["provider"].fillna("Unknown"),
            "region": grp["region"].fillna("US (All)"),
            "price_hourly_usd": grp["price_hourly_usd"],
            "gpu_count": 1,  # unknown from legacy; equal weight
        })
        res = calculate_hcpi(tmp, verbose=False)
        rows.append({
            "timestamp": ts.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
            "us_index": res.get("us_index"),
            "US-East":  (res.get("regional_indices", {}) or {}).get("US-East"),
            "US-Central": (res.get("regional_indices", {}) or {}).get("US-Central"),
            "US-West":  (res.get("regional_indices", {}) or {}).get("US-West"),
        })
    # Sort chronologically
    rows = sorted([r for r in rows if r["us_index"] is not None], key=lambda r: r["timestamp"])
    return rows

def merge_into_json(new_rows: list[dict], out_path: Path):
    try:
        existing = json.loads(out_path.read_text())
        if not isinstance(existing, list): existing = []
    except Exception:
        existing = []
    merged = {r["timestamp"]: r for r in existing}
    for r in new_rows:
        merged[r["timestamp"]] = r
    final = [merged[k] for k in sorted(merged.keys())]
    out_path.write_text(json.dumps(final, indent=2))
    print(f"Backfilled {len(new_rows)} points. Total now: {len(final)}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/history")
    ap.add_argument("--out", default="hcpi/hcpi_history.json")
    args = ap.parse_args()

    src_dir = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = load_legacy_rows(src_dir)
    if df.empty:
        print("No legacy rows found.")
        return 0

    rows = compute_index_timeseries(df)
    if not rows:
        print("No index rows produced.")
        return 0

    merge_into_json(rows, out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
