#!/usr/bin/env python3
# Backfill HCPI history from legacy provider CSVs -> hcpi/hcpi_history.json
# Filters: H100 only, On-Demand only

import argparse, json
from pathlib import Path
import pandas as pd

from index_calculator import calculate_hcpi  # your calculator

HCPI_DIR = Path("hcpi")
HISTORY_JSON = HCPI_DIR / "hcpi_history.json"

# Legacy column aliases
COLMAP = {
    "price_hourly_usd": "price_hourly_usd",   # already matches your calc
    "effective_price_usd_per_gpu_hr": "price_hourly_usd",
    "fetched_at_utc": "timestamp",
    "quote_timestamp": "timestamp",
    "timestamp": "timestamp",
}

def load_legacy_folder(src: Path) -> pd.DataFrame:
    files = sorted(src.glob("*_history.csv"))
    frames = []
    for p in files:
        try:
            df = pd.read_csv(p)
            frames.append(df)
        except Exception as e:
            print(f"skip {p.name}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # unify columns
    for k, v in list(COLMAP.items()):
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    need = ["provider","region","gpu_model","type","duration","gpu_count","price_hourly_usd","timestamp"]
    for c in need:
        if c not in df.columns:
            df[c] = None

    # filters: H100 + On-Demand
    df["gpu_model"] = df["gpu_model"].astype(str).str.upper().str.replace(r"\s*SXM\b", "", regex=True)
    df = df[df["gpu_model"] == "H100"]
    df["type"] = df["type"].astype(str)
    df = df[df["type"].str.contains("on", case=False, na=False)]  # keep On-Demand / on demand

    # price/count
    df["price_hourly_usd"] = pd.to_numeric(df["price_hourly_usd"], errors="coerce")
    df["gpu_count"] = pd.to_numeric(df["gpu_count"], errors="coerce").fillna(1).astype(int)
    df = df[(df["price_hourly_usd"].notna()) & (df["price_hourly_usd"] > 0) & (df["gpu_count"] > 0)]

    # timestamps → hourly bins
    ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df = df[ts.notna()].copy()
    df["timestamp"] = ts.dt.floor("H")

    # minimal schema your calc needs
    return df[["timestamp","provider","region","price_hourly_usd","gpu_count"]]

def compute_hourly_index(df_norm: pd.DataFrame) -> list[dict]:
    rows = []
    for ts, chunk in df_norm.groupby("timestamp"):
        # your calc expects provider, region, price_hourly_usd, gpu_count
        res = calculate_hcpi(chunk.rename(columns={
            "price_hourly_usd": "price_hourly_usd",
            "gpu_count": "gpu_count",
            "provider": "provider",
            "region": "region",
        }), verbose=False)
        reg = res.get("regional_indices") or {}
        us = res.get("us_index")
        if us is None:
            continue
        rows.append({
            "timestamp": ts.isoformat(),
            "us_index": float(us),
            "US-East": reg.get("US-East"),
            "US-Central": reg.get("US-Central"),
            "US-West": reg.get("US-West"),
            "_source": "legacy-backfill"
        })
    return sorted(rows, key=lambda r: r["timestamp"])

def merge_history(new_rows: list[dict], keep_last=5000):
    HCPI_DIR.mkdir(parents=True, exist_ok=True)
    try:
        cur = json.loads(HISTORY_JSON.read_text())
        assert isinstance(cur, list)
    except Exception:
        cur = []
    by_ts = {r["timestamp"]: r for r in cur if "timestamp" in r}
    for r in new_rows:
        by_ts.setdefault(r["timestamp"], r)
    merged = sorted(by_ts.values(), key=lambda r: r["timestamp"])[-keep_last:]
    HISTORY_JSON.write_text(json.dumps(merged, indent=2))
    print(f"✅ wrote {len(merged)} points → {HISTORY_JSON}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/history", help="folder with *_history.csv")
    ap.add_argument("--write", action="store_true", help="write hcpi_history.json")
    args = ap.parse_args()

    raw = load_legacy_folder(Path(args.src))
    if raw.empty:
        print("No legacy rows found.")
        return
    norm = normalize(raw)
    if norm.empty:
        print("No valid H100 On-Demand rows after normalization.")
        return
    rows = compute_hourly_index(norm)
    print(f"Generated {len(rows)} hourly index points.")
    if args.write and rows:
        merge_history(rows)

if __name__ == "__main__":
    main()
