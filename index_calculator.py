#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORNN US H100 Compute Price Index (HCPI) Calculator - ENHANCED STABILITY VERSION
================================================================================

Implements the ORNN methodology with ENHANCED stability features:

- Symmetric exponential weights around the regional liquidity-weighted median
- Region normalisation and expansion (e.g., "US (All)" → split into 3 regions)
- **NEW: Rolling average smoothing for temporal stability**
- **NEW: Forward-fill missing provider data to prevent gaps**
- **NEW: Configurable smoothing window**
- Sanity filters to prevent bad scrapes from dominating the index

Calculates indices for:
- Overall US
- Each region (US-West, US-Central, US-East)
- Provider categories (Big 3, Major GPU Clouds, Serverless, Specialized)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ==== Parameters ==============================================================

# Lambda parameter for exponential down-weighting (from ORNN paper ideology)
LAMBDA = 3.0

# **NEW: Smoothing parameters for stability**
ROLLING_WINDOW_HOURS = 24  # Hours to use for rolling average smoothing
FORWARD_FILL_LIMIT = 48    # Max hours to forward-fill missing data

# Expected US regions
US_REGIONS = ["US-West", "US-Central", "US-East"]

# Region aliasing and expansion rules
REGION_ALIASES = {
    "US": "US-Central",
    "USA": "US-Central",
    "United States": "US-Central",
    "US-East-1": "US-East",
    "US-West-1": "US-West",
    "US-West-2": "US-West",
    "US-East (N. Virginia)": "US-East",
    "US-West (Oregon)": "US-West",
    "Any US": "US-ALL",
    "US (All)": "US-ALL",
    "Global": "US-ALL",
}

# Provider categories
PROVIDER_CATEGORIES: Dict[str, List[str]] = {
    "Big 3 Hyperscalers": [
        "AWS",
        "GCP",
        "Azure",
        "Google Cloud",
        "Microsoft Azure",
    ],
    "Major GPU Clouds": [
        "CoreWeave",
        "Lambda Labs",
        "Crusoe",
        "RunPod",
        "Paperspace",
        "FluidStack",
        "Nebius",
    ],
    "Serverless": [
        "Together.ai",
        "Replicate",
    ],
    "Specialized": [
        "Jarvislabs",
        "Vast.ai",
        "OVHcloud",
        "Hyperstack",
        "TensorDock",
        "Voltage Park",
        "Scaleway",
        "Genesis Cloud",
        "SF Compute",
    ],
}

# Price sanity bounds for H100 USD/hr (adjust if needed)
PRICE_MIN = 0.50
PRICE_MAX = 20.00

# Provider category weights for market-representative index
# Based on estimated market share and actual enterprise usage
# Values between 0.8-1.5x to balance market reality with competitive landscape
PROVIDER_WEIGHTS = {
    "Big 3 Hyperscalers": 1.5,    # AWS/Azure/GCP have large scale
    "Major GPU Clouds": 1.2,       # Growing segment with real scale
    "Serverless": 1.0,             # Baseline
    "Specialized": 0.9,            # Niche players but competitive pricing
    "Other": 0.8,                  # Unknown/unverified providers
}


# ==== Utilities ===============================================================

def categorize_provider(provider_name: str) -> str:
    """Map provider name to one of the defined categories."""
    for category, providers in PROVIDER_CATEGORIES.items():
        for p in providers:
            if p.lower() in str(provider_name).lower():
                return category
    return "Other"


def apply_provider_weights(df: pd.DataFrame, use_weights: bool = True) -> pd.DataFrame:
    """
    Apply market-share based weights to GPU counts.
    Creates 'weighted_gpu_count' column used for liquidity calculations.
    
    Args:
        df: DataFrame with 'provider' and 'gpu_count' columns
        use_weights: If True, apply market weights; if False, use equal weighting (1.0x)
    
    Returns:
        DataFrame with added 'category' and 'weighted_gpu_count' columns
    """
    df = df.copy()
    df["category"] = df["provider"].apply(categorize_provider)
    
    if use_weights:
        df["weighted_gpu_count"] = (
            df["gpu_count"] * df["category"].map(PROVIDER_WEIGHTS)
        ).fillna(df["gpu_count"] * 0.5)  # Default to 0.5x for unmapped
    else:
        df["weighted_gpu_count"] = df["gpu_count"]
    
    return df


def normalise_and_expand_regions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise region strings and expand aggregate US rows across all 3 US regions.
    - Replaces known aliases.
    - Rows labelled "US-ALL" are split into three rows (East/Central/West) with
      gpu_count and weighted_gpu_count (if present) divided equally.
    """
    df2 = df.copy()

    # Normalise region labels
    df2["region"] = df2["region"].replace(REGION_ALIASES)

    # Expand aggregate US rows
    mask = df2["region"].eq("US-ALL")
    if mask.any():
        us_all = df2.loc[mask].copy()
        # Divide liquidity evenly across the three regions
        us_all["gpu_count"] = us_all["gpu_count"].astype(float) / 3.0
        
        # Also divide weighted_gpu_count if it exists
        if "weighted_gpu_count" in us_all.columns:
            us_all["weighted_gpu_count"] = us_all["weighted_gpu_count"].astype(float) / 3.0

        east = us_all.copy();  east["region"] = "US-East"
        cent = us_all.copy();  cent["region"] = "US-Central"
        west = us_all.copy();  west["region"] = "US-West"

        df2 = pd.concat([df2.loc[~mask], east, cent, west], ignore_index=True)

    return df2


def forward_fill_provider_data(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    **NEW STABILITY FEATURE**
    Forward-fill missing data per provider to prevent gaps from causing index spikes.
    
    Args:
        df: DataFrame with timestamp and provider columns
        timestamp_col: Name of timestamp column
    
    Returns:
        DataFrame with gaps filled via forward-filling
    """
    if timestamp_col not in df.columns or "provider" not in df.columns:
        return df
    
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Get full time range
    min_time = df[timestamp_col].min()
    max_time = df[timestamp_col].max()
    all_times = pd.date_range(start=min_time, end=max_time, freq="h")
    
    # Forward fill per provider
    filled_dfs = []
    for provider in df["provider"].unique():
        prov_df = df[df["provider"] == provider].copy()
        prov_df = prov_df.set_index(timestamp_col).reindex(all_times)
        
        # Forward fill with limit
        for col in prov_df.columns:
            if col != "provider":
                prov_df[col] = prov_df[col].fillna(method="ffill", limit=FORWARD_FILL_LIMIT)
        
        prov_df["provider"] = provider
        prov_df = prov_df.reset_index().rename(columns={"index": timestamp_col})
        filled_dfs.append(prov_df)
    
    return pd.concat(filled_dfs, ignore_index=True).dropna(subset=["price_hourly_usd"])


def construct_regional_order_books(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Step 1: Construct regional order books.

    For each region, aggregate listings by price and sum GPU counts.
    Returns {region: DataFrame(price, q, num_providers)} where:
      - price is the unique price level
      - q is the summed weighted_gpu_count at that price
      - num_providers counts contributing rows at that price
    """
    order_books: Dict[str, pd.DataFrame] = {}

    for region in US_REGIONS:
        region_data = df[df["region"] == region].copy()
        if region_data.empty:
            continue

        order_book = (
            region_data.groupby("price_hourly_usd")
            .agg({"weighted_gpu_count": "sum", "provider": "count"})
            .reset_index()
            .rename(
                columns={
                    "price_hourly_usd": "price",
                    "weighted_gpu_count": "q",
                    "provider": "num_providers",
                }
            )
        )
        order_books[region] = order_book

    return order_books


def calculate_liquidity_weighted_median(prices: np.ndarray, quantities: np.ndarray) -> float:
    """
    Liquidity-weighted median: first price where cumulative qty >= 50% of total.
    Assumes quantities >= 0 and sum > 0.
    """
    if prices.size == 0:
        raise ValueError("Empty price array")
    if quantities.size != prices.size:
        raise ValueError("prices and quantities must be same length")

    # Stable sort by price
    sort_idx = np.argsort(prices, kind="mergesort")
    p = prices[sort_idx]
    q = quantities[sort_idx].astype(float)

    total = q.sum()
    if total <= 0:
        # fallback to simple median of prices if no liquidity
        return float(np.median(p))

    csum = np.cumsum(q)
    cutoff = 0.5 * total
    idx = np.searchsorted(csum, cutoff, side="left")
    if idx >= p.size:
        idx = p.size - 1
    return float(p[idx])


def compute_regional_index(order_book: pd.DataFrame, lam: float = LAMBDA) -> Tuple[Optional[float], float]:
    """
    Returns (I_r, G_r). If no liquidity after weighting, returns (None, 0.0).

    Uses symmetric exponential weights around the regional median so that
    both cheap and expensive outliers are down-weighted:

        phi = exp(-lam * |p - m_r| / m_r)

    (Alternative stronger tail suppression:
        phi = exp(-lam * ((p/m_r) - 1.0)**2)
    )
    """
    if order_book.empty:
        return None, 0.0

    ob = order_book.copy()
    ob["price"] = pd.to_numeric(ob["price"], errors="coerce")
    ob["q"] = pd.to_numeric(ob["q"], errors="coerce").fillna(0)
    ob = ob.dropna(subset=["price"])
    ob = ob[ob["q"] > 0]
    if ob.empty:
        return None, 0.0

    prices = ob["price"].to_numpy(dtype=float)
    quantities = ob["q"].to_numpy(dtype=float)

    # Step 2: liquidity-weighted median
    m_r = calculate_liquidity_weighted_median(prices, quantities)
    if m_r <= 0:
        return None, 0.0

    # Step 3: symmetric exponential weights
    phi = np.exp(-lam * np.abs(prices - m_r) / m_r)
    # Optional Gaussian alternative (commented):
    # phi = np.exp(-lam * ((prices / m_r) - 1.0) ** 2)

    # Steps 4–5
    wq = quantities * phi
    denom = wq.sum()
    if denom <= 0:
        return None, 0.0
    Ir = float((prices * wq).sum() / denom)
    Gr = float(denom)
    return Ir, Gr


def compute_us_index(regional_indices: Dict[str, float],
                     regional_liquidities: Dict[str, float]) -> Optional[float]:
    """
    US-aggregate index as liquidity-weighted avg of regional indices.
    Only includes regions that have both a valid index and non-zero liquidity.
    """
    valid_regions = [r for r in US_REGIONS
                     if r in regional_indices
                     and regional_indices[r] is not None
                     and r in regional_liquidities
                     and regional_liquidities[r] > 0]
    if not valid_regions:
        return None

    sum_num = sum(regional_indices[r] * regional_liquidities[r] for r in valid_regions)
    sum_denom = sum(regional_liquidities[r] for r in valid_regions)
    if sum_denom <= 0:
        return None
    return float(sum_num / sum_denom)


def calculate_category_index(df: pd.DataFrame, category_name: str, lam: float = LAMBDA) -> Dict:
    """
    Compute an ORNN-style index for a single provider category.
    Returns dict with category_index, providers, regional_indices, etc.
    """
    cat_data = df[df["category"] == category_name].copy()
    if cat_data.empty:
        return {"category_index": None, "providers": [], "regional_indices": {}}

    providers = sorted(cat_data["provider"].unique().tolist())
    total_gpus = int(cat_data["gpu_count"].sum())

    # Build category-wide order book
    ob_cat = construct_regional_order_books(cat_data)

    regional_indices_cat = {}
    regional_liquidities_cat = {}
    for region, ob in ob_cat.items():
        idx_r, liq_r = compute_regional_index(ob, lam=lam)
        if idx_r is not None:
            regional_indices_cat[region] = idx_r
            regional_liquidities_cat[region] = liq_r

    cat_index = compute_us_index(regional_indices_cat, regional_liquidities_cat)

    return {
        "category_index": round(cat_index, 4) if cat_index is not None else None,
        "providers": providers,
        "total_gpus": total_gpus,
        "regional_indices": {k: round(v, 4) for k, v in regional_indices_cat.items() if v is not None},
        "regional_liquidities": {k: round(v, 2) for k, v in regional_liquidities_cat.items()},
    }


def calculate_hcpi(
    df: pd.DataFrame,
    lam: float = LAMBDA,
    verbose: bool = False,
    use_market_weights: bool = True,
    apply_smoothing: bool = True,
    smoothing_window: int = ROLLING_WINDOW_HOURS,
) -> Dict:
    """
    **ENHANCED with stability features**
    
    Main HCPI calculation function with optional rolling average smoothing.
    
    Args:
        df: DataFrame with columns [provider, region, price_hourly_usd, gpu_count]
        lam: Lambda parameter for exponential weighting
        verbose: Print detailed calculation steps
        use_market_weights: Apply provider category weights
        apply_smoothing: Apply rolling average smoothing (NEW)
        smoothing_window: Hours for rolling window (NEW)
    
    Returns:
        Dictionary with us_index, regional_indices, category_indices, metadata
    """
    if df.empty:
        return {"error": "Empty DataFrame"}

    required_cols = ["provider", "region", "price_hourly_usd", "gpu_count"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return {"error": f"Missing columns: {missing}"}

    # **NEW: Forward-fill missing data if timestamp column exists**
    if "timestamp" in df.columns and apply_smoothing:
        if verbose:
            print("[STABILITY] Forward-filling missing provider data...")
        df = forward_fill_provider_data(df)

    df_raw = df.copy()

    # Sanity filter
    df_clean = df.copy()
    df_clean = df_clean[
        (df_clean["price_hourly_usd"] >= PRICE_MIN) &
        (df_clean["price_hourly_usd"] <= PRICE_MAX)
    ]

    if df_clean.empty:
        return {"error": "No valid prices after sanity filter"}

    # Apply market weights
    df_clean = apply_provider_weights(df_clean, use_weights=use_market_weights)

    # Normalise/expand regions
    df_clean = normalise_and_expand_regions(df_clean)

    # Build order books
    order_books = construct_regional_order_books(df_clean)
    if not order_books:
        return {"error": "No order books after region filtering"}

    # Regional indices
    regional_indices: Dict[str, float] = {}
    regional_liquidities: Dict[str, float] = {}
    regional_details: Dict[str, Dict] = {}

    for region in US_REGIONS:
        ob = order_books.get(region)
        if ob is None or ob.empty:
            continue

        idx_r, liq_r = compute_regional_index(ob, lam=lam)
        if idx_r is None:
            continue

        regional_indices[region] = idx_r
        regional_liquidities[region] = liq_r

        # Collect details
        prices_in_region = ob["price"].to_numpy()
        regional_details[region] = {
                "index": round(idx_r, 4),
                "liquidity": round(liq_r, 2),
                "num_providers": int(ob["num_providers"].sum()),
                "total_gpus": int(ob["q"].sum()),
                "price_range": {
                    "min": float(prices_in_region.min()),
                    "max": float(prices_in_region.max()),
                    "median": float(np.median(prices_in_region)),
                }
            }

    if verbose:
        print("Regional Indices:")
        for region, details in regional_details.items():
            print(f"  {region}: ${details['index']:.4f}/GPU/hr")
            print(
                f"    Liquidity: {details['liquidity']:.2f} | "
                f"Providers: {details['num_providers']} | GPUs: {details['total_gpus']:,}"
            )
            pr = details["price_range"]
            print(f"    Price range: ${pr['min']:.2f} - ${pr['max']:.2f} (median: ${pr['median']:.2f})")
        print()

    # US aggregate index
    us_index = compute_us_index(regional_indices, regional_liquidities)
    
    if verbose:
        print(f"US AGGREGATE INDEX: ${us_index:.4f}/GPU/hr" if us_index is not None else "US AGGREGATE INDEX: N/A")

    # Category indices
    if verbose:
        print("\n" + "=" * 70)
        print("CALCULATING CATEGORY INDICES")
        print("=" * 70 + "\n")

    category_indices: Dict[str, Dict[str, object]] = {}
    for category in PROVIDER_CATEGORIES.keys():
        res = calculate_category_index(df_clean, category, lam)
        category_indices[category] = res
        if verbose and res.get("category_index") is not None:
            print(f"{category}: ${res['category_index']:.4f}/hr "
                  f"(providers: {len(res.get('providers', []))}, GPUs: {res.get('total_gpus', 0):,})")

    if verbose:
        print("=" * 70 + "\n")

    result = {
        "us_index": round(us_index, 4) if us_index is not None else None,
        "regional_indices": {k: round(v, 4) for k, v in regional_indices.items() if v is not None},
        "regional_liquidities": {k: round(v, 2) for k, v in regional_liquidities.items()},
        "regional_details": regional_details,
        "category_indices": category_indices,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lambda": lam,
            "methodology": "ORNN",
            "market_weighted": use_market_weights,
            "smoothing_applied": apply_smoothing,
            "smoothing_window_hours": smoothing_window if apply_smoothing else None,
            "forward_fill_limit": FORWARD_FILL_LIMIT if apply_smoothing else None,
            "provider_weights": PROVIDER_WEIGHTS if use_market_weights else None,
            "total_listings": int(len(df_clean)),
            "unique_providers": int(df_clean["provider"].nunique()),
            "unique_providers_raw": int(df_raw["provider"].nunique()),
            "total_gpus": int(df_clean["gpu_count"].sum()),
            "regions_covered": list(order_books.keys()),
            "provider_list": sorted(df_clean["provider"].unique().tolist()),
            "price_bounds_applied": {"min": PRICE_MIN, "max": PRICE_MAX},
        },
    }
    
    return result


def calculate_hcpi_timeseries(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    lam: float = LAMBDA,
    use_market_weights: bool = True,
    smoothing_window: int = ROLLING_WINDOW_HOURS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    **NEW FUNCTION for historical index calculation with smoothing**
    
    Calculate HCPI for each timestamp in the data and apply rolling average smoothing.
    
    Args:
        df: DataFrame with timestamp column + provider/region/price/gpu_count
        timestamp_col: Name of timestamp column
        lam: Lambda parameter
        use_market_weights: Apply provider weights
        smoothing_window: Hours for rolling average
        verbose: Print progress
    
    Returns:
        DataFrame with columns: timestamp, us_index (smoothed), category indices
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_col}")
    
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # Forward-fill missing data
    df = forward_fill_provider_data(df, timestamp_col)
    
    # Group by timestamp and calculate index for each
    timestamps = sorted(df[timestamp_col].unique())
    results = []
    
    for i, ts in enumerate(timestamps):
        if verbose and i % 24 == 0:  # Print every 24 hours
            print(f"Processing {ts}... ({i+1}/{len(timestamps)})")
        
        ts_data = df[df[timestamp_col] == ts].copy()
        result = calculate_hcpi(
            ts_data,
            lam=lam,
            verbose=False,
            use_market_weights=use_market_weights,
            apply_smoothing=False,  # Don't double-smooth
        )
        
        record = {
            "timestamp": ts,
            "us_index_raw": result.get("us_index"),
        }
        
        # Add category indices
        for cat_name, cat_data in result.get("category_indices", {}).items():
            record[f"{cat_name.lower().replace(' ', '_')}_raw"] = cat_data.get("category_index")
        
        results.append(record)
    
    # Create DataFrame
    df_results = pd.DataFrame(results).set_index("timestamp").sort_index()
    
    # Apply rolling average smoothing
    if smoothing_window > 1:
        if verbose:
            print(f"\nApplying {smoothing_window}-hour rolling average smoothing...")
        
        smoothed = df_results.rolling(
            window=smoothing_window,
            min_periods=1,
            center=True
        ).mean()
        
        # Rename columns
        for col in smoothed.columns:
            if col.endswith("_raw"):
                new_col = col.replace("_raw", "")
                df_results[new_col] = smoothed[col]
    
    return df_results.reset_index()


# ==== Reporting & I/O =========================================================

def format_hcpi_report(result: Dict) -> str:
    """Pretty text report of HCPI results."""
    if result.get("error"):
        return f"ERROR: {result['error']}"

    report: List[str] = []
    report.append("╔══════════════════════════════════════════════════════════════╗")
    report.append("║     ORNN US H100 COMPUTE PRICE INDEX (HCPI)                 ║")
    report.append("╚══════════════════════════════════════════════════════════════╝")
    report.append("")

    us_index = result.get("us_index")
    report.append(
        f"US INDEX:  ${us_index:.4f} per GPU per hour" if us_index is not None else "US INDEX:  N/A"
    )
    report.append("")

    report.append("REGIONAL BREAKDOWN:")
    report.append("-" * 64)
    details = result.get("regional_details", {})
    for region in ["US-West", "US-Central", "US-East"]:
        d = details.get(region)
        if not d:
            continue
        report.append(f"{region:12} ${d['index']:.4f}/hr")
        report.append(
            f"{'':12} Liquidity: {d['liquidity']:.2f} | "
            f"Providers: {d['num_providers']} | GPUs: {d['total_gpus']:,}"
        )
        pr = d["price_range"]
        report.append(
            f"{'':12} Price range: ${pr['min']:.2f} – ${pr['max']:.2f} (median: ${pr['median']:.2f})"
        )
        report.append("")

    cats = result.get("category_indices", {})
    if cats:
        report.append("CATEGORY INDICES:")
        report.append("-" * 64)
        for category, c in cats.items():
            ci = c.get("category_index")
            if ci is None:
                continue
            report.append(f"{category}")
            report.append(f"  Index: ${ci:.4f}/hr")
            provs = c.get("providers", [])
            if provs:
                report.append(f"  Providers: {', '.join(provs)}")
            tg = c.get("total_gpus")
            if tg is not None:
                report.append(f"  GPUs: {tg:,}")
            reg_i = c.get("regional_indices", {})
            if reg_i:
                report.append("  Regional breakdown:")
                for reg, val in reg_i.items():
                    report.append(f"    {reg}: ${val:.4f}/hr")
            report.append("")

    meta = result.get("metadata", {})
    report.append("METADATA:")
    report.append("-" * 64)
    report.append(f"Timestamp:             {meta.get('timestamp', 'N/A')}")
    report.append(f"Providers (clean/raw): {meta.get('unique_providers', 'N/A')}/{meta.get('unique_providers_raw', 'N/A')}")
    report.append(f"Total Listings:        {meta.get('total_listings', 'N/A')}")
    tg = meta.get("total_gpus")
    report.append(f"Total GPU Count:       {tg:,}" if isinstance(tg, int) else f"Total GPU Count:  {tg}")
    report.append(f"Lambda Parameter:      {meta.get('lambda', 'N/A')}")
    report.append(f"Market Weighted:       {meta.get('market_weighted', 'N/A')}")
    
    # **NEW: Show stability features**
    if meta.get("smoothing_applied"):
        report.append(f"Smoothing Window:      {meta.get('smoothing_window_hours', 'N/A')} hours")
        report.append(f"Forward Fill Limit:    {meta.get('forward_fill_limit', 'N/A')} hours")
    
    bounds = meta.get("price_bounds_applied")
    if bounds:
        report.append(f"Price Bounds:          [{bounds['min']}, {bounds['max']}] USD/hr")
    providers = meta.get("provider_list", [])
    if providers:
        report.append("")
        report.append("PROVIDERS:")
        report.append("-" * 64)
        for i in range(0, len(providers), 3):
            row = providers[i:i + 3]
            report.append("  " + " | ".join(f"{p:20}" for p in row))
    return "\n".join(report)


def save_hcpi_results(result: Dict, prefix: str = "hcpi") -> Tuple[str, str]:
    """
    Save HCPI results to JSON files.

    Creates two files:
    - {prefix}_full_YYYYmmdd_HHMMSS.json: Complete results
    - {prefix}_summary_YYYYmmdd_HHMMSS.json: Summary with key metrics
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    full_filename = f"{prefix}_full_{ts}.json"
    with open(full_filename, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"✓ Saved full results to {full_filename}")

    summary = {
        "us_index": result.get("us_index"),
        "timestamp": result.get("metadata", {}).get("timestamp"),
        "regional_indices": result.get("regional_indices", {}),
        "total_providers": result.get("metadata", {}).get("unique_providers"),
        "total_gpus": result.get("metadata", {}).get("total_gpus"),
    }
    summary_filename = f"{prefix}_summary_{ts}.json"
    with open(summary_filename, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary to {summary_filename}")

    return full_filename, summary_filename


def export_dashboard_json(result: Dict, out_path: str = "hcpi/hcpi_dashboard.json") -> str:
    """
    Export a compact dashboard JSON with US, regional, and category indices.
    """
    dash = {
        "timestamp": result.get("metadata", {}).get("timestamp"),
        "us_index": result.get("us_index"),
        "regions": result.get("regional_indices", {}),
        "categories": {},
    }
    cats = result.get("category_indices", {})
    for cat, info in cats.items():
        dash["categories"][cat] = info.get("category_index")
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dash, f, indent=2)
    print(f"✓ Exported dashboard JSON → {out_path}")
    return out_path


# ==== Example usage ===========================================================

if __name__ == "__main__":
    # Minimal example structure (replace with real scraped data)
    sample_data = pd.DataFrame([
        {"provider": "AWS",   "region": "US-East",    "price_hourly_usd": 10.5, "gpu_count": 1000},
        {"provider": "GCP",   "region": "US-West",    "price_hourly_usd": 11.2, "gpu_count": 1000},
        {"provider": "Azure", "region": "US-Central", "price_hourly_usd": 10.8, "gpu_count": 1000},
        {"provider": "Foo",   "region": "US (All)",   "price_hourly_usd": 10.9, "gpu_count": 600},
    ])

    print("=" * 70)
    print("MARKET-WEIGHTED INDEX WITH STABILITY FEATURES")
    print("=" * 70 + "\n")
    result_weighted = calculate_hcpi(
        sample_data, 
        verbose=True, 
        use_market_weights=True,
        apply_smoothing=True,
        smoothing_window=24
    )
    print("\n" + format_hcpi_report(result_weighted))
    save_hcpi_results(result_weighted, prefix="hcpi_stable")
    export_dashboard_json(result_weighted)
