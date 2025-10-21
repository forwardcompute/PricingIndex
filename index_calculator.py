#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ORNN US H100 Compute Price Index (HCPI) Calculator
==================================================

Implements the ORNN methodology with practical guards:

- Symmetric exponential weights around the regional liquidity-weighted median
- Region normalisation and expansion (e.g., "US (All)" → split into 3 regions)
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


# ==== Utilities ===============================================================

def categorize_provider(provider_name: str) -> str:
    """Map provider name to one of the defined categories."""
    for category, providers in PROVIDER_CATEGORIES.items():
        for p in providers:
            if p.lower() in str(provider_name).lower():
                return category
    return "Other"


def normalise_and_expand_regions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise region strings and expand aggregate US rows across all 3 US regions.
    - Replaces known aliases.
    - Rows labelled "US-ALL" are split into three rows (East/Central/West) with
      gpu_count divided equally.
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

        east = us_all.copy();  east["region"] = "US-East"
        cent = us_all.copy();  cent["region"] = "US-Central"
        west = us_all.copy();  west["region"] = "US-West"

        df2 = pd.concat([df2.loc[~mask], east, cent, west], ignore_index=True)

    return df2


def construct_regional_order_books(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Step 1: Construct regional order books.

    For each region, aggregate listings by price and sum GPU counts.
    Returns {region: DataFrame(price, q, num_providers)} where:
      - price is the unique price level
      - q is the summed gpu_count at that price
      - num_providers counts contributing rows at that price
    """
    order_books: Dict[str, pd.DataFrame] = {}

    for region in US_REGIONS:
        region_data = df[df["region"] == region].copy()
        if region_data.empty:
            continue

        order_book = (
            region_data.groupby("price_hourly_usd")
            .agg({"gpu_count": "sum", "provider": "count"})
            .reset_index()
            .rename(
                columns={
                    "price_hourly_usd": "price",
                    "gpu_count": "q",
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
    Step 6: Aggregate regional indices into US index:
        I_US = Σ(I_r × G_r) / Σ(G_r)
    """
    if not regional_indices:
        return None

    valid = [
        (idx, regional_liquidities[reg])
        for reg, idx in regional_indices.items()
        if idx is not None and regional_liquidities.get(reg, 0.0) > 0.0
    ]
    if not valid:
        return None

    num = sum(idx * liq for idx, liq in valid)
    den = sum(liq for _, liq in valid)
    if den <= 0:
        return None
    return num / den


def calculate_category_index(df: pd.DataFrame, category: str, lam: float = LAMBDA) -> Dict:
    """Calculate HCPI for a specific provider category."""
    df_copy = df.copy()
    df_copy["category"] = df_copy["provider"].apply(categorize_provider)
    df_category = df_copy[df_copy["category"] == category].copy()

    if df_category.empty:
        return {
            "category_index": None,
            "regional_indices": {},
            "regional_liquidities": {},
            "providers": [],
            "total_listings": 0,
            "total_gpus": 0,
        }

    order_books = construct_regional_order_books(df_category)

    regional_indices: Dict[str, Optional[float]] = {}
    regional_liquidities: Dict[str, float] = {}

    for region, order_book in order_books.items():
        index, liquidity = compute_regional_index(order_book, lam)
        regional_indices[region] = index
        regional_liquidities[region] = liquidity

    category_index = compute_us_index(regional_indices, regional_liquidities)

    return {
        "category_index": round(category_index, 4) if category_index is not None else None,
        "regional_indices": {k: round(v, 4) for k, v in regional_indices.items() if v is not None},
        "regional_liquidities": {k: round(v, 2) for k, v in regional_liquidities.items()},
        "providers": sorted(df_category["provider"].unique().tolist()),
        "total_listings": int(len(df_category)),
        "total_gpus": int(df_category["gpu_count"].sum()),
    }


# ==== Main calculator =========================================================

def calculate_hcpi(df: pd.DataFrame, lam: float = LAMBDA, verbose: bool = True) -> Dict:
    """
    Calculate the complete HCPI following ORNN methodology with guards.

    Expects df with columns:
        [provider, region, price_hourly_usd, gpu_count]
    """
    if verbose:
        print("=" * 70)
        print("CALCULATING ORNN US H100 COMPUTE PRICE INDEX (HCPI)")
        print("=" * 70)

    # Keep a copy for raw stats
    df_raw = df.copy()

    # Region normalisation/expansion first
    df = normalise_and_expand_regions(df)

    # Validate & clean
    df_clean = df[df["region"].isin(US_REGIONS)].copy()
    df_clean = df_clean.dropna(subset=["price_hourly_usd", "gpu_count"])
    df_clean["price_hourly_usd"] = pd.to_numeric(df_clean["price_hourly_usd"], errors="coerce")
    df_clean["gpu_count"] = pd.to_numeric(df_clean["gpu_count"], errors="coerce")
    df_clean = df_clean.dropna(subset=["price_hourly_usd", "gpu_count"])

    # Sanity price bounds (USD/H100-hour)
    df_clean = df_clean[(df_clean["price_hourly_usd"] >= PRICE_MIN) & (df_clean["price_hourly_usd"] <= PRICE_MAX)]
    df_clean = df_clean[df_clean["gpu_count"] > 0]

    # Optional: clip extreme tails within each region relative to its median
    clipped_parts = []
    for reg in US_REGIONS:
        r = df_clean["region"].eq(reg)
        part = df_clean[r]
        if not part.empty:
            m = part["price_hourly_usd"].median()
            lo, hi = 0.3 * m, 3.0 * m
            part = part[part["price_hourly_usd"].between(lo, hi)]
            clipped_parts.append(part)
    df_clean = pd.concat(clipped_parts, ignore_index=True) if clipped_parts else df_clean

    if df_clean.empty:
        return {
            "us_index": None,
            "error": "No valid data after cleaning",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    if verbose:
        print(f"Lambda parameter: {lam}")
        print(f"Total listings (raw): {len(df_raw)}; after clean: {len(df_clean)}")
        print(f"Unique providers (raw): {df_raw['provider'].nunique()}; after clean: {df_clean['provider'].nunique()}")
        print(f"Total GPU count (clean): {df_clean['gpu_count'].sum():,.0f}\n")

    # Step 1: Regional order books
    order_books = construct_regional_order_books(df_clean)

    if verbose:
        print("Regional Order Books:")
        for region, ob in order_books.items():
            print(f"  {region}: {len(ob)} unique prices, {ob['q'].sum():,.0f} total GPUs")
        print()

    # Steps 2–5: Regional indices and liquidities
    regional_indices: Dict[str, Optional[float]] = {}
    regional_liquidities: Dict[str, float] = {}
    regional_details: Dict[str, Dict[str, object]] = {}

    for region, order_book in order_books.items():
        index, liquidity = compute_regional_index(order_book, lam)
        regional_indices[region] = index
        regional_liquidities[region] = liquidity

        if index is not None:
            regional_details[region] = {
                "index": round(index, 4),
                "liquidity": round(liquidity, 2),
                "num_prices": int(len(order_book)),
                "total_gpus": int(order_book["q"].sum()),
                "num_providers": int(order_book["num_providers"].sum()),
                "price_range": {
                    "min": round(float(order_book["price"].min()), 2),
                    "max": round(float(order_book["price"].max()), 2),
                    "median": round(
                        calculate_liquidity_weighted_median(
                            order_book["price"].values,
                            order_book["q"].values,
                        ),
                        2,
                    ),
                },
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

    # Step 6: US aggregate index
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

    return {
        "us_index": round(us_index, 4) if us_index is not None else None,
        "regional_indices": {k: round(v, 4) for k, v in regional_indices.items() if v is not None},
        "regional_liquidities": {k: round(v, 2) for k, v in regional_liquidities.items()},
        "regional_details": regional_details,
        "category_indices": category_indices,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lambda": lam,
            "methodology": "ORNN",
            "total_listings": int(len(df_clean)),
            "unique_providers": int(df_clean["provider"].nunique()),
            "unique_providers_raw": int(df_raw["provider"].nunique()),
            "total_gpus": int(df_clean["gpu_count"].sum()),
            "regions_covered": list(order_books.keys()),
            "provider_list": sorted(df_clean["provider"].unique().tolist()),
            "price_bounds_applied": {"min": PRICE_MIN, "max": PRICE_MAX},
        },
    }


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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
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

    result = calculate_hcpi(sample_data, verbose=True)
    print(format_hcpi_report(result))
    save_hcpi_results(result)
    export_dashboard_json(result)
