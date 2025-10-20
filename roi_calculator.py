"""
ROI and Total Cost Calculator
=============================

This module calculates:
1. Total cost for any job duration and GPU count
2. Basic ROI (value per dollar = 1 / price)
3. Optional utilisation-adjusted ROI (if utilisation column exists)
"""

import pandas as pd

def calculate_total_cost(df: pd.DataFrame, duration_hours: float = 1.0) -> pd.DataFrame:
    """
    Add a total cost column for each listing.
    Formula: total_cost = price_per_gpu_hour * gpu_count * duration_hours
    """
    df = df.copy()
    if 'price_hourly_usd' not in df.columns or 'gpu_count' not in df.columns:
        raise ValueError("DataFrame must contain 'price_hourly_usd' and 'gpu_count' columns")

    df['total_cost_usd'] = df['price_hourly_usd'] * df['gpu_count'] * duration_hours
    df['duration_hours'] = duration_hours
    return df


def calculate_basic_roi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a simple ROI column based purely on price.
    ROI = 1 / price_per_gpu_hour
    """
    df = df.copy()
    if 'price_hourly_usd' not in df.columns:
        raise ValueError("DataFrame must contain 'price_hourly_usd' column")

    df['roi_basic'] = 1 / df['price_hourly_usd']
    return df


def calculate_utilisation_adjusted_roi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional ROI that factors in utilisation rate if available.
    ROI = utilisation / price_per_gpu_hour
    """
    df = df.copy()
    if 'price_hourly_usd' not in df.columns:
        raise ValueError("DataFrame must contain 'price_hourly_usd' column")

    if 'utilisation_rate' not in df.columns:
        print("⚠️  No 'utilisation_rate' column found — skipping adjustment.")
        df['roi_utilisation_adj'] = None
        return df

    df['roi_utilisation_adj'] = df['utilisation_rate'] / df['price_hourly_usd']
    return df


def print_roi_summary(df: pd.DataFrame, top_n: int = 10):
    """
    Print a quick summary table of ROI and cost efficiency.
    """
    print("\nTop Providers by ROI (Value per Dollar):")
    print("=" * 60)
    cols = ['provider', 'region', 'price_hourly_usd', 'roi_basic', 'total_cost_usd']
    sub = df[cols].sort_values('roi_basic', ascending=False).head(top_n)
    for _, row in sub.iterrows():
        print(f"{row['provider']:15} {row['region']:10} "
              f"Price: ${row['price_hourly_usd']:.2f}/hr | "
              f"ROI: {row['roi_basic']:.3f} | "
              f"Total (1h): ${row['total_cost_usd']:.2f}")
    print("=" * 60)
