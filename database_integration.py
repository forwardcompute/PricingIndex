#!/usr/bin/env python3
"""
HCPI Database Integration
=========================

Stores scraped data and HCPI calculations in SQLite (or PostgreSQL).

Tables:
- scraped_prices: Raw provider prices with timestamps
- hcpi_history: Calculated HCPI values over time
- regional_indices: Regional breakdown over time

Requirements:
    pip install sqlalchemy pandas

For PostgreSQL:
    pip install psycopg2-binary
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime,
    Text, JSON, Index, UniqueConstraint, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

Base = declarative_base()

# ============================================================
# DATABASE MODELS
# ============================================================

class ScrapedPrice(Base):
    """Raw scraped prices from providers."""
    __tablename__ = 'scraped_prices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    provider = Column(String(64), nullable=False, index=True)
    region = Column(String(24), nullable=False, index=True)
    gpu_model = Column(String(32), nullable=False, index=True)

    # New fields matching scraper schema
    product = Column(String(64), nullable=True)
    term = Column(String(24), nullable=False, default="on_demand")
    price_hourly_usd = Column(Float, nullable=False)
    gpu_count = Column(Integer, nullable=False)
    source = Column(String(16), nullable=False)          # 'api', 'web', 'aws_price_list_api', ...
    source_url = Column(Text, nullable=True)
    listing_id = Column(String(32), nullable=True, index=True)

    # Composite indexes for common queries & a safety uniqueness
    __table_args__ = (
        # latest by provider, region
        Index('idx_prices_provider_region_ts', 'provider', 'region', 'timestamp'),
        # latest by region
        Index('idx_prices_region_ts', 'region', 'timestamp'),
        # protect against exact duplicate inserts within a run / repeated write
        UniqueConstraint('timestamp', 'provider', 'region', 'gpu_model', 'term', 'listing_id',
                         name='uq_prices_point_in_time'),
    )


class HCPIHistory(Base):
    """HCPI index values over time."""
    __tablename__ = 'hcpi_history'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, unique=True, index=True)

    us_index = Column(Float, nullable=False)
    lambda_param = Column(Float, nullable=False)
    total_providers = Column(Integer, nullable=False)
    total_listings = Column(Integer, nullable=False)
    total_gpus = Column(Integer, nullable=False)
    api_sources_count = Column(Integer, nullable=False, default=0)  # providers contributing via API
    methodology = Column(String(20), default='ORNN')

    # Store full calculation details as JSON (SQLite supports JSON; falls back to TEXT if needed)
    calculation_details = Column(JSON, nullable=True)


class RegionalIndex(Base):
    """Regional index breakdown over time."""
    __tablename__ = 'regional_indices'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)

    region = Column(String(20), nullable=False, index=True)
    regional_index = Column(Float, nullable=False)
    regional_liquidity = Column(Float, nullable=False)
    num_providers = Column(Integer, nullable=False)
    num_prices = Column(Integer, nullable=False)
    total_gpus = Column(Integer, nullable=False)
    price_min = Column(Float, nullable=False)
    price_max = Column(Float, nullable=False)
    price_median = Column(Float, nullable=False)

    __table_args__ = (
        UniqueConstraint('timestamp', 'region', name='uq_regional_ts_region'),
        Index('idx_region_timestamp', 'region', 'timestamp'),
    )


# ============================================================
# DATABASE CONNECTION
# ============================================================

class HCPIDatabase:
    """Database manager for HCPI data."""

    def __init__(self, database_url: str = "sqlite:///hcpi.db"):
        """
        Initialize database connection.

        Args:
            database_url: SQLAlchemy database URL
                - SQLite: "sqlite:///hcpi.db" or "sqlite:///:memory:"
                - PostgreSQL: "postgresql://user:password@localhost:5432/hcpi"
                - MySQL: "mysql://user:password@localhost:3306/hcpi"
        """
        # Use StaticPool only for in-memory SQLite; file-based uses default pooling
        if database_url.startswith('sqlite'):
            in_memory = database_url.endswith(':memory:') or database_url.endswith('://')
            connect_args = {'check_same_thread': False}
            if in_memory:
                self.engine = create_engine(
                    database_url,
                    connect_args=connect_args,
                    poolclass=StaticPool
                )
            else:
                self.engine = create_engine(
                    database_url,
                    connect_args=connect_args
                )
        else:
            self.engine = create_engine(database_url, pool_pre_ping=True)

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.Session = sessionmaker(bind=self.engine)

    def get_session(self):
        """Get a new database session."""
        return self.Session()

    # ========================================
    # INSERT METHODS
    # ========================================

    def insert_scraped_prices(self, df: pd.DataFrame) -> int:
        """
        Insert scraped prices from DataFrame (schema from scrapers_all_providers.py).

        Expected columns (any extras ignored safely):
        ['timestamp','provider','region','gpu_model','product','term',
         'price_hourly_usd','gpu_count','source','source_url','listing_id']

        Returns:
            Number of rows inserted
        """
        if df is None or df.empty:
            return 0

        # Keep only known columns; fill defaults
        cols = ['timestamp','provider','region','gpu_model','product','term',
                'price_hourly_usd','gpu_count','source','source_url','listing_id']
        for c in cols:
            if c not in df.columns:
                df[c] = None

        # Parse/normalize
        df = df[cols].copy()
        df['gpu_count'] = pd.to_numeric(df['gpu_count'], errors='coerce').fillna(0).astype(int)
        df['price_hourly_usd'] = pd.to_numeric(df['price_hourly_usd'], errors='coerce')
        df = df.dropna(subset=['provider','region','gpu_model','price_hourly_usd'])
        # de-dup exact duplicate rows within this run (same unique key)
        df = (df.drop_duplicates(subset=['timestamp','provider','region','gpu_model','term','listing_id'])
                .reset_index(drop=True))

        session = self.get_session()
        try:
            records = []
            for _, row in df.iterrows():
                ts = row['timestamp']
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))

                records.append(ScrapedPrice(
                    timestamp=ts,
                    provider=str(row['provider']),
                    region=str(row['region']),
                    gpu_model=str(row['gpu_model']),
                    product=(None if pd.isna(row['product']) else str(row['product'])),
                    term=(str(row['term']) if row['term'] else "on_demand"),
                    price_hourly_usd=float(row['price_hourly_usd']),
                    gpu_count=int(row['gpu_count']),
                    source=str(row['source'] or 'web'),
                    source_url=(None if pd.isna(row['source_url']) else str(row['source_url'])),
                    listing_id=(None if pd.isna(row['listing_id']) else str(row['listing_id'])[:32]),
                ))

            # Use bulk insert; uniqueness constraint will skip exact dupes by raising.
            # We swallow duplicate errors by row to keep idempotency.
            inserted = 0
            for rec in records:
                try:
                    session.add(rec)
                    session.commit()
                    inserted += 1
                except Exception:
                    session.rollback()  # ignore duplicates per uq_prices_point_in_time
            return inserted
        finally:
            session.close()

    def insert_hcpi_result(self, result: Dict, api_sources_count: Optional[int] = None) -> int:
        """
        Insert HCPI calculation result.

        Args:
            result: Dict from calculate_hcpi()
            api_sources_count: optional override; if None, falls back to 0

        Returns:
            ID of inserted record
        """
        session = self.get_session()
        try:
            ts_str = result['metadata']['timestamp']
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))

            hcpi_record = HCPIHistory(
                timestamp=ts,
                us_index=float(result['us_index']) if result.get('us_index') is not None else None,
                lambda_param=float(result['metadata'].get('lambda', 3.0)),
                total_providers=int(result['metadata']['unique_providers']),
                total_listings=int(result['metadata']['total_listings']),
                total_gpus=int(result['metadata']['total_gpus']),
                api_sources_count=int(api_sources_count or 0),
                methodology=str(result['metadata'].get('methodology', 'ORNN')),
                calculation_details=result
            )
            session.add(hcpi_record)

            # Insert regional indices (if present)
            for region, details in (result.get('regional_details') or {}).items():
                regional_record = RegionalIndex(
                    timestamp=ts,
                    region=str(region),
                    regional_index=float(details['index']),
                    regional_liquidity=float(details['liquidity']),
                    num_providers=int(details['num_providers']),
                    num_prices=int(details['num_prices']),
                    total_gpus=int(details['total_gpus']),
                    price_min=float(details['price_range']['min']),
                    price_max=float(details['price_range']['max']),
                    price_median=float(details['price_range']['median']),
                )
                session.add(regional_record)

            session.commit()
            return hcpi_record.id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ========================================
    # QUERY METHODS
    # ========================================

    def get_latest_hcpi(self) -> Optional[Dict]:
        """Get the most recent HCPI value."""
        session = self.get_session()
        try:
            record = (session.query(HCPIHistory)
                      .order_by(HCPIHistory.timestamp.desc())
                      .first())
            if not record:
                return None
            return {
                'timestamp': record.timestamp.isoformat(),
                'us_index': record.us_index,
                'total_providers': record.total_providers,
                'total_gpus': record.total_gpus
            }
        finally:
            session.close()

    def get_hcpi_history(self, hours: int = 24) -> pd.DataFrame:
        """Get HCPI history for the last N hours."""
        session = self.get_session()
        try:
            cutoff = datetime.now(timezone.utc) - pd.Timedelta(hours=hours)
            records = (session.query(HCPIHistory)
                       .filter(HCPIHistory.timestamp >= cutoff)
                       .order_by(HCPIHistory.timestamp.asc())
                       .all())
            if not records:
                return pd.DataFrame()
            return pd.DataFrame([{
                'timestamp': r.timestamp,
                'us_index': r.us_index,
                'total_providers': r.total_providers,
                'total_gpus': r.total_gpus
            } for r in records])
        finally:
            session.close()

    def get_regional_history(self, region: str, hours: int = 24) -> pd.DataFrame:
        """Get regional index history."""
        session = self.get_session()
        try:
            cutoff = datetime.now(timezone.utc) - pd.Timedelta(hours=hours)
            records = (session.query(RegionalIndex)
                       .filter(RegionalIndex.region == region)
                       .filter(RegionalIndex.timestamp >= cutoff)
                       .order_by(RegionalIndex.timestamp.asc())
                       .all())
            if not records:
                return pd.DataFrame()
            return pd.DataFrame([{
                'timestamp': r.timestamp,
                'regional_index': r.regional_index,
                'regional_liquidity': r.regional_liquidity,
                'num_providers': r.num_providers,
                'total_gpus': r.total_gpus,
                'price_min': r.price_min,
                'price_max': r.price_max,
                'price_median': r.price_median
            } for r in records])
        finally:
            session.close()

    def get_provider_prices(self, provider: str, hours: int = 24) -> pd.DataFrame:
        """Get price history for a specific provider."""
        session = self.get_session()
        try:
            cutoff = datetime.now(timezone.utc) - pd.Timedelta(hours=hours)
            records = (session.query(ScrapedPrice)
                       .filter(ScrapedPrice.provider == provider)
                       .filter(ScrapedPrice.timestamp >= cutoff)
                       .order_by(ScrapedPrice.timestamp.asc())
                       .all())
            if not records:
                return pd.DataFrame()
            return pd.DataFrame([{
                'timestamp': r.timestamp,
                'region': r.region,
                'price_hourly_usd': r.price_hourly_usd,
                'gpu_count': r.gpu_count,
                'term': r.term,
                'source': r.source
            } for r in records])
        finally:
            session.close()

    def get_all_providers_latest(self) -> pd.DataFrame:
        """Get latest price row for each provider (latest timestamp overall per provider)."""
        session = self.get_session()
        try:
            subq = (session.query(
                        ScrapedPrice.provider,
                        func.max(ScrapedPrice.timestamp).label('max_ts')
                    )
                    .group_by(ScrapedPrice.provider)
                    .subquery())

            records = (session.query(ScrapedPrice)
                       .join(subq,
                             (ScrapedPrice.provider == subq.c.provider) &
                             (ScrapedPrice.timestamp == subq.c.max_ts))
                       .all())
            if not records:
                return pd.DataFrame()
            return pd.DataFrame([{
                'provider': r.provider,
                'region': r.region,
                'price_hourly_usd': r.price_hourly_usd,
                'gpu_count': r.gpu_count,
                'term': r.term,
                'timestamp': r.timestamp
            } for r in records])
        finally:
            session.close()

    def get_statistics(self) -> Dict:
        """Get database statistics."""
        session = self.get_session()
        try:
            total_scrapes = session.query(func.count(ScrapedPrice.id)).scalar()
            total_calculations = session.query(func.count(HCPIHistory.id)).scalar()
            unique_providers = session.query(func.count(func.distinct(ScrapedPrice.provider))).scalar()
            first_scrape = session.query(func.min(ScrapedPrice.timestamp)).scalar()
            last_scrape = session.query(func.max(ScrapedPrice.timestamp)).scalar()
            return {
                'total_price_records': int(total_scrapes or 0),
                'total_hcpi_calculations': int(total_calculations or 0),
                'unique_providers': int(unique_providers or 0),
                'first_scrape': first_scrape.isoformat() if first_scrape else None,
                'last_scrape': last_scrape.isoformat() if last_scrape else None,
                'database_url': str(self.engine.url)
            }
        finally:
            session.close()


# ============================================================
# INTEGRATION WITH MASTER RUNNER
# ============================================================

def save_to_database(df_scraped: pd.DataFrame, hcpi_result: Dict,
                     database_url: str = "sqlite:///hcpi.db"):
    """
    Save scraped data and HCPI results to database.

    Args:
        df_scraped: DataFrame from scrape_all_providers()
        hcpi_result: Dict from calculate_hcpi()
        database_url: Database connection string
    """
    db = HCPIDatabase(database_url)

    # Insert scraped prices (idempotent per unique key in a run)
    inserted = 0
    if df_scraped is not None and not df_scraped.empty:
        inserted = db.insert_scraped_prices(df_scraped)
        print(f"✓ Inserted {inserted} price records into database")

    # Derive API providers count from this scrape (unique providers with source == 'api')
    api_sources_count = 0
    if df_scraped is not None and not df_scraped.empty and 'source' in df_scraped.columns:
        api_sources_count = df_scraped[df_scraped['source'] == 'api']['provider'].nunique()

    # Insert HCPI calculation
    if hcpi_result.get('us_index') is not None:
        hcpi_id = db.insert_hcpi_result(hcpi_result, api_sources_count=api_sources_count)
        print(f"✓ Inserted HCPI calculation (ID: {hcpi_id}) into database")

    # Print stats
    stats = db.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Total price records: {stats['total_price_records']:,}")
    print(f"  Total HCPI calculations: {stats['total_hcpi_calculations']}")
    print(f"  Unique providers: {stats['unique_providers']}")
    print(f"  Database: {stats['database_url']}")


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    # Initialize database
    db = HCPIDatabase("sqlite:///hcpi.db")

    # Example: Get latest HCPI
    latest = db.get_latest_hcpi()
    if latest:
        print(f"Latest HCPI: ${latest['us_index']:.4f} at {latest['timestamp']}")

    # Example: Get 24-hour history
    history = db.get_hcpi_history(hours=24)
    if not history.empty:
        print(f"\n24-hour HCPI history: {len(history)} records")
        print(history.tail())

    # Example: Get all providers' latest prices
    providers = db.get_all_providers_latest()
    if not providers.empty:
        print(f"\nLatest prices from {len(providers)} providers:")
        print(providers.sort_values('price_hourly_usd'))
    
    # Example: Get statistics
    stats = db.get_statistics()
    print("\nDatabase Statistics:")
    print(json.dumps(stats, indent=2, default=str))
