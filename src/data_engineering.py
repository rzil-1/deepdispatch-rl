"""
Phase 0: Data Engineering & Feature Extraction
===============================================
This module handles:
  - Loading and cleaning the raw CSV dataset
  - Converting timestamps from UTC to IST
  - Geohash encoding of lat/lon coordinates (6-char precision)
  - Temporal binning into 1-hour slots
  - Vehicle-type congestion weight mapping
  - Violation severity scoring from offence codes
  - Computing the Congestion Impact Score (CIS) per violation
  - Building a spatial adjacency matrix between Geohash cells
  - Filtering to the top-N police stations for prototype scope

WHY these decisions:
  - 6-char Geohash ≈ 1.2km × 0.6km cells: good balance for patrol routing
  - 1-hour slots: statistically robust (enough events per cell-hour)
  - CIS = vehicle_weight × severity × road_factor: a practical proxy for the
    M/M/c queueing model described in the research doc (we lack real traffic
    flow data for the full model)
"""

import pandas as pd
import numpy as np
import geohash2 as geohash
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Constants
# =============================================================================

# Vehicle congestion weights — derived from physical footprint and lane-blocking
# potential. A TANKER blocking an arterial lane causes catastrophically more
# delay than a SCOOTER tucked against the curb.
# Reference: Research doc Section "Vehicle Typology and Congestion Scaling"
VEHICLE_CONGESTION_WEIGHTS = {
    'SCOOTER':             0.3,
    'MOTOR CYCLE':         0.3,
    'MOPED':               0.3,
    'CAR':                 1.0,
    'JEEP':                1.0,
    'VAN':                 1.2,
    'PASSENGER AUTO':      0.5,
    'GOODS AUTO':          1.5,
    'MAXI-CAB':            2.0,
    'LGV':                 2.5,
    'LORRY/GOODS VEHICLE': 2.5,
    'TEMPO':               2.0,
    'HGV':                 3.5,
    'TANKER':              3.0,
    'MINI LORRY':          2.0,
    'PRIVATE BUS':         2.5,
    'BUS (BMTC/KSRTC)':    2.5,
    'TOURIST BUS':         2.5,
    'SCHOOL VEHICLE':      2.0,
    'FACTORY BUS':         2.5,
    'TRACTOR':             2.0,
    'OTHERS':              1.0,
}

# Violation severity weights — "WRONG PARKING" on a main road is more
# dangerous than simple "NO PARKING" in a side lane. Double parking and
# blocking bus stops / schools create acute safety + congestion hazards.
VIOLATION_SEVERITY = {
    'WRONG PARKING':                               1.0,
    'NO PARKING':                                  0.8,
    'PARKING IN A MAIN ROAD':                      1.5,
    'PARKING ON FOOTPATH':                          0.6,
    'PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC':     1.8,
    'DOUBLE PARKING':                              2.0,
    'PARKING OPPOSITE TO ANOTHER PARKED VEHICLE':  1.3,
    'PARKING NEAR ROAD CROSSING':                  1.6,
    'DEFECTIVE NUMBER PLATE':                      0.2,  # not parking-related
}

# Top police stations for prototype scope
TOP_STATIONS = [
    'Upparpet', 'Shivajinagar', 'Malleshwaram', 'HAL Old Airport',
    'City Market', 'Vijayanagara', 'Rajajinagar', 'Kodigehalli',
    'Magadi Road', 'Jeevanbheemanagar'
]

GEOHASH_PRECISION = 6  # ~1.2km × 0.6km cells


# =============================================================================
# Core Functions
# =============================================================================

def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    Load the raw CSV and perform initial cleaning.
    
    Key decisions:
    - Drop 'description' column (100% NULL)
    - Drop 'closed_datetime' column (100% NULL)  
    - Drop 'action_taken_timestamp' column (100% NULL)
    - Convert created_datetime to proper datetime with IST timezone
    """
    print("[Phase 0.1] Loading raw data...")
    df = pd.read_csv(csv_path, low_memory=False)
    
    initial_rows = len(df)
    print(f"  Loaded {initial_rows:,} rows, {len(df.columns)} columns")
    
    # Drop entirely null columns — these are useless for our prototype
    cols_to_drop = ['description', 'closed_datetime', 'action_taken_timestamp']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Parse timestamps
    df['created_datetime'] = pd.to_datetime(df['created_datetime'], errors='coerce', utc=True)
    df['modified_datetime'] = pd.to_datetime(df['modified_datetime'], errors='coerce', utc=True)
    df['validation_timestamp'] = pd.to_datetime(df['validation_timestamp'], errors='coerce', utc=True)
    
    # Convert UTC → IST (+5:30) — the data is from Bengaluru
    df['created_ist'] = df['created_datetime'].dt.tz_convert('Asia/Kolkata')
    
    # Drop rows with null lat/lon (critical for geospatial analysis)
    df = df.dropna(subset=['latitude', 'longitude', 'created_datetime'])
    
    # Drop rows with null police_station (only 5 rows)
    df = df.dropna(subset=['police_station'])
    
    print(f"  After cleaning: {len(df):,} rows ({initial_rows - len(df)} dropped)")
    return df


def filter_top_stations(df: pd.DataFrame, stations: list = None) -> pd.DataFrame:
    """
    Filter to top-N police stations for prototype scope.
    
    WHY: Training on all 54 stations creates a huge state space with very
    sparse data in many cells. Top-10 stations cover ~65% of all violations
    and include the most congestion-critical areas of Bengaluru.
    """
    if stations is None:
        stations = TOP_STATIONS
    
    print(f"[Phase 0.2] Filtering to {len(stations)} stations...")
    df_filtered = df[df['police_station'].isin(stations)].copy()
    pct = len(df_filtered) / len(df) * 100
    print(f"  Kept {len(df_filtered):,} rows ({pct:.1f}% of total)")
    return df_filtered


def encode_geohash(df: pd.DataFrame, precision: int = GEOHASH_PRECISION) -> pd.DataFrame:
    """
    Encode lat/lon into Geohash cells.
    
    WHY Geohash over raw coordinates:
    - Raw lat/lon is continuous and infinitely sparse — useless as RL state
    - Geohash discretizes space into uniform rectangular cells
    - 6-char precision ≈ 1.2km × 0.6km — good for patrol routing
    - Adjacent cells share prefixes → natural adjacency structure
    """
    print(f"[Phase 0.3] Encoding Geohash (precision={precision})...")
    df['geohash'] = df.apply(
        lambda row: geohash.encode(row['latitude'], row['longitude'], precision),
        axis=1
    )
    n_cells = df['geohash'].nunique()
    print(f"  Created {n_cells} unique Geohash cells")
    return df


def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features for the RL environment.
    
    WHY 1-hour bins:
    - 15-minute bins would give 96 slots/day but most cells would have 0 events
    - 1-hour bins give 24 slots/day with reasonable event density
    - Day-of-week captures weekly patterns (weekday rush vs. weekend markets)
    """
    print("[Phase 0.4] Extracting temporal features...")
    df['hour'] = df['created_ist'].dt.hour
    df['day_of_week'] = df['created_ist'].dt.dayofweek  # 0=Monday
    df['date'] = df['created_ist'].dt.date
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Month for seasonal patterns
    df['month'] = df['created_ist'].dt.month
    
    print(f"  Hours range: {df['hour'].min()} to {df['hour'].max()}")
    print(f"  Days covered: {df['date'].nunique()}")
    return df


def compute_vehicle_weight(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map vehicle types to congestion impact weights.
    
    WHY these weights:
    - A double-parked TANKER on Outer Ring Road reduces effective lane capacity
      by ~50% (research doc cites 25.89% additional reduction for double parking)
    - A SCOOTER at the curb barely affects flow
    - Weights are relative to CAR=1.0 baseline
    
    NOTE: We use 'updated_vehicle_type' when available (it's the validated type),
    falling back to original 'vehicle_type'.
    """
    print("[Phase 0.5] Computing vehicle congestion weights...")
    # Use validated vehicle type where available
    df['effective_vehicle_type'] = df['updated_vehicle_type'].fillna(df['vehicle_type'])
    df['vehicle_weight'] = df['effective_vehicle_type'].map(VEHICLE_CONGESTION_WEIGHTS).fillna(1.0)
    
    print(f"  Weight distribution:")
    print(f"    Mean: {df['vehicle_weight'].mean():.2f}")
    print(f"    Min:  {df['vehicle_weight'].min():.1f} (two-wheelers)")
    print(f"    Max:  {df['vehicle_weight'].max():.1f} (HGV/TANKER)")
    return df


def parse_violation_types(violation_str: str) -> list:
    """Parse the JSON-like violation_type string into a list of violation names."""
    try:
        # The format is like: ["WRONG PARKING","NO PARKING"]
        return json.loads(violation_str)
    except (json.JSONDecodeError, TypeError):
        return []


def compute_violation_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute violation severity from the combination of offence types.
    
    WHY sum, not max:
    - A vehicle committing BOTH "PARKING IN A MAIN ROAD" AND "DOUBLE PARKING"
      is categorically more dangerous than either alone
    - Summing captures compound severity
    - We then normalize to keep scores manageable
    """
    print("[Phase 0.6] Computing violation severity scores...")
    
    def severity_score(violation_str):
        violations = parse_violation_types(violation_str)
        if not violations:
            return 1.0  # default
        total = sum(VIOLATION_SEVERITY.get(v.strip(), 0.5) for v in violations)
        return total
    
    df['violation_severity'] = df['violation_type'].apply(severity_score)
    
    print(f"  Severity distribution:")
    print(f"    Mean: {df['violation_severity'].mean():.2f}")
    print(f"    Max:  {df['violation_severity'].max():.2f}")
    return df


def compute_road_factor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate a road importance factor.
    
    WHY:
    - Violations at junctions cause more congestion (blocking intersection throughput)
    - Main roads carry more traffic than residential lanes
    - We use junction_name as a proxy for road importance
    
    Logic:
    - Named junction → 1.5× (these are major intersections BTP monitors)
    - "No Junction" → 1.0× (could be main road or side lane)
    - Known arterial keywords → 2.0× (Outer Ring Road, MG Road, etc.)
    """
    print("[Phase 0.7] Computing road importance factor...")
    
    arterial_keywords = [
        'outer ring', 'ring road', 'mg road', 'mahatma gandhi',
        'hosur road', 'mysore road', 'tumkur road', 'bellary road',
        'old madras road', 'airport road', 'nh ', 'flyover'
    ]
    
    def road_factor(row):
        factor = 1.0
        
        # Junction presence boosts impact
        junction = str(row.get('junction_name', 'No Junction')).lower()
        if junction != 'no junction' and junction != 'nan':
            factor = 1.5
        
        # Arterial road detection from location name
        location = str(row.get('location', '')).lower()
        for keyword in arterial_keywords:
            if keyword in location:
                factor = max(factor, 2.0)
                break
        
        return factor
    
    df['road_factor'] = df.apply(road_factor, axis=1)
    print(f"  Road factor distribution: {df['road_factor'].value_counts().to_dict()}")
    return df


def compute_congestion_impact_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Congestion Impact Score (CIS) per violation.
    
    CIS = vehicle_weight × violation_severity × road_factor
    
    WHY this formula:
    - It's a practical proxy for the full M/M/c queueing delay integral
      described in the research doc (Section "Capacity Reduction")
    - We lack real traffic flow data (arrival rate λ, service rate μ), so
      we approximate the "additional delay" using these heuristic factors
    - The product form captures the intuition: a TANKER (3.0) double-parked (2.0)
      on an arterial (2.0) = CIS of 12.0, vs. a SCOOTER (0.3) wrong-parked (1.0)
      on a side lane (1.0) = CIS of 0.3 — a 40× difference in priority
    
    This is the KEY INNOVATION for the hackathon: quantifying congestion impact,
    not just counting violations.
    """
    print("[Phase 0.8] Computing Congestion Impact Score (CIS)...")
    df['cis'] = df['vehicle_weight'] * df['violation_severity'] * df['road_factor']
    
    print(f"  CIS distribution:")
    print(f"    Mean:   {df['cis'].mean():.2f}")
    print(f"    Median: {df['cis'].median():.2f}")
    print(f"    P95:    {df['cis'].quantile(0.95):.2f}")
    print(f"    Max:    {df['cis'].max():.2f}")
    return df


def compute_survival_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate violation 'lifespan' using validation_timestamp - created_datetime.
    
    WHY this proxy:
    - closed_datetime is 100% NULL in our dataset
    - validation_timestamp is when the ticket was reviewed/approved
    - This gives us an upper bound on how long the violation was 'active'
    - We'll use this to set stochastic lifespans in the RL environment
    
    LIMITATION: This is NOT the true violation duration (vehicle may have left
    long before validation). We treat it as a noisy upper bound.
    """
    print("[Phase 0.9] Computing survival proxy from validation timestamps...")
    
    mask = df['validation_timestamp'].notna() & df['created_datetime'].notna()
    df.loc[mask, 'response_hours'] = (
        df.loc[mask, 'validation_timestamp'] - df.loc[mask, 'created_datetime']
    ).dt.total_seconds() / 3600
    
    # Clamp to reasonable range (0 to 168 hours = 1 week)
    df['response_hours'] = df['response_hours'].clip(0, 168)
    
    valid = df['response_hours'].dropna()
    if len(valid) > 0:
        print(f"  Valid observations: {len(valid):,}")
        print(f"  Mean response time: {valid.mean():.1f} hours")
        print(f"  Median: {valid.median():.1f} hours")
        print(f"  P25: {valid.quantile(0.25):.1f}h, P75: {valid.quantile(0.75):.1f}h")
    
    return df


def build_zone_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate violation data by (geohash, hour, day_of_week) to create
    the zone-level statistics that feed the RL environment.
    
    This is the "predictive heatmap" from the research doc — simplified
    from a full STGNN to historical frequency estimation.
    
    Output columns per zone-time slot:
    - violation_count: number of violations (raw demand)
    - mean_cis: average congestion impact score
    - total_cis: sum of CIS (prioritization metric)
    - dominant_vehicle: most common vehicle type
    - p_violation: empirical probability (count / days observed)
    """
    print("[Phase 0.10] Building zone-level statistics...")
    
    # Count days observed for probability estimation
    total_days = df['date'].nunique()
    
    zone_stats = df.groupby(['geohash', 'hour', 'day_of_week']).agg(
        violation_count=('id', 'count'),
        mean_cis=('cis', 'mean'),
        total_cis=('cis', 'sum'),
        max_cis=('cis', 'max'),
        mean_vehicle_weight=('vehicle_weight', 'mean'),
        lat=('latitude', 'mean'),
        lon=('longitude', 'mean'),
        dominant_vehicle=('effective_vehicle_type', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'UNKNOWN'),
        has_junction=('road_factor', lambda x: (x > 1.0).any()),
    ).reset_index()
    
    # Compute empirical violation probability per zone-hour-dow
    # P = count / (number of weeks observed × 1 slot)
    weeks_observed = max(1, total_days // 7)
    zone_stats['p_violation'] = (zone_stats['violation_count'] / weeks_observed).clip(0, 1)
    
    print(f"  Total zone-time slots: {len(zone_stats):,}")
    print(f"  Unique zones (geohash cells): {zone_stats['geohash'].nunique()}")
    print(f"  Days in dataset: {total_days}, Weeks: {weeks_observed}")
    
    return zone_stats


def _geohash_neighbors(cell: str) -> list:
    """
    Compute the 8 neighbors of a Geohash cell manually.
    
    WHY manual computation:
    - geohash2 library doesn't have a neighbors() function
    - We decode to lat/lon center, apply small offsets in 8 directions,
      and re-encode to find neighbor cell hashes
    - The offset is slightly larger than half the cell size to ensure
      we land in the neighboring cell
    """
    lat, lon = geohash.decode(cell)
    lat, lon = float(lat), float(lon)
    precision = len(cell)
    
    # Approximate cell dimensions at this precision
    # 6-char geohash: ~1.2km lat × 0.6km lon → ~0.011° lat, ~0.005° lon
    dlat = 0.012  # slightly more than half cell height
    dlon = 0.006  # slightly more than half cell width
    
    # 8 directions: N, NE, E, SE, S, SW, W, NW
    offsets = [
        (dlat, 0), (dlat, dlon), (0, dlon), (-dlat, dlon),
        (-dlat, 0), (-dlat, -dlon), (0, -dlon), (dlat, -dlon)
    ]
    
    neighbors = []
    for olat, olon in offsets:
        nbr = geohash.encode(lat + olat, lon + olon, precision)
        if nbr != cell and nbr not in neighbors:
            neighbors.append(nbr)
    
    return neighbors


def build_adjacency(geohash_cells: list) -> dict:
    """
    Build an adjacency dictionary for the Geohash grid.
    
    WHY Geohash neighbors:
    - Each cell has up to 8 neighbors (N, NE, E, SE, S, SW, W, NW)
    - We only keep neighbors that exist in our dataset (no phantom cells)
    - This becomes the action space for the RL agent
    """
    print("[Phase 0.11] Building spatial adjacency matrix...")
    
    cell_set = set(geohash_cells)
    adjacency = {}
    
    for cell in geohash_cells:
        nbrs = _geohash_neighbors(cell)
        # Only keep neighbors that exist in our dataset
        valid_nbrs = [v for v in nbrs if v in cell_set]
        adjacency[cell] = valid_nbrs
    
    # Stats
    n_edges = sum(len(v) for v in adjacency.values())
    avg_degree = n_edges / max(1, len(adjacency))
    isolated = sum(1 for v in adjacency.values() if len(v) == 0)
    
    print(f"  Nodes: {len(adjacency)}")
    print(f"  Total edges: {n_edges}")
    print(f"  Avg degree: {avg_degree:.1f}")
    print(f"  Isolated cells: {isolated}")
    
    return adjacency


def run_pipeline(csv_path: str) -> tuple:
    """
    Execute the full Phase 0 data engineering pipeline.
    
    Returns:
        (df, zone_stats, adjacency) — processed data, zone aggregates, and graph
    """
    print("=" * 60)
    print("PHASE 0: DATA ENGINEERING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load and clean
    df = load_raw_data(csv_path)
    
    # Step 2: Filter to top stations
    df = filter_top_stations(df)
    
    # Step 3: Geohash encoding
    df = encode_geohash(df)
    
    # Step 4: Temporal features
    df = extract_temporal_features(df)
    
    # Step 5: Vehicle weights
    df = compute_vehicle_weight(df)
    
    # Step 6: Violation severity
    df = compute_violation_severity(df)
    
    # Step 7: Road factor
    df = compute_road_factor(df)
    
    # Step 8: Congestion Impact Score
    df = compute_congestion_impact_score(df)
    
    # Step 9: Survival proxy
    df = compute_survival_proxy(df)
    
    # Step 10: Zone-level statistics
    zone_stats = build_zone_stats(df)
    
    # Step 11: Adjacency graph
    adjacency = build_adjacency(df['geohash'].unique().tolist())
    
    print("\n" + "=" * 60)
    print("PHASE 0 COMPLETE")
    print("=" * 60)
    print(f"  Processed violations: {len(df):,}")
    print(f"  Zone-time slots: {len(zone_stats):,}")
    print(f"  Spatial cells: {zone_stats['geohash'].nunique()}")
    print(f"  Adjacency nodes: {len(adjacency)}")
    
    return df, zone_stats, adjacency


# =============================================================================
# Main — run as script for testing
# =============================================================================
if __name__ == "__main__":
    import os
    
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 
                            'jan to may police violation_anonymized791b166.csv')
    df, zone_stats, adjacency = run_pipeline(csv_path)
    
    # Save processed data for downstream phases
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    
    df.to_parquet(os.path.join(output_dir, 'violations_processed.parquet'), index=False)
    zone_stats.to_parquet(os.path.join(output_dir, 'zone_stats.parquet'), index=False)
    
    import pickle
    with open(os.path.join(output_dir, 'adjacency.pkl'), 'wb') as f:
        pickle.dump(adjacency, f)
    
    print(f"\n  Saved processed data to {output_dir}")
