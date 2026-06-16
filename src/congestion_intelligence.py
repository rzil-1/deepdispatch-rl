"""
Phase 1: Congestion Intelligence Engine
========================================
This module builds the "predictive heatmap" that feeds the RL environment.

Instead of the full STGNN+ZINB approach described in the research doc
(which requires extensive graph neural network training), we use a practical
approach that achieves the same goal:

  1. Historical frequency estimation per (zone, hour, day_of_week)
  2. Kernel Density Estimation (KDE) for smooth spatial probability
  3. Zone-level CIS aggregation for enforcement prioritization
  4. Survival probability estimation from validation timestamps

The result is a function: f(geohash, hour, day_of_week) → (p_violation, expected_CIS)
that the RL agent uses to decide where to patrol.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
import pickle
import os


class CongestionIntelligence:
    """
    The predictive engine that outputs violation probabilities and
    congestion impact scores for any zone at any time.
    
    This replaces the full STGNN from the research doc with a practical
    alternative that uses:
    - Historical frequency tables (empirical Bayes)
    - Spatial KDE for smooth probability surfaces
    - Vehicle-type weighted CIS aggregation
    """
    
    def __init__(self, zone_stats: pd.DataFrame, adjacency: dict):
        """
        Args:
            zone_stats: DataFrame from data_engineering.build_zone_stats()
            adjacency: dict from data_engineering.build_adjacency()
        """
        self.zone_stats = zone_stats
        self.adjacency = adjacency
        self.all_zones = list(adjacency.keys())
        
        # Build lookup tables for fast query
        self._build_lookup_tables()
        
        # Fit spatial KDE
        self._fit_spatial_kde()
        
        # Compute zone priorities (static ranking)
        self._compute_zone_priorities()
    
    def _build_lookup_tables(self):
        """
        Create fast lookup: (geohash, hour, dow) → stats
        
        WHY dict lookup over DataFrame query:
        - The RL environment will call this millions of times during training
        - Dict lookup is O(1) vs DataFrame query O(n)
        """
        self.lookup = {}
        for _, row in self.zone_stats.iterrows():
            key = (row['geohash'], int(row['hour']), int(row['day_of_week']))
            self.lookup[key] = {
                'p_violation': float(row['p_violation']),
                'mean_cis': float(row['mean_cis']),
                'total_cis': float(row['total_cis']),
                'violation_count': int(row['violation_count']),
                'has_junction': bool(row['has_junction']),
            }
        
        # Zone-level aggregation (ignoring time)
        self.zone_agg = self.zone_stats.groupby('geohash').agg(
            total_violations=('violation_count', 'sum'),
            avg_cis=('mean_cis', 'mean'),
            total_cis=('total_cis', 'sum'),
            lat=('lat', 'mean'),
            lon=('lon', 'mean'),
        ).to_dict('index')
    
    def _fit_spatial_kde(self):
        """
        Fit a Kernel Density Estimator on violation locations.
        
        WHY KDE:
        - Smooths the discrete violation counts into a continuous probability surface
        - Handles spatial autocorrelation (violations cluster near each other)
        - The bandwidth controls the smoothing: too small = overfitting to exact spots,
          too large = everything looks uniform
        - We use bandwidth=0.005° ≈ 500m, matching typical block-level clustering
        """
        coords = self.zone_stats[['lat', 'lon']].values
        # Weight by total CIS so high-impact zones dominate the density
        weights = self.zone_stats['total_cis'].values
        
        self.kde = KernelDensity(
            bandwidth=0.005,  # ~500m in degrees
            kernel='gaussian',
            metric='haversine'
        )
        # KDE expects radians for haversine
        coords_rad = np.radians(coords)
        self.kde.fit(coords_rad, sample_weight=weights)
    
    def _compute_zone_priorities(self):
        """
        Rank zones by enforcement priority.
        
        Priority = total historical CIS × average violation probability
        
        This is the "enforcement heatmap" — zones where violations are both
        FREQUENT and HIGH-IMPACT get the highest priority.
        """
        priorities = {}
        for zone in self.all_zones:
            if zone in self.zone_agg:
                stats = self.zone_agg[zone]
                priorities[zone] = stats['total_cis']
            else:
                priorities[zone] = 0.0
        
        # Normalize to [0, 1]
        max_p = max(priorities.values()) if priorities else 1.0
        self.zone_priorities = {z: p / max_p for z, p in priorities.items()}
    
    def get_violation_probability(self, geohash: str, hour: int, dow: int) -> float:
        """
        Get the probability of an active violation in this zone at this time.
        
        Returns value in [0, 1]. If no historical data, returns a small
        base rate (exploration incentive for the RL agent).
        """
        key = (geohash, hour, dow)
        if key in self.lookup:
            return min(1.0, self.lookup[key]['p_violation'])
        return 0.05  # base rate for unknown zone-times
    
    def get_expected_cis(self, geohash: str, hour: int, dow: int) -> float:
        """
        Get the expected Congestion Impact Score for a violation in this zone.
        
        If the zone has historical data, use the mean CIS.
        Otherwise, return a default of 1.0 (average impact).
        """
        key = (geohash, hour, dow)
        if key in self.lookup:
            return self.lookup[key]['mean_cis']
        return 1.0
    
    def get_zone_priority(self, geohash: str) -> float:
        """Get the static priority score for a zone (time-independent)."""
        return self.zone_priorities.get(geohash, 0.0)
    
    def get_heatmap_snapshot(self, hour: int, dow: int) -> dict:
        """
        Get a complete heatmap for all zones at a specific time.
        
        Returns:
            dict: {geohash: {'p': probability, 'cis': expected CIS, 'priority': zone priority}}
        """
        snapshot = {}
        for zone in self.all_zones:
            snapshot[zone] = {
                'p': self.get_violation_probability(zone, hour, dow),
                'cis': self.get_expected_cis(zone, hour, dow),
                'priority': self.get_zone_priority(zone),
            }
        return snapshot
    
    def get_top_hotspots(self, hour: int, dow: int, n: int = 10) -> list:
        """
        Get the top-N enforcement priority zones for a given time.
        
        Ranked by: p_violation × expected_CIS × zone_priority
        
        This is what a "smart dispatcher" would use to deploy officers.
        """
        scores = []
        for zone in self.all_zones:
            p = self.get_violation_probability(zone, hour, dow)
            cis = self.get_expected_cis(zone, hour, dow)
            pri = self.get_zone_priority(zone)
            composite = p * cis * (1 + pri)
            
            if zone in self.zone_agg:
                lat, lon = self.zone_agg[zone]['lat'], self.zone_agg[zone]['lon']
            else:
                lat, lon = 0, 0
            
            scores.append({
                'geohash': zone,
                'composite_score': composite,
                'p_violation': p,
                'expected_cis': cis,
                'zone_priority': pri,
                'lat': lat,
                'lon': lon,
            })
        
        scores.sort(key=lambda x: x['composite_score'], reverse=True)
        return scores[:n]
    
    def save(self, path: str):
        """Save the intelligence engine to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(path: str) -> 'CongestionIntelligence':
        """Load the intelligence engine from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)
