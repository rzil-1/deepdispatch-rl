"""
Phase 3: Streamlit Dashboard
============================
Interactive dashboard for the GridLock Hackathon demo.

Features:
  1. Violation Heatmap — interactive Folium map showing hotspots
  2. Congestion Impact Score overlay — zones colored by CIS
  3. Agent patrol path visualization — animated routes
  4. KPI comparison — RL vs. baselines bar charts
  5. Time-of-day slider — explore temporal patterns
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import json
import pickle
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Page Config
# =============================================================================
st.set_page_config(
    page_title="DeepDispatch RL - Parking Enforcement Intelligence",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Custom CSS for premium look
# =============================================================================
st.markdown("""
<style>
    /* Dark theme overrides */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        color: #8892b0 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        border-right: 1px solid #30363d;
    }
    
    /* Headers */
    h1 {
        background: linear-gradient(135deg, #00d2ff, #7928ca, #ff0080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    h2, h3 {
        color: #e6edf3 !important;
    }
    
    /* Cards */
    .css-1r6slb0, .css-12w0qpk {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 8px;
        border: 1px solid #30363d;
        color: #8892b0;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3a5f, #2d5a88) !important;
        border-color: #3a7bd5 !important;
        color: #ffffff !important;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(58, 123, 213, 0.1);
        border: 1px solid rgba(58, 123, 213, 0.3);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Gradient divider */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #3a7bd5, #7928ca, transparent);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Data Loading
# =============================================================================
@st.cache_data
def load_processed_data():
    """Load pre-processed data from Phase 0."""
    df = pd.read_parquet('data/processed/violations_processed.parquet')
    zone_stats = pd.read_parquet('data/processed/zone_stats.parquet')
    return df, zone_stats

@st.cache_data
def load_adjacency():
    with open('data/processed/adjacency.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_results():
    """Load evaluation results from training."""
    try:
        with open('results/evaluation_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# =============================================================================
# Header
# =============================================================================
st.markdown("# 🚔 DeepDispatch RL — Parking Enforcement Intelligence")
st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown("""
**AI-driven parking intelligence** that detects illegal parking hotspots, 
quantifies their **congestion impact**, and enables **targeted enforcement** 
through Reinforcement Learning.
""")

# Check if data exists
if not os.path.exists('data/processed/violations_processed.parquet'):
    st.error("⚠️ Processed data not found. Please run `python train.py` first.")
    st.stop()

# Load data
df, zone_stats = load_processed_data()
adjacency = load_adjacency()
results = load_results()


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    # Time selection
    selected_hour = st.slider("🕐 Hour of Day (IST)", 0, 23, 10)
    selected_dow = st.selectbox(
        "📅 Day of Week",
        options=list(range(7)),
        format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                'Friday', 'Saturday', 'Sunday'][x],
        index=0
    )
    
    st.markdown("---")
    
    # Station filter
    all_stations = sorted(df['police_station'].unique())
    selected_stations = st.multiselect(
        "🏢 Police Stations",
        options=all_stations,
        default=all_stations[:5]
    )
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Summary")
    st.metric("Total Violations", f"{len(df):,}")
    st.metric("Spatial Zones", f"{zone_stats['geohash'].nunique():,}")
    st.metric("Police Stations", f"{df['police_station'].nunique()}")
    st.metric("Date Range", f"{df['created_ist'].min().strftime('%b %Y')} — {df['created_ist'].max().strftime('%b %Y')}")


# =============================================================================
# Main Content Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Violation Heatmap", 
    "📊 Congestion Analysis",
    "🤖 RL Agent Performance",
    "⏰ Temporal Patterns",
    "🔬 Deep Dive"
])


# =============================================================================
# Tab 1: Violation Heatmap
# =============================================================================
with tab1:
    st.markdown("### 🗺️ Predictive Violation Hotspot Map")
    st.markdown("""
    This map shows **parking violation density** across Bengaluru, weighted by 
    **Congestion Impact Score (CIS)**. Brighter zones = higher enforcement priority.
    """)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Filter data by time
        time_filtered = df[df['hour'] == selected_hour]
        if selected_stations:
            time_filtered = time_filtered[time_filtered['police_station'].isin(selected_stations)]
        
        # Create Folium map centered on Bengaluru
        m = folium.Map(
            location=[12.9716, 77.5946],
            zoom_start=12,
            tiles='cartodbdark_matter',  # dark theme map
        )
        
        # Add heatmap layer weighted by CIS
        heat_data = time_filtered[['latitude', 'longitude', 'cis']].dropna().values.tolist()
        if heat_data:
            HeatMap(
                heat_data,
                min_opacity=0.3,
                max_zoom=15,
                radius=20,
                blur=15,
                gradient={0.2: '#0000ff', 0.4: '#00ffff', 0.6: '#00ff00', 
                         0.8: '#ffff00', 1.0: '#ff0000'},
            ).add_to(m)
        
        # Add top violation clusters as markers
        top_zones = zone_stats[zone_stats['hour'] == selected_hour].nlargest(10, 'total_cis')
        for _, zone in top_zones.iterrows():
            folium.CircleMarker(
                location=[zone['lat'], zone['lon']],
                radius=max(5, min(20, zone['total_cis'] / 10)),
                color='#ff4444',
                fill=True,
                fill_opacity=0.7,
                popup=f"Zone: {zone['geohash']}<br>"
                      f"Violations: {zone['violation_count']}<br>"
                      f"Avg CIS: {zone['mean_cis']:.1f}<br>"
                      f"Total CIS: {zone['total_cis']:.0f}",
            ).add_to(m)
        
        st_folium(m, width=800, height=500)
    
    with col2:
        st.markdown("### 🎯 Top Hotspots")
        st.markdown(f"**{selected_hour:02d}:00 IST**")
        
        for i, (_, zone) in enumerate(top_zones.iterrows()):
            if i >= 5:
                break
            severity = "🔴" if zone['mean_cis'] > 3 else "🟡" if zone['mean_cis'] > 1.5 else "🟢"
            st.markdown(f"""
            {severity} **Zone {zone['geohash'][:6]}**  
            CIS: `{zone['mean_cis']:.1f}` | Count: `{zone['violation_count']}`
            """)


# =============================================================================
# Tab 2: Congestion Analysis
# =============================================================================
with tab2:
    st.markdown("### 📊 Congestion Impact Analysis")
    st.markdown("""
    Understanding **which** violations cause the most congestion, not just 
    **where** they happen. This is the key insight that drives our RL agent.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CIS by Vehicle Type
        vtype_cis = df.groupby('effective_vehicle_type').agg(
            count=('id', 'count'),
            avg_cis=('cis', 'mean'),
            total_cis=('cis', 'sum')
        ).reset_index().sort_values('total_cis', ascending=True)
        
        fig_vehicle = px.bar(
            vtype_cis.tail(12),
            x='total_cis',
            y='effective_vehicle_type',
            orientation='h',
            color='avg_cis',
            color_continuous_scale='RdYlGn_r',
            title='Congestion Impact by Vehicle Type',
            labels={'total_cis': 'Total CIS', 'effective_vehicle_type': 'Vehicle Type',
                    'avg_cis': 'Avg CIS'},
        )
        fig_vehicle.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
        )
        st.plotly_chart(fig_vehicle, use_container_width=True)
    
    with col2:
        # CIS by Police Station
        station_cis = df.groupby('police_station').agg(
            count=('id', 'count'),
            avg_cis=('cis', 'mean'),
            total_cis=('cis', 'sum')
        ).reset_index().sort_values('total_cis', ascending=True)
        
        fig_station = px.bar(
            station_cis.tail(10),
            x='total_cis',
            y='police_station',
            orientation='h',
            color='avg_cis',
            color_continuous_scale='Viridis',
            title='Congestion Impact by Police Station',
            labels={'total_cis': 'Total CIS', 'police_station': 'Station',
                    'avg_cis': 'Avg CIS'},
        )
        fig_station.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
        )
        st.plotly_chart(fig_station, use_container_width=True)
    
    # Key insight callout
    st.markdown("""
    <div class="info-box">
    💡 <b>Key Insight:</b> While SCOOTERS and CARS make up 60%+ of violations, 
    MAXI-CABs, LGVs, and TANKERs contribute disproportionately to congestion impact. 
    Our RL agent learns to <b>prioritize high-CIS violations</b> — a TANKER on 
    Outer Ring Road is worth 10× more than a scooter in a residential lane.
    </div>
    """, unsafe_allow_html=True)
    
    # CIS Distribution
    fig_dist = px.histogram(
        df, x='cis', nbins=50,
        title='Distribution of Congestion Impact Scores',
        color_discrete_sequence=['#3a7bd5'],
        labels={'cis': 'Congestion Impact Score', 'count': 'Frequency'},
    )
    fig_dist.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# =============================================================================
# Tab 3: RL Agent Performance
# =============================================================================
with tab3:
    st.markdown("### 🤖 RL Agent vs. Baseline Comparison")
    
    if results is None:
        st.warning("⚠️ Training results not found. Run `python train.py` first.")
        st.markdown("""
        Once training completes, this tab will show:
        - **Citation Yield** comparison (RL vs. random, greedy, fixed-route)
        - **Congestion Relief** measured by total CIS collected
        - **Patrol Efficiency** (citations per km traveled)
        - **Zone Coverage** analysis
        """)
    else:
        # KPI metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Find best RL agent
        rl_agents = [k for k in results if 'DQN' in k or 'PPO' in k]
        best_rl = max(rl_agents, key=lambda x: results[x]['citations_mean']) if rl_agents else None
        random_key = 'Random Patrol'
        
        if best_rl and random_key in results:
            rl_r = results[best_rl]
            rand_r = results[random_key]
            
            improvement = ((rl_r['citations_mean'] - rand_r['citations_mean']) / 
                          max(1, rand_r['citations_mean']) * 100)
            
            with col1:
                st.metric("Citations/Shift (RL)", f"{rl_r['citations_mean']:.1f}",
                         delta=f"+{improvement:.0f}% vs Random")
            with col2:
                st.metric("CIS Collected", f"{rl_r['cis_collected_mean']:.1f}",
                         delta=f"{rl_r['cis_collected_mean'] - rand_r['cis_collected_mean']:.1f}")
            with col3:
                st.metric("Zone Coverage", f"{rl_r['coverage_mean']:.0%}")
            with col4:
                st.metric("Patrol Efficiency", f"{rl_r['efficiency']:.3f}", 
                         help="Citations per unit distance")
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        
        # Bar chart comparison
        metrics_to_plot = ['citations_mean', 'cis_collected_mean', 'coverage_mean', 'efficiency']
        metric_labels = ['Citations/Shift', 'CIS Collected', 'Zone Coverage', 'Efficiency']
        
        fig = make_subplots(rows=1, cols=4, subplot_titles=metric_labels)
        
        colors = {
            'DQN (Ours)': '#00d2ff',
            'PPO (Ours)': '#7928ca',
            'Random Patrol': '#ff6b6b',
            'Greedy Nearest': '#ffa502',
            'Fixed Route': '#a4b0be',
        }
        
        for col_idx, metric in enumerate(metrics_to_plot):
            for agent_name, agent_results in results.items():
                fig.add_trace(
                    go.Bar(
                        name=agent_name if col_idx == 0 else None,
                        x=[agent_name.split(' ')[0]],
                        y=[agent_results[metric]],
                        marker_color=colors.get(agent_name, '#888'),
                        showlegend=(col_idx == 0),
                    ),
                    row=1, col=col_idx + 1,
                )
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed results table
        st.markdown("#### Detailed Results")
        results_df = pd.DataFrame(results).T
        results_df.index.name = 'Agent'
        st.dataframe(
            results_df[['citations_mean', 'citations_std', 'cis_collected_mean',
                        'distance_mean', 'coverage_mean', 'efficiency', 'reward_mean']].round(2),
            use_container_width=True
        )


# =============================================================================
# Tab 4: Temporal Patterns
# =============================================================================
with tab4:
    st.markdown("### ⏰ Temporal Violation Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly distribution
        hourly = df.groupby('hour').agg(
            count=('id', 'count'),
            avg_cis=('cis', 'mean'),
            total_cis=('cis', 'sum'),
        ).reset_index()
        
        fig_hourly = go.Figure()
        fig_hourly.add_trace(go.Bar(
            x=hourly['hour'], y=hourly['count'],
            name='Violation Count',
            marker_color='rgba(58, 123, 213, 0.6)',
        ))
        fig_hourly.add_trace(go.Scatter(
            x=hourly['hour'], y=hourly['total_cis'],
            name='Total CIS',
            yaxis='y2',
            line=dict(color='#ff0080', width=3),
        ))
        fig_hourly.update_layout(
            title='Hourly Violation & Congestion Pattern (IST)',
            xaxis_title='Hour of Day',
            yaxis_title='Violation Count',
            yaxis2=dict(title='Total CIS', overlaying='y', side='right'),
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
        )
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        # Day of week distribution
        daily = df.groupby('day_of_week').agg(
            count=('id', 'count'),
            avg_cis=('cis', 'mean'),
        ).reset_index()
        daily['day_name'] = daily['day_of_week'].map({
            0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 
            4: 'Fri', 5: 'Sat', 6: 'Sun'
        })
        
        fig_daily = px.bar(
            daily, x='day_name', y='count',
            color='avg_cis',
            color_continuous_scale='RdYlGn_r',
            title='Violations by Day of Week',
            labels={'count': 'Violations', 'day_name': 'Day', 'avg_cis': 'Avg CIS'},
        )
        fig_daily.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
        )
        st.plotly_chart(fig_daily, use_container_width=True)
    
    # Heatmap: hour × day_of_week
    pivot = df.groupby(['hour', 'day_of_week']).size().unstack(fill_value=0)
    pivot.columns = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    fig_heatmap = px.imshow(
        pivot.values,
        x=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        y=[f"{h:02d}:00" for h in range(24)],
        color_continuous_scale='Inferno',
        title='Violation Density: Hour × Day of Week',
        labels={'x': 'Day', 'y': 'Hour', 'color': 'Violations'},
    )
    fig_heatmap.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500,
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)


# =============================================================================
# Tab 5: Deep Dive
# =============================================================================
with tab5:
    st.markdown("### 🔬 Technical Deep Dive")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Violation Type Breakdown")
        # Parse violation types
        vtype_counts = df['violation_type'].value_counts().head(10).reset_index()
        vtype_counts.columns = ['type', 'count']
        # Clean up the JSON-like strings
        vtype_counts['type'] = vtype_counts['type'].str.replace('[', '').str.replace(']', '').str.replace('"', '')
        
        fig_vtype = px.pie(
            vtype_counts, values='count', names='type',
            title='Top 10 Violation Types',
            color_discrete_sequence=px.colors.sequential.Plasma_r,
        )
        fig_vtype.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
        )
        st.plotly_chart(fig_vtype, use_container_width=True)
    
    with col2:
        st.markdown("#### Validation Status")
        if 'validation_status' in df.columns:
            val_counts = df['validation_status'].value_counts().reset_index()
            val_counts.columns = ['status', 'count']
            val_counts = val_counts[val_counts['status'].notna()]
            
            fig_val = px.pie(
                val_counts, values='count', names='status',
                title='Violation Validation Status',
                color_discrete_sequence=['#00d2ff', '#ff6b6b', '#ffa502', '#a4b0be', '#7928ca'],
            )
            fig_val.update_layout(
                template='plotly_dark',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=400,
            )
            st.plotly_chart(fig_val, use_container_width=True)
    
    # Architecture explanation
    st.markdown("#### 🏗️ System Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────┐
    │                    RAW VIOLATION DATA                     │
    │              298K records, Nov 2023 - Apr 2024           │
    └──────────────────────┬──────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Phase 0:   │
                    │  Data Eng.  │ → Geohash encoding, temporal bins,
                    │             │   CIS scoring, spatial adjacency
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Phase 1:   │
                    │  Congestion │ → Violation probability maps,
                    │  Intel      │   KDE smoothing, hotspot ranking
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Phase 2:   │
                    │  RL Env &   │ → Gymnasium environment,
                    │  Training   │   DQN + PPO agents
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Phase 3:   │
                    │  Dashboard  │ → This Streamlit app!
                    └─────────────┘
    ```
    """)
    
    st.markdown("""
    <div class="info-box">
    🧠 <b>Why Reinforcement Learning?</b><br>
    Traditional enforcement is patrol-based and reactive. Our RL agent learns 
    to <b>proactively position itself</b> in zones where high-CIS violations are 
    <b>likely to appear</b>, rather than chasing them after they occur. This is 
    the core of the <b>Traveling Officer Problem (TOP)</b> formulation.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #8892b0; font-size: 0.8rem;">
    Built for <b>GridLock Hackathon 2.0</b> | DeepDispatch RL: Parking Enforcement Intelligence |
    Powered by DQN/PPO + Geospatial Analytics
</div>
""", unsafe_allow_html=True)
