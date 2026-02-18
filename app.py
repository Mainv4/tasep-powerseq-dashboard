#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dashboard/app.py
Active Polymer Simulation Dashboard — PowerSeq data.

Usage:
    streamlit run dashboard/app.py
"""

import streamlit as st

from data_loading import load_physics_data, load_corruption_data
import tab_corruption
import tab_heatmaps
import tab_global_correlation
import tab_local_correlation

# Page configuration
st.set_page_config(
    page_title="Active Polymer Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data
df_physics = load_physics_data()
df_corruption = load_corruption_data()

# Sidebar - Dataset overview
st.sidebar.title("Dataset Overview")
st.sidebar.metric("Total Simulations", len(df_physics))
st.sidebar.metric("α_in range", f"{df_physics['alpha_in'].min():.1f} - {df_physics['alpha_in'].max():.1f}")
st.sidebar.metric("α_out range", f"{df_physics['alpha_out'].min():.1f} - {df_physics['alpha_out'].max():.1f}")

# Main title
st.title("Active Polymer Simulation Dashboard")
st.markdown("---")

# Tab layout
tab_corrupt, tab_heat, tab_global, tab_local = st.tabs([
    "Corruption", "Heatmaps", "Global Correlation", "Local Correlation"
])

with tab_corrupt:
    tab_corruption.render(df_corruption)

with tab_heat:
    tab_heatmaps.render(df_physics)

with tab_global:
    tab_global_correlation.render(df_physics)

with tab_local:
    tab_local_correlation.render()

