"""
dashboard/tab_corruption.py
Corruption analysis tab for the Active Polymer Dashboard.
"""

import streamlit as st
from utils import create_heatmap, display_simulation_details, calculate_statistics
from constants import CORRUPTION_METRICS


def render(df_corruption):
    """Render the Corruption analysis tab."""
    st.header("Trajectory Corruption Analysis")

    # Filter to standard TASEP simulations only (Pe=10, pol=1)
    df_corruption = df_corruption[
        (df_corruption['Pe'] == 10.0) &
        (df_corruption['alpha_in'] > 0) &
        (df_corruption['alpha_out'] > 0) &
        (df_corruption['pol'] == 1)
    ].copy()

    stats = calculate_statistics(df_corruption)

    # Corruption-specific sidebar metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Corruption")
    st.sidebar.metric("Sims with corruption",
                      f"{stats['sims_with_corruption']} ({stats['corruption_rate']:.1f}%)")
    st.sidebar.metric("Avg Blocks Lost", f"{stats['avg_blocks_lost_pct']:.2f}%")
    st.sidebar.metric("Worst Case", f"{stats['max_blocks_lost_pct']:.2f}%")

    # Metric selector
    selected_metric = st.selectbox(
        "Select Metric to Visualize",
        options=list(CORRUPTION_METRICS.keys()),
        format_func=lambda x: CORRUPTION_METRICS[x],
        index=0,
        key="corrupt_metric"
    )

    # Heatmap
    st.markdown("### Heatmap: Corruption in Parameter Space")
    colorscale = 'RdYlGn_r' if 'lost' in selected_metric or 'eliminated' in selected_metric else 'Viridis'
    fig_heatmap = create_heatmap(df_corruption, selected_metric, CORRUPTION_METRICS[selected_metric], colorscale)
    st.plotly_chart(fig_heatmap, width="stretch")

    # Simulation selector
    st.markdown("### Select a Simulation for Details")
    col1, col2 = st.columns(2)
    with col1:
        alpha_in_selected = st.selectbox(
            "α_in (Entry Rate)",
            options=sorted(df_corruption['alpha_in'].unique()),
            index=0,
            key="corrupt_alpha_in"
        )
    with col2:
        alpha_out_selected = st.selectbox(
            "α_out (Exit Rate)",
            options=sorted(df_corruption['alpha_out'].unique()),
            index=0,
            key="corrupt_alpha_out"
        )

    # Simulation details
    display_simulation_details(df_corruption, alpha_in_selected, alpha_out_selected)
