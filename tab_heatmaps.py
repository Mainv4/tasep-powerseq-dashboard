"""
dashboard/tab_heatmaps.py
Parameter Space Heatmaps tab for the Active Polymer Dashboard.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

from utils import create_heatmap, find_simulation_figures
from constants import HEATMAP_METRICS, NORMALIZE_OPTIONS
from data_loading import get_global_reference_values


def render(df_physics):
    """Render the Parameter Space Heatmaps tab."""
    st.header("Parameter Space Heatmaps")
    st.markdown("Visualize observables as functions of (α_in, α_out).")

    # Filter to standard TASEP simulations only (Pe=10, pol=1, exclude references)
    df_tasep = df_physics[
        (df_physics['Pe'] == 10.0) &
        (df_physics['alpha_in'] > 0) &
        (df_physics['alpha_out'] > 0) &
        (df_physics['pol'].astype(str) == '1')
    ].copy()

    col1, col2 = st.columns([1, 3])

    with col1:
        # Observable selector
        available_metrics = {k: v for k, v in HEATMAP_METRICS.items()
                           if k in df_tasep.columns and df_tasep[k].notna().sum() > 0}

        observable = st.selectbox(
            "Select observable",
            options=list(available_metrics.keys()),
            format_func=lambda x: available_metrics[x],
            index=0,
            key="heat_observable"
        )

        # Normalization selector
        normalize_by = st.selectbox(
            "Normalize by",
            options=NORMALIZE_OPTIONS,
            index=0,
            key="heat_normalize_by"
        )

        # Colorscale selector
        colorscale = st.selectbox(
            "Colorscale",
            options=['Jet', 'Viridis', 'Plasma', 'RdYlGn_r', 'Inferno'],
            index=0,
            key="heat_colorscale"
        )

    with col2:
        heatmap_df = df_tasep[['alpha_in', 'alpha_out', observable]].dropna()

        # Apply normalization
        title_suffix = ""
        if normalize_by != "None" and len(heatmap_df) > 0:
            ref_type = "PASSIVE" if normalize_by == "Passive" else "ACTIVE_PUR"
            ref_values = get_global_reference_values(df_physics)
            if ref_type in ref_values and observable in ref_values[ref_type]:
                ref_val = ref_values[ref_type][observable]
                if ref_val != 0 and not np.isnan(ref_val):
                    heatmap_df[observable] = heatmap_df[observable] / ref_val
                    title_suffix = f" / {normalize_by}"
                else:
                    st.warning(f"Reference value for {available_metrics[observable]} is {'zero' if ref_val == 0 else 'NaN'}. Showing raw values.")
            else:
                st.warning(f"No {normalize_by} reference found for {available_metrics[observable]}. Showing raw values.")

        if len(heatmap_df) == 0:
            st.warning(f"No data available for {available_metrics[observable]}")
        else:
            title = available_metrics[observable] + title_suffix
            fig = create_heatmap(heatmap_df, observable, title, colorscale)
            st.plotly_chart(fig, use_container_width=True)

            # Statistics row
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Min", f"{heatmap_df[observable].min():.4f}")
            col_s2.metric("Max", f"{heatmap_df[observable].max():.4f}")
            col_s3.metric("Mean", f"{heatmap_df[observable].mean():.4f}")
            col_s4.metric("Std", f"{heatmap_df[observable].std():.4f}")

    # Simulation detail panel
    st.markdown("---")
    st.markdown("### Simulation Details")

    col1, col2 = st.columns(2)
    alpha_in_values = sorted(df_tasep['alpha_in'].unique())
    alpha_out_values = sorted(df_tasep['alpha_out'].unique())

    with col1:
        alpha_in_sel = st.selectbox("α_in", options=alpha_in_values, index=0, key="heat_alpha_in")
    with col2:
        alpha_out_sel = st.selectbox("α_out", options=alpha_out_values, index=0, key="heat_alpha_out")

    # Show all observable values for this simulation
    sim = df_tasep[(df_tasep['alpha_in'] == alpha_in_sel) & (df_tasep['alpha_out'] == alpha_out_sel)]

    if len(sim) == 0:
        st.warning(f"No simulation found for α_in={alpha_in_sel}, α_out={alpha_out_sel}")
    else:
        sim_data = sim.iloc[0]

        # Display metric cards for key observables
        metrics_to_show = {k: v for k, v in HEATMAP_METRICS.items()
                          if k in sim_data.index and not pd.isna(sim_data.get(k))}
        if metrics_to_show:
            cols = st.columns(min(len(metrics_to_show), 4))
            for i, (col_name, display_name) in enumerate(metrics_to_show.items()):
                cols[i % len(cols)].metric(display_name, f"{sim_data[col_name]:.4f}")

        # Display figures
        base_path = Path(__file__).parent.parent
        sim_name = (f"Pe_{sim_data['Pe']}_alpha_in_{alpha_in_sel}"
                    f"_alpha_out_{alpha_out_sel}_kappa_{sim_data['kappa']}"
                    f"_pol_{sim_data['pol']}")
        figures = find_simulation_figures(sim_name, base_path)

        if figures:
            st.markdown("#### Analysis Figures")
            st.markdown(f"Found **{len(figures)}** figures:")
            fig_cols = st.columns(2)
            for i, fig_info in enumerate(figures):
                with fig_cols[i % 2]:
                    st.markdown(f"**{fig_info['name']}**")
                    try:
                        st.image(str(fig_info['path']), use_container_width=True)
                    except Exception:
                        st.warning(f"Could not load: {fig_info['filename']}")
                    st.caption(fig_info['filename'])
