"""
dashboard/tab_global_correlation.py
Global Correlation (chain-level) scatter explorer tab.
"""

import streamlit as st
import numpy as np
import plotly.express as px

from constants import GLOBAL_OBSERVABLES, CORR_LABELS, NORMALIZE_OPTIONS
from data_loading import get_global_reference_values


def render(df_physics):
    """Render the Global Correlation tab."""
    st.header("Global Correlation (chain-level)")
    st.markdown("Explore correlations between chain-level observables across the parameter space.")

    # Controls row
    ctrl_col1, ctrl_col2 = st.columns([1, 3])

    with ctrl_col1:
        # Reference toggle
        show_refs = st.checkbox("Show reference data", value=True, key="global_show_refs")

        # Normalization selector
        normalize_by = st.selectbox(
            "Normalize by",
            options=NORMALIZE_OPTIONS,
            index=0,
            key="global_normalize_by"
        )

        # Alpha filter
        filter_mode = st.radio(
            "Filter",
            options=["All", "Fix α_in", "Fix α_out", "α_in = α_out"],
            index=0,
            key="global_filter_mode"
        )

    # Filter dataframe
    df = df_physics.copy()

    # Reference handling: references have alpha_in==0 and alpha_out==0
    if not show_refs:
        df = df[(df['alpha_in'] != 0) | (df['alpha_out'] != 0)]

    # Alpha filter
    if filter_mode == "Fix α_in":
        alpha_in_values = sorted(df[df['alpha_in'] > 0]['alpha_in'].unique())
        if alpha_in_values:
            selected_alpha_in = st.selectbox("α_in value", options=alpha_in_values, key="global_fix_alpha_in")
            # Keep selected alpha_in + references (if shown)
            mask = df['alpha_in'] == selected_alpha_in
            if show_refs:
                mask = mask | ((df['alpha_in'] == 0) & (df['alpha_out'] == 0))
            df = df[mask]
    elif filter_mode == "Fix α_out":
        alpha_out_values = sorted(df[df['alpha_out'] > 0]['alpha_out'].unique())
        if alpha_out_values:
            selected_alpha_out = st.selectbox("α_out value", options=alpha_out_values, key="global_fix_alpha_out")
            mask = df['alpha_out'] == selected_alpha_out
            if show_refs:
                mask = mask | ((df['alpha_in'] == 0) & (df['alpha_out'] == 0))
            df = df[mask]
    elif filter_mode == "α_in = α_out":
        mask = df['alpha_in'] == df['alpha_out']
        if show_refs:
            mask = mask | ((df['alpha_in'] == 0) & (df['alpha_out'] == 0))
        df = df[mask]

    # Filter available observables to those present in data with non-null values
    available_obs = [col for col in GLOBAL_OBSERVABLES if col in df.columns and df[col].notna().sum() > 0]

    if len(available_obs) < 2:
        st.warning("Not enough observables with data.")
        return

    # Axis selectors
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        x_var = st.selectbox(
            "X-axis",
            options=available_obs,
            index=0,
            format_func=lambda x: CORR_LABELS.get(x, x),
            key="global_x_var"
        )

    with col2:
        y_var = st.selectbox(
            "Y-axis",
            options=available_obs,
            index=min(1, len(available_obs) - 1),
            format_func=lambda x: CORR_LABELS.get(x, x),
            key="global_y_var"
        )

    with col3:
        color_options = ['alpha_in', 'alpha_out', 'alpha_ratio', 'alpha_product'] + available_obs
        color_var = st.selectbox(
            "Color by",
            options=color_options,
            index=0,
            format_func=lambda x: CORR_LABELS.get(x, x),
            key="global_color_var"
        )

    with col4:
        size_options = ['None'] + available_obs
        size_var = st.selectbox(
            "Size by",
            options=size_options,
            index=0,
            format_func=lambda x: CORR_LABELS.get(x, x) if x != 'None' else 'None',
            key="global_size_var"
        )

    # Scale toggles
    scale_col1, scale_col2, _ = st.columns([1, 1, 6])
    with scale_col1:
        log_x = st.checkbox("Log X", value=False, key="global_log_x")
    with scale_col2:
        log_y = st.checkbox("Log Y", value=False, key="global_log_y")

    # Filter to non-NaN data
    plot_cols = [x_var, y_var, color_var]
    if size_var != 'None':
        plot_cols.append(size_var)
    # Add hover columns
    for c in ['alpha_in', 'alpha_out']:
        if c not in plot_cols:
            plot_cols.append(c)

    plot_df = df[plot_cols].dropna(subset=[x_var, y_var])

    if len(plot_df) == 0:
        st.warning(f"No data available for {CORR_LABELS.get(x_var, x_var)} vs {CORR_LABELS.get(y_var, y_var)}")
        return

    # Apply normalization to X and Y axes
    norm_suffix_x = ""
    norm_suffix_y = ""
    if normalize_by != "None":
        ref_type = "PASSIVE" if normalize_by == "Passive" else "ACTIVE_PUR"
        ref_values = get_global_reference_values(df_physics)
        if ref_type in ref_values:
            ref = ref_values[ref_type]
            for var, suffix_attr in [(x_var, 'x'), (y_var, 'y')]:
                if var in ref and ref[var] != 0 and not np.isnan(ref[var]):
                    plot_df[var] = plot_df[var] / ref[var]
                    if suffix_attr == 'x':
                        norm_suffix_x = f" / {normalize_by}"
                    else:
                        norm_suffix_y = f" / {normalize_by}"
                elif var in GLOBAL_OBSERVABLES:
                    st.warning(f"No valid {normalize_by} reference for {CORR_LABELS.get(var, var)}. Showing raw values.")
        else:
            st.warning(f"No {normalize_by} reference data found. Showing raw values.")

    # Build labels with normalization suffixes
    plot_labels = {col: CORR_LABELS.get(col, col) for col in plot_cols}
    if norm_suffix_x:
        plot_labels[x_var] = CORR_LABELS.get(x_var, x_var) + norm_suffix_x
    if norm_suffix_y:
        plot_labels[y_var] = CORR_LABELS.get(y_var, y_var) + norm_suffix_y

    scatter_kwargs = dict(
        data_frame=plot_df,
        x=x_var,
        y=y_var,
        color=color_var,
        hover_data=['alpha_in', 'alpha_out'],
        labels=plot_labels,
        color_continuous_scale='jet',
        log_x=log_x,
        log_y=log_y,
        height=600,
    )

    if size_var != 'None':
        scatter_kwargs['size'] = size_var
        scatter_kwargs['size_max'] = 20

    fig = px.scatter(**scatter_kwargs)

    fig.update_traces(marker=dict(line=dict(width=0.5, color='white')))
    fig.update_layout(
        font=dict(size=12),
        template='plotly_white',
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        margin=dict(r=120, b=100),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Statistics
    st.metric("Data points", len(plot_df))
