"""
dashboard/tab_local_correlation.py
Local Correlation (segment-level) scatter explorer tab.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    SIM_TYPE_STYLES, REFERENCE_TYPES,
    CORR_LOCAL_GROUPS, CORR_GLOBAL_GROUPS, CORR_LABELS,
    MAX_SUBPLOTS_DISJOINT, MAX_SUBPLOTS_SLIDING,
)
from data_loading import load_segment_data, compute_reference_table
from utils import calc_marker_size


def _flatten_groups(groups):
    """Flatten list of (group_name, [cols]) into flat list of cols."""
    return [col for _, cols in groups for col in cols]


def render():
    """Render the Local Correlation (segment-level) tab."""
    st.header("Local Correlation (segment-level)")
    st.markdown("Explore correlations between segment-level observables along the polymer chain.")

    # Controls
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns(4)

    with ctrl_col1:
        seg_mode = st.radio("Segment mode", ["Disjoint", "Sliding"], index=0, key="local_seg_mode")

    with ctrl_col2:
        show_refs = st.checkbox("Show reference data", value=True, key="local_show_refs")

    with ctrl_col3:
        normalize_by = st.selectbox(
            "Normalize by",
            options=["None", "Passive", "Active pur"],
            index=0,
            key="local_normalize_by"
        )

    with ctrl_col4:
        filter_mode = st.radio(
            "Filter",
            options=["All", "Fix α_in", "Fix α_out", "α_in = α_out"],
            index=0,
            key="local_filter_mode"
        )

    # Load segment data
    mode = seg_mode.lower()
    df_seg = load_segment_data(mode)

    if df_seg is None:
        st.error(f"Segment data file not found: data_compiled_segments_{mode}.csv")
        return

    # Build selectable columns
    local_cols = [c for c in _flatten_groups(CORR_LOCAL_GROUPS) if c in df_seg.columns]
    global_cols = [c for c in _flatten_groups(CORR_GLOBAL_GROUPS) if c in df_seg.columns]
    all_cols = local_cols + [c for c in global_cols if c not in local_cols]

    # Axis selectors
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        x_var = st.selectbox(
            "X-axis", options=local_cols,
            index=local_cols.index("segment_center") if "segment_center" in local_cols else 0,
            format_func=lambda x: CORR_LABELS.get(x, x),
            key="local_x_var"
        )

    with col2:
        y_var = st.selectbox(
            "Y-axis", options=local_cols,
            index=local_cols.index("rho_tasep") if "rho_tasep" in local_cols else 1,
            format_func=lambda x: CORR_LABELS.get(x, x),
            key="local_y_var"
        )

    with col3:
        color_var = st.selectbox(
            "Color by", options=all_cols,
            index=all_cols.index("alpha_in") if "alpha_in" in all_cols else 0,
            format_func=lambda x: CORR_LABELS.get(x, x),
            key="local_color_var"
        )

    with col4:
        size_options = ["None"] + all_cols
        size_var = st.selectbox(
            "Size by", options=size_options, index=0,
            format_func=lambda x: CORR_LABELS.get(x, x) if x != "None" else "None",
            key="local_size_var"
        )

    # Plot options
    opt_col1, opt_col2, opt_col3, opt_col4 = st.columns([1, 1, 1, 3])
    with opt_col1:
        log_x = st.checkbox("Log X", value=False, key="local_log_x")
    with opt_col2:
        log_y = st.checkbox("Log Y", value=False, key="local_log_y")
    with opt_col3:
        link_points = st.checkbox("Link points", value=False, key="local_link_points",
                                  disabled=(mode != "disjoint"))
    with opt_col4:
        display_mode = st.radio(
            "Display", options=["All couples", "Separate figures"],
            index=0, horizontal=True, key="local_display_mode"
        )
    # Only link in disjoint mode
    link_points = link_points and (mode == "disjoint")

    # Apply filters
    df = df_seg.copy()

    # Reference filtering
    if not show_refs:
        df = df[~df['sim_type'].isin(REFERENCE_TYPES)]

    # Alpha filter
    if filter_mode == "Fix α_in":
        alpha_in_values = sorted(df[~df['sim_type'].isin(REFERENCE_TYPES)]['alpha_in'].unique())
        if alpha_in_values:
            sel = st.selectbox("α_in value", options=alpha_in_values, key="local_fix_alpha_in")
            mask = df['alpha_in'] == sel
            if show_refs:
                mask = mask | df['sim_type'].isin(REFERENCE_TYPES)
            df = df[mask]
    elif filter_mode == "Fix α_out":
        alpha_out_values = sorted(df[~df['sim_type'].isin(REFERENCE_TYPES)]['alpha_out'].unique())
        if alpha_out_values:
            sel = st.selectbox("α_out value", options=alpha_out_values, key="local_fix_alpha_out")
            mask = df['alpha_out'] == sel
            if show_refs:
                mask = mask | df['sim_type'].isin(REFERENCE_TYPES)
            df = df[mask]
    elif filter_mode == "α_in = α_out":
        mask = df['alpha_in'] == df['alpha_out']
        if show_refs:
            mask = mask | df['sim_type'].isin(REFERENCE_TYPES)
        df = df[mask]

    # Apply normalization
    if normalize_by != "None":
        ref_type = "PASSIVE" if normalize_by == "Passive" else "ACTIVE_PUR"
        ref_table = compute_reference_table(df_seg)

        if ref_type not in ref_table:
            st.warning(f"No {normalize_by} reference data found. Showing raw values.")
            normalize_by = "None"
        else:
            ref_lookup = ref_table[ref_type]
            # Normalize y_var by reference at same segment_center
            original_y = df[y_var].copy()
            for idx, row in df.iterrows():
                sc = row['segment_center']
                if sc in ref_lookup and y_var in ref_lookup[sc]:
                    ref_val = ref_lookup[sc][y_var]
                    if ref_val != 0 and not np.isnan(ref_val):
                        df.at[idx, y_var] = row[y_var] / ref_val
                    else:
                        df.at[idx, y_var] = np.nan

            # Check if normalization produced all NaN
            if df[y_var].notna().sum() == 0:
                st.warning(f"Normalization by {normalize_by} for {CORR_LABELS.get(y_var, y_var)} produced all NaN (reference values are NaN). Showing raw values.")
                df[y_var] = original_y

    # Drop NaN for plot variables
    plot_df = df.dropna(subset=[x_var, y_var])

    if len(plot_df) == 0:
        st.warning(f"No data for {CORR_LABELS.get(x_var, x_var)} vs {CORR_LABELS.get(y_var, y_var)}")
        return

    # Get unique couples (excluding references)
    couples = sorted(plot_df[~plot_df['sim_type'].isin(REFERENCE_TYPES)]['couple'].unique())

    # Determine max subplots
    max_subplots = MAX_SUBPLOTS_DISJOINT if mode == "disjoint" else MAX_SUBPLOTS_SLIDING

    # Get color range from TASEP data
    df_tasep = plot_df[~plot_df['sim_type'].isin(REFERENCE_TYPES)]
    if color_var in df_tasep.columns and len(df_tasep) > 0:
        cmin = df_tasep[color_var].min() if np.issubdtype(df_tasep[color_var].dtype, np.number) else 0
        cmax = df_tasep[color_var].max() if np.issubdtype(df_tasep[color_var].dtype, np.number) else 1
    else:
        cmin, cmax = 0, 1

    y_label = CORR_LABELS.get(y_var, y_var)
    if normalize_by != "None":
        y_label = f"{y_label} / {normalize_by}"

    if display_mode == "All couples":
        fig = go.Figure()
        legend_shown = set()
        colorbar_shown = False

        # Plot each sim_type
        for sim_type in plot_df['sim_type'].unique():
            style = SIM_TYPE_STYLES.get(sim_type, {"symbol": "circle", "color": "gray", "name": sim_type})
            type_df = plot_df[plot_df['sim_type'] == sim_type]
            is_reference = sim_type in REFERENCE_TYPES

            if is_reference:
                # Fixed color, distinct symbol, single trace
                ref_df = type_df.sort_values(x_var) if link_points else type_df
                marker_size = calc_marker_size(ref_df, size_var, plot_df)
                show_legend = sim_type not in legend_shown
                legend_shown.add(sim_type)

                hover_text = [
                    f"{style['name']}<br>seg={row['segment_center']}<br>{x_var}={row[x_var]:.4f}<br>{y_var}={row[y_var]:.4f}"
                    for _, row in ref_df.iterrows()
                ]

                trace_mode = "markers+lines" if link_points else "markers"
                fig.add_trace(go.Scatter(
                    x=ref_df[x_var], y=ref_df[y_var],
                    mode=trace_mode,
                    marker=dict(
                        size=marker_size if isinstance(marker_size, int) else marker_size.tolist(),
                        symbol=style["symbol"],
                        color=style["color"],
                        line=dict(width=0.5, color="white"),
                        opacity=0.8,
                    ),
                    name=style["name"],
                    hovertext=hover_text, hoverinfo="text",
                    showlegend=show_legend,
                    legendgroup=sim_type,
                ))
            else:
                # Color by colorscale, per couple
                trace_mode = "markers+lines" if link_points else "markers"
                for couple in type_df['couple'].unique():
                    couple_data = type_df[type_df['couple'] == couple]
                    if link_points:
                        couple_data = couple_data.sort_values(x_var)
                    marker_size = calc_marker_size(couple_data, size_var, plot_df)

                    hover_text = [
                        f"TASEP<br>{couple}<br>seg={row['segment_center']}<br>{x_var}={row[x_var]:.4f}<br>{y_var}={row[y_var]:.4f}"
                        for _, row in couple_data.iterrows()
                    ]

                    use_colorscale = color_var in couple_data.columns and np.issubdtype(couple_data[color_var].dtype, np.number)

                    if use_colorscale:
                        show_colorbar = not colorbar_shown
                        if show_colorbar:
                            colorbar_shown = True
                        marker_dict = dict(
                            size=marker_size if isinstance(marker_size, int) else marker_size.tolist(),
                            symbol=style["symbol"],
                            color=couple_data[color_var],
                            colorscale="jet",
                            cmin=cmin, cmax=cmax,
                            showscale=show_colorbar,
                            colorbar=dict(title=CORR_LABELS.get(color_var, color_var), len=0.7, x=1.02, y=0.5),
                            line=dict(width=0.5, color="white"),
                            opacity=0.8,
                        )
                    else:
                        marker_dict = dict(
                            size=marker_size if isinstance(marker_size, int) else marker_size.tolist(),
                            symbol=style["symbol"],
                            color="steelblue",
                            line=dict(width=0.5, color="white"),
                            opacity=0.8,
                        )

                    show_legend = sim_type not in legend_shown
                    legend_shown.add(sim_type)

                    fig.add_trace(go.Scatter(
                        x=couple_data[x_var], y=couple_data[y_var],
                        mode=trace_mode,
                        marker=marker_dict,
                        name=style["name"],
                        hovertext=hover_text, hoverinfo="text",
                        showlegend=show_legend,
                        legendgroup=sim_type,
                    ))

        if log_x:
            fig.update_xaxes(type="log")
        if log_y:
            fig.update_yaxes(type="log")

        fig.update_layout(
            height=600,
            template="plotly_white",
            font=dict(size=14),
            xaxis_title=CORR_LABELS.get(x_var, x_var),
            yaxis_title=y_label,
            legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
            margin=dict(r=120, b=100),
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        # Separate figures mode
        if len(couples) > max_subplots:
            st.warning(f"Showing first {max_subplots} of {len(couples)} couples (cap for {mode} mode).")
            couples = couples[:max_subplots]

        if len(couples) == 0:
            st.warning("No TASEP couples to display.")
            return

        fig = make_subplots(
            rows=1, cols=len(couples),
            shared_yaxes=True,
            subplot_titles=couples,
            horizontal_spacing=0.03,
        )

        legend_shown = set()
        colorbar_shown = False

        for sim_type in plot_df['sim_type'].unique():
            style = SIM_TYPE_STYLES.get(sim_type, {"symbol": "circle", "color": "gray", "name": sim_type})
            is_reference = sim_type in REFERENCE_TYPES
            type_df = plot_df[plot_df['sim_type'] == sim_type]

            if len(type_df) == 0:
                continue

            trace_mode = "markers+lines" if link_points else "markers"
            for i, couple in enumerate(couples):
                if is_reference:
                    couple_data = type_df  # References in ALL subplots
                else:
                    couple_data = type_df[type_df['couple'] == couple]

                if len(couple_data) == 0:
                    continue

                if link_points:
                    couple_data = couple_data.sort_values(x_var)

                marker_size = calc_marker_size(couple_data, size_var, plot_df)

                hover_text = [
                    f"{style['name']}<br>seg={row['segment_center']}<br>{x_var}={row[x_var]:.4f}<br>{y_var}={row[y_var]:.4f}"
                    for _, row in couple_data.iterrows()
                ]

                use_colorscale = (
                    style["color"] is None
                    and color_var in couple_data.columns
                    and np.issubdtype(couple_data[color_var].dtype, np.number)
                )

                if use_colorscale:
                    show_colorbar = (i == len(couples) - 1) and not colorbar_shown
                    if show_colorbar:
                        colorbar_shown = True
                    marker_dict = dict(
                        size=marker_size if isinstance(marker_size, int) else marker_size.tolist(),
                        symbol=style["symbol"],
                        color=couple_data[color_var],
                        colorscale="jet",
                        cmin=cmin, cmax=cmax,
                        showscale=show_colorbar,
                        colorbar=dict(title=CORR_LABELS.get(color_var, color_var), len=0.7, x=1.02, y=0.5),
                        line=dict(width=0.5, color="white"),
                        opacity=0.8,
                    )
                else:
                    marker_dict = dict(
                        size=marker_size if isinstance(marker_size, int) else marker_size.tolist(),
                        symbol=style["symbol"],
                        color=style["color"] or "steelblue",
                        line=dict(width=0.5, color="white"),
                        opacity=0.8,
                    )

                show_legend = sim_type not in legend_shown
                if show_legend:
                    legend_shown.add(sim_type)

                fig.add_trace(
                    go.Scatter(
                        x=couple_data[x_var], y=couple_data[y_var],
                        mode=trace_mode,
                        marker=marker_dict,
                        name=style["name"],
                        hovertext=hover_text, hoverinfo="text",
                        showlegend=show_legend,
                        legendgroup=sim_type,
                    ),
                    row=1, col=i + 1,
                )

        # Update axes
        fig.update_yaxes(title_text=y_label, row=1, col=1)
        for i in range(len(couples)):
            fig.update_xaxes(title_text=CORR_LABELS.get(x_var, x_var), row=1, col=i + 1)
            if log_x:
                fig.update_xaxes(type="log", row=1, col=i + 1)
            if log_y:
                fig.update_yaxes(type="log", row=1, col=i + 1)

        fig.update_layout(
            height=500,
            template="plotly_white",
            font=dict(size=12),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
            margin=dict(r=100),
        )

        st.plotly_chart(fig, use_container_width=True)

    # Statistics
    st.metric("Data points", len(plot_df))
