"""
dashboard/utils.py
Shared utility functions for the Active Polymer Dashboard.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import re


def get_block_number(timestep, size_block=20):
    """Calculate block number for a timestep in power-of-2 sequence."""
    block_size = 1 << size_block  # 2^20 = 1048576
    return (timestep - 1) // block_size


def format_timestep_millions(timestep):
    """Format timestep as millions (e.g., 268.4M)."""
    return f"{timestep / 1_000_000:.1f}M"


def calculate_statistics(df):
    """Calculate summary statistics from corruption data."""
    stats = {
        'total_simulations': len(df),
        'sims_with_corruption': (df['blocks_eliminated'] > 0).sum(),
        'corruption_rate': 100 * (df['blocks_eliminated'] > 0).sum() / len(df),
        'avg_blocks_lost_pct': df['blocks_lost_pct'].mean(),
        'avg_frames_lost_pct': df['frames_lost_pct'].mean(),
        'max_blocks_lost_pct': df['blocks_lost_pct'].max(),
        'max_frames_lost_pct': df['frames_lost_pct'].max(),
        'worst_sim': df.loc[df['blocks_lost_pct'].idxmax(), 'simulation_dir'],
        'worst_sim_loss': df['blocks_lost_pct'].max()
    }
    return stats


def create_heatmap(df, metric, title, colorscale='RdYlGn_r'):
    """Create an interactive heatmap for corruption metrics."""
    # Create pivot table
    pivot_data = df.pivot_table(
        values=metric,
        index='alpha_out',
        columns='alpha_in',
        aggfunc='mean'
    )

    # Calculate dimensions for square pixels
    n_rows = len(pivot_data.index)
    n_cols = len(pivot_data.columns)
    pixel_size = 60  # Size in pixels per cell

    # Determine text format based on metric type
    if metric in ['blocks_original', 'blocks_valid', 'blocks_eliminated',
                  'frames_original', 'frames_written', 'frames_lost',
                  'duplicate_timesteps', 'angles_restart_cases']:
        # Integer values
        text_template = '%{text:.0f}'
        hover_format = ':.0f'
    elif metric in ['timestep_millions', 'valid_timestep_millions']:
        # Timesteps in millions - show one decimal
        text_template = '%{text:.1f}'
        hover_format = ':.1f'
    else:
        # Percentages and other floats
        text_template = '%{text:.2f}'
        hover_format = ':.2f'

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale=colorscale,
        text=pivot_data.values,
        texttemplate=text_template,
        textfont={"size": 10},
        colorbar=dict(title=metric.replace('_', ' ').title()),
        hovertemplate='α_in: %{x}<br>α_out: %{y}<br>' +
                      metric.replace('_', ' ').title() + f': %{{z{hover_format}}}<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='α_in (Entry Rate)',
        yaxis_title='α_out (Exit Rate)',
        width=n_cols * pixel_size + 200,   # +200 for colorbar and margins
        height=n_rows * pixel_size + 150,  # +150 for title and margins
        font=dict(size=12),
        xaxis=dict(
            scaleanchor="y",  # Force 1:1 aspect ratio
            scaleratio=1,
            constrain='domain'
        ),
        yaxis=dict(
            constrain='domain'
        )
    )

    return fig


def create_block_timeline(sim_data, eliminated_blocks):
    """Create a timeline visualization of valid/eliminated blocks."""
    blocks_original = sim_data['blocks_original']

    # Create array of block statuses (1=valid, 0=eliminated)
    block_status = np.ones(blocks_original, dtype=int)
    for block_num in eliminated_blocks:
        if block_num < blocks_original:
            block_status[block_num] = 0

    # Create colors array
    colors = ['#28a745' if status == 1 else '#dc3545' for status in block_status]

    # Create bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=np.arange(blocks_original),
        y=np.ones(blocks_original),
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        hovertemplate='Block %{x}<br>Status: %{customdata}<extra></extra>',
        customdata=['Valid' if status == 1 else 'Eliminated' for status in block_status],
        showlegend=False
    ))

    # Add annotations for consecutive corruption regions
    in_corruption = False
    corruption_start = None

    for i, status in enumerate(block_status):
        if status == 0 and not in_corruption:
            corruption_start = i
            in_corruption = True
        elif status == 1 and in_corruption:
            # End of corruption region
            corruption_end = i - 1
            if corruption_end - corruption_start > 5:  # Only annotate regions > 5 blocks
                fig.add_annotation(
                    x=(corruption_start + corruption_end) / 2,
                    y=0.5,
                    text=f"Corrupted<br>{corruption_start}-{corruption_end}",
                    showarrow=False,
                    font=dict(size=9, color='white'),
                    bgcolor='rgba(220, 53, 69, 0.8)',
                    borderpad=2
                )
            in_corruption = False

    # Handle case where corruption extends to end
    if in_corruption:
        corruption_end = blocks_original - 1
        if corruption_end - corruption_start > 5:
            fig.add_annotation(
                x=(corruption_start + corruption_end) / 2,
                y=0.5,
                text=f"Corrupted<br>{corruption_start}-{corruption_end}",
                showarrow=False,
                font=dict(size=9, color='white'),
                bgcolor='rgba(220, 53, 69, 0.8)',
                borderpad=2
            )

    fig.update_layout(
        title="Block Timeline (Green=Valid, Red=Eliminated)",
        xaxis_title="Block Number",
        yaxis_title="",
        yaxis=dict(showticklabels=False, range=[0, 1.2]),
        height=200,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='white'
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    return fig


def find_simulation_figures(sim_name, base_path):
    """Find all relevant figures for a simulation."""
    figures = []

    # Extract parameters from simulation name for pattern matching (including pol)
    match = re.match(r'(Pe_[\d.]+_alpha_in_[\d.]+_alpha_out_[\d.]+_kappa_[\d.]+_pol_[\w]+)', sim_name)
    if not match:
        return figures

    base_pattern = match.group(1)

    # Rg figures aggregate all replicates, so their filenames use "pol_" without a number.
    # Extract a pattern without the _pol_X suffix for matching Rg files.
    match_no_pol = re.match(r'(Pe_[\d.]+_alpha_in_[\d.]+_alpha_out_[\d.]+_kappa_[\d.]+)', sim_name)
    base_pattern_no_pol = match_no_pol.group(1) if match_no_pol else base_pattern

    # Define figure locations and their display names
    figure_locations = [
        ('FIGURES/RG', f'DATA_RG_{base_pattern_no_pol}_pol_*_rg.png', 'Radius of Gyration (Linear)', ['_log.png']),
        ('FIGURES/RG', f'DATA_RG_{base_pattern_no_pol}_pol_*_rg_log.png', 'Radius of Gyration (Log)', []),
        ('FIGURES_PROV/chain_segments', f'{base_pattern}*_chain_segments.png', 'Chain Segments', []),
        ('FIGURES_PROV/COM', f'{base_pattern}*_msd.png.png', 'MSD (COM)', []),
        ('FIGURES_PROV/target_monomer', f'{base_pattern}*_msd.png.png', 'MSD (Target Monomer)', []),
    ]

    for dir_path, pattern, display_name, exclude_patterns in figure_locations:
        full_dir = base_path / dir_path
        if full_dir.exists():
            for fig_path in full_dir.glob(pattern):
                # Check exclusions
                should_exclude = any(excl in str(fig_path) for excl in exclude_patterns)
                if not should_exclude:
                    figures.append({
                        'path': fig_path,
                        'name': display_name,
                        'filename': fig_path.name
                    })

    return figures


def display_simulation_figures(sim_data):
    """Display figures associated with the simulation."""
    st.markdown("---")
    st.markdown("#### Analysis Figures")

    base_path = Path(__file__).parent.parent
    sim_name = sim_data['simulation_dir']

    figures = find_simulation_figures(sim_name, base_path)

    if not figures:
        st.info("No figures found for this simulation.")
        return

    st.markdown(f"Found **{len(figures)}** figures for this simulation:")

    # Display figures in a grid (2 columns)
    cols = st.columns(2)

    for i, fig_info in enumerate(figures):
        col = cols[i % 2]
        with col:
            st.markdown(f"**{fig_info['name']}**")
            try:
                st.image(str(fig_info['path']), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not load: {fig_info['filename']}")
            st.caption(fig_info['filename'])


def display_simulation_details(df, alpha_in_selected, alpha_out_selected):
    """Display detailed information for a selected simulation."""
    # Find the simulation matching the selected parameters
    sim = df[(df['alpha_in'] == alpha_in_selected) &
             (df['alpha_out'] == alpha_out_selected)]

    if len(sim) == 0:
        st.warning(f"No simulation found for α_in={alpha_in_selected}, α_out={alpha_out_selected}")
        return

    sim_data = sim.iloc[0]

    st.markdown("---")
    st.markdown(f"### Simulation Details: `{sim_data['simulation_dir']}`")

    # Display key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Blocks Original", int(sim_data['blocks_original']))
        st.metric("Blocks Valid", int(sim_data['blocks_valid']))

    with col2:
        st.metric("Blocks Eliminated", int(sim_data['blocks_eliminated']))
        st.metric("Blocks Lost %", f"{sim_data['blocks_lost_pct']:.2f}%")

    with col3:
        st.metric("Frames Original", int(sim_data['frames_original']))
        st.metric("Frames Written", int(sim_data['frames_written']))

    with col4:
        st.metric("Frames Lost", int(sim_data['frames_lost']))
        st.metric("Frames Lost %", f"{sim_data['frames_lost_pct']:.2f}%")

    # Timestep information
    st.markdown("#### Timestep Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        last_ts = sim_data.get('last_timestep', 0)
        st.metric("Last Timestep", f"{last_ts:,.0f}" if last_ts else "N/A")
    with col2:
        valid_ts = sim_data.get('valid_timestep', 0)
        st.metric("Valid Timestep", f"{valid_ts:,.0f}" if valid_ts else "N/A")
    with col3:
        if last_ts and valid_ts:
            st.metric("Timesteps Lost", f"{last_ts - valid_ts:,.0f}")

    # Display block timeline
    st.markdown("#### Block Timeline")
    eliminated_blocks = sim_data['eliminated_blocks_list']

    if len(eliminated_blocks) == 0:
        st.markdown('<div class="success-box">No corruption detected in this simulation!</div>',
                   unsafe_allow_html=True)
    else:
        fig_timeline = create_block_timeline(sim_data, eliminated_blocks)
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Display eliminated blocks details
        st.markdown("#### Eliminated Blocks")

        # Identify consecutive regions
        if len(eliminated_blocks) > 0:
            eliminated_sorted = sorted(eliminated_blocks)
            regions = []
            current_region = [eliminated_sorted[0]]

            for i in range(1, len(eliminated_sorted)):
                if eliminated_sorted[i] == current_region[-1] + 1:
                    current_region.append(eliminated_sorted[i])
                else:
                    regions.append(current_region)
                    current_region = [eliminated_sorted[i]]
            regions.append(current_region)

            # Display regions
            st.markdown("**Corruption Regions:**")
            for i, region in enumerate(regions, 1):
                if len(region) == 1:
                    st.markdown(f"- Region {i}: Block {region[0]} (isolated)")
                else:
                    st.markdown(f"- Region {i}: Blocks {region[0]}-{region[-1]} ({len(region)} consecutive blocks)")

        # Detailed list (collapsible)
        with st.expander("Show all eliminated block numbers"):
            st.write(eliminated_blocks)

    # Timestamp
    if 'timestamp' in sim_data.index:
        st.caption(f"Processing timestamp: {sim_data['timestamp']}")

    # Display associated figures
    display_simulation_figures(sim_data)

    # Dry-run analysis for continue_from_dump
    st.markdown("---")
    st.markdown("#### Continue From Dump Preview")
    st.markdown("This section shows how `--continue_from_dump` would handle this simulation.")

    valid_ts = sim_data.get('valid_timestep', 0)
    last_ts = sim_data.get('last_timestep', 0)

    if valid_ts and last_ts:
        st.markdown("**Recovery Analysis:**")

        if last_ts > valid_ts:
            timesteps_lost = last_ts - valid_ts
            st.warning(f"""
            **Mismatch detected**: dump has {timesteps_lost:,} timesteps beyond angles.dat

            - Dump last timestep: **{last_ts:,}** ({last_ts/1e6:.1f}M)
            - Angles last timestep: **{valid_ts:,}** ({valid_ts/1e6:.1f}M)

            `continue_from_dump` would:
            1. Search dump.lammpstrj for frame at timestep {valid_ts:,}
            2. Synchronize to this timestep
            3. Continue simulation from this point
            """)

            # Target timesteps input
            target_ts = st.number_input(
                "Target timesteps (millions)",
                min_value=float(valid_ts / 1e6),
                max_value=1000.0,
                value=float(last_ts / 1e6 + 10),
                step=10.0,
                help="Enter the target number of timesteps in millions",
                key="corrupt_target_ts"
            )

            target_ts_actual = int(target_ts * 1e6)
            steps_to_run = target_ts_actual - valid_ts

            st.info(f"""
            **Dry-run command:**
            ```bash
            python3 src/main.py --sampling powerseq --continue_from_dump \\
                --Pe {sim_data.get('Pe', 10.0)} \\
                --alpha_in {sim_data.get('alpha_in', 0.5)} \\
                --alpha_out {sim_data.get('alpha_out', 0.5)} \\
                --kappa {sim_data.get('kappa', 1.0)} \\
                --total_md_steps {target_ts_actual} \\
                --tasep_time_unit 10000 --seed 1234 --pol 1 \\
                --output_dir {sim_data['simulation_dir']}
            ```

            This would run **{steps_to_run:,}** additional timesteps ({steps_to_run/1e6:.1f}M)
            """)
        else:
            st.success(f"""
            **No mismatch**: dump and angles.dat are synchronized at {valid_ts:,} ({valid_ts/1e6:.1f}M)

            `continue_from_dump` would simply continue from the last timestep.
            """)
    else:
        st.info("Timestep information not available for dry-run analysis.")


def calc_marker_size(data_df, size_col, ref_df):
    """Calculate marker sizes with linear scaling [5, 20]."""
    if size_col != "None" and size_col in data_df.columns:
        size_values = data_df[size_col]
        size_min, size_max = ref_df[size_col].min(), ref_df[size_col].max()
        if size_max > size_min:
            return 5 + 15 * (size_values - size_min) / (size_max - size_min)
    return 10
