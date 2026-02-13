"""
dashboard/data_loading.py
Data loading functions for the Active Polymer Dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from constants import REFERENCE_TYPES


@st.cache_data
def load_physics_data():
    """Load compiled global simulation data, enriched with TASEP bulk averages."""
    data_file = Path(__file__).parent / "data_compiled.csv"
    if not data_file.exists():
        st.error(f"Data file not found: {data_file}")
        st.stop()
    df = pd.read_csv(data_file)

    # Compute derived columns
    df['alpha_product'] = df['alpha_in'] * df['alpha_out']
    with np.errstate(divide='ignore', invalid='ignore'):
        df['alpha_ratio'] = np.where(df['alpha_out'] != 0, df['alpha_in'] / df['alpha_out'], np.nan)
        df['alpha_ratio_inv'] = np.where(df['alpha_in'] != 0, df['alpha_out'] / df['alpha_in'], np.nan)

    # Enrich with TASEP bulk averages from segment data (rho, J, sigma)
    seg_file = Path(__file__).parent / "data_compiled_segments_disjoint.csv"
    if seg_file.exists():
        df_seg = pd.read_csv(seg_file)
        # Average TASEP profiles per simulation (group by Pe, alpha_in, alpha_out, kappa, pol)
        seg_agg = df_seg.groupby(['Pe', 'alpha_in', 'alpha_out', 'kappa', 'pol']).agg(
            rho_mean=('rho_tasep', 'mean'),
            J_mean=('J_tasep', 'mean'),
            sigma_rho_mean=('sigma_tasep', 'mean'),
        ).reset_index()
        # Merge — pol types must match (global CSV has string pol)
        seg_agg['pol'] = seg_agg['pol'].astype(str)
        df['pol'] = df['pol'].astype(str)
        df = df.merge(seg_agg, on=['Pe', 'alpha_in', 'alpha_out', 'kappa', 'pol'], how='left')

    return df


@st.cache_data
def load_corruption_data():
    """Load corruption data from CSV report."""
    csv_path = Path(__file__).parent / "corruption_report.csv"
    if not csv_path.exists():
        st.error("No corruption data found. Run process_vx_pwrsq.sh first.")
        st.stop()

    df = pd.read_csv(csv_path)

    # Parse eliminated block numbers from string to list
    def parse_blocks(block_str):
        if pd.isna(block_str) or block_str == '':
            return []
        return [int(b) for b in str(block_str).split(',')]

    df['eliminated_blocks_list'] = df['eliminated_block_numbers'].apply(parse_blocks)

    # Ensure parameter columns exist and have correct types
    if 'Pe' not in df.columns:
        df['Pe'] = df['simulation_dir'].str.extract(r'Pe_([\d.]+)').astype(float)
    else:
        df['Pe'] = df['Pe'].astype(float)
    if 'alpha_in' not in df.columns:
        df['alpha_in'] = df['simulation_dir'].str.extract(r'alpha_in_([\d.]+)').astype(float)
    else:
        df['alpha_in'] = df['alpha_in'].astype(float)
    if 'alpha_out' not in df.columns:
        df['alpha_out'] = df['simulation_dir'].str.extract(r'alpha_out_([\d.]+)').astype(float)
    else:
        df['alpha_out'] = df['alpha_out'].astype(float)
    if 'kappa' not in df.columns:
        df['kappa'] = df['simulation_dir'].str.extract(r'kappa_([\d.]+)').astype(float)
    else:
        df['kappa'] = df['kappa'].astype(float)
    if 'pol' not in df.columns:
        pol_match = df['simulation_dir'].str.extract(r'pol_(\w+)')
        df['pol'] = pd.to_numeric(pol_match[0], errors='coerce').fillna(0).astype(int)
    else:
        df['pol'] = pd.to_numeric(df['pol'], errors='coerce').fillna(0).astype(int)

    return df


@st.cache_data
def load_segment_data(mode):
    """Load segment-level compiled data.

    Args:
        mode: "disjoint" or "sliding"
    """
    csv_path = Path(__file__).parent / f"data_compiled_segments_{mode}.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def get_global_reference_values(df_physics):
    """Extract reference simulation values from global physics data.

    References are identified by alpha_in == 0 AND alpha_out == 0.
    Returns:
        dict: {"PASSIVE": {col: value}, "ACTIVE_PUR": {col: value}}
    """
    ref_values = {}
    refs = df_physics[(df_physics['alpha_in'] == 0) & (df_physics['alpha_out'] == 0)]
    for _, row in refs.iterrows():
        if row['Pe'] == 0.0:
            ref_values["PASSIVE"] = row.to_dict()
        else:
            ref_values["ACTIVE_PUR"] = row.to_dict()
    return ref_values


def compute_reference_table(df_seg):
    """Build lookup table for normalization by reference simulations.

    Returns:
        dict: {ref_type: {segment_center: {col: value}}}
    """
    ref_table = {}
    for ref_type in REFERENCE_TYPES:
        ref_rows = df_seg[df_seg['sim_type'] == ref_type]
        if len(ref_rows) > 0:
            ref_table[ref_type] = ref_rows.set_index('segment_center').to_dict('index')
    return ref_table
