"""
dashboard/constants.py
Shared constants and configuration for the Active Polymer Dashboard.
"""

# Marker styles for each simulation type
# color=None means "use colorscale"; fixed color string for references
SIM_TYPE_STYLES = {
    "TASEP": {"symbol": "circle", "color": None, "name": "TASEP"},
    "PASSIVE": {"symbol": "diamond", "color": "gray", "name": "Passive (Pe=0)", "reference": True},
    "ACTIVE_PUR": {"symbol": "square", "color": "red", "name": "Active pur", "reference": True},
}

REFERENCE_TYPES = ["PASSIVE", "ACTIVE_PUR"]

# Global Correlation: X/Y axis selectors (chain-level observables)
GLOBAL_OBSERVABLES = [
    'Rg_mean', 'Rg_std', 'Rg_cv', 'D_lin_fit_long',
    'rho_mean', 'J_mean', 'sigma_rho_mean',
]

# Heatmaps tab: observable selector
HEATMAP_METRICS = {
    'Rg_mean': 'Radius of Gyration (mean)',
    'Rg_std': 'Radius of Gyration (std)',
    'Rg_cv': 'Radius of Gyration (CV = std/mean)',
    'D_lin_fit_long': 'Diffusion Coefficient (linear fit)',
    'rho_mean': 'TASEP Density ρ (mean)',
    'J_mean': 'TASEP Current J (mean)',
    'sigma_rho_mean': 'TASEP Density Fluctuations σ_ρ (mean)',
}

# Normalization options for heatmaps and global correlation
NORMALIZE_OPTIONS = ["None", "Passive", "Active pur"]

# Local Correlation: X/Y axis selectors (segment-level)
CORR_LOCAL_GROUPS = [
    ("Position", ["segment_center"]),
    ("TASEP profile", ["rho_tasep", "J_tasep", "sigma_tasep"]),
    ("Local (segment)", ["Rg_local", "D_local"]),
    ("MSD short-time", ["MSD_t1", "MSD_t4", "MSD_t16", "MSD_t65", "MSD_t262", "MSD_t524"]),
    ("MSD long-time", ["MSD_t1k", "MSD_t5k", "MSD_t10k", "MSD_t50k", "MSD_t100k", "MSD_t300k"]),
    ("Fluctuations", ["Rg_local_std"]),
]

# Local Correlation: Color/Size selectors (global + system)
CORR_GLOBAL_GROUPS = [
    ("System", ["alpha_in", "alpha_out", "alpha_ratio", "sim_type"]),
    ("Global (chain)", ["Rg_chain", "D_eff"]),
    ("TASEP bulk theory", ["rho_mf", "J_mf_exact"]),
]

# Corruption tab: metric selector
CORRUPTION_METRICS = {
    'blocks_lost_pct': 'Percentage of Blocks Lost (%)',
    'frames_lost_pct': 'Percentage of Frames Lost (%)',
    'blocks_eliminated': 'Number of Blocks Eliminated',
    'blocks_original': 'Total Blocks in Trajectory',
    'frames_original': 'Total Frames in Trajectory',
}

# Display labels for all column names (used in format_func)
CORR_LABELS = {
    # System parameters
    'alpha_in': 'α_in (entry rate)',
    'alpha_out': 'α_out (exit rate)',
    'alpha_product': 'α_in × α_out',
    'alpha_ratio': 'α_in / α_out',
    'alpha_ratio_inv': 'α_out / α_in',
    'sim_type': 'Simulation type',
    # Global (chain-level)
    'Rg_mean': 'Rg (mean)',
    'Rg_std': 'Rg (std)',
    'Rg_cv': 'Rg (CV = σ/⟨Rg⟩)',
    'D_lin_fit_long': 'D (linear fit)',
    'D_long_plateau': 'D (plateau)',
    'D_log_fit_long': 'D (log fit)',
    'rho_mean': 'ρ (TASEP density)',
    'J_mean': 'J (TASEP current)',
    'sigma_rho_mean': 'σ_ρ (density fluctuations)',
    'blocks_lost_pct': 'Blocks lost (%)',
    'frames_lost_pct': 'Frames lost (%)',
    # Segment-level
    'segment_center': 'Segment position',
    'Rg_local': 'Rg_local (segment)',
    'Rg_local_std': 'σ(Rg_local)',
    'D_local': 'D_local (segment)',
    'rho_local': 'ρ_local (measured)',
    'rho_local_std': 'σ(ρ_local)',
    'rho_tasep': 'ρ_TASEP (profile)',
    'sigma_tasep': 'σ_TASEP (profile)',
    'J_tasep': 'J_TASEP (current profile)',
    'MSD_t1': 'MSD(t≈1)',
    'MSD_t4': 'MSD(t≈4)',
    'MSD_t16': 'MSD(t≈16)',
    'MSD_t65': 'MSD(t≈65)',
    'MSD_t262': 'MSD(t≈262)',
    'MSD_t524': 'MSD(t≈524)',
    'MSD_t1k': 'MSD(t=1k)',
    'MSD_t5k': 'MSD(t=5k)',
    'MSD_t10k': 'MSD(t=10k)',
    'MSD_t50k': 'MSD(t=50k)',
    'MSD_t100k': 'MSD(t=100k)',
    'MSD_t300k': 'MSD(t=300k)',
    # Global used in local context
    'Rg_chain': 'Rg_chain (global)',
    'D_eff': 'D_chain (COM)',
    'rho_mf': 'ρ_MF (mean-field)',
    'J_mf_exact': 'J_MF (exact)',
    'rho_exact': 'ρ_exact',
    'couple': '(α_in, α_out) couple',
}

# Subplot caps
MAX_SUBPLOTS_DISJOINT = 30
MAX_SUBPLOTS_SLIDING = 10
