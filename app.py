#!/usr/bian/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Streamlit app title
st.title("STOIIP Calculator with Monte Carlo Simulation")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")

area_km2 = st.sidebar.slider("Area (km²)", 1.0, 1000.0, 100.0, step=1.0)
area_unc = st.sidebar.slider("Area Uncertainty (±%)", 0.0, 50.0, 10.0, step=1.0)
thickness = st.sidebar.slider("Thickness (ft)", 10.0, 500.0, 100.0, step=10.0)
thick_unc = st.sidebar.slider("Thickness Uncertainty (±%)", 0.0, 50.0, 15.0, step=1.0)
porosity = st.sidebar.slider("Porosity (fraction)", 0.05, 0.4, 0.2, step=0.01)
por_unc = st.sidebar.slider("Porosity Uncertainty (±%)", 0.0, 50.0, 20.0, step=1.0)
oil_saturation = st.sidebar.slider("Oil Saturation (fraction)", 0.2, 0.95, 0.7, step=0.01)
sat_unc = st.sidebar.slider("Oil Saturation Uncertainty (±%)", 0.0, 50.0, 10.0, step=1.0)
fvf = st.sidebar.slider("Formation Volume Factor", 1.0, 2.0, 1.2, step=0.05)
fvf_unc = st.sidebar.slider("FVF Uncertainty (±%)", 0.0, 50.0, 5.0, step=1.0)
ntg = st.sidebar.slider("Net to Gross (fraction)", 0.1, 1.0, 0.8, step=0.01)
ntg_unc = st.sidebar.slider("NTG Uncertainty (±%)", 0.0, 50.0, 10.0, step=1.0)
iterations = st.sidebar.number_input("Iterations", min_value=100, max_value=5000, value=1000, step=100)

# Conversion factor: 1,917,134 barrels per km²-ft
barrel_conversion = 1917134

# Monte Carlo simulation function
def run_monte_carlo(area_km2, area_unc, thickness, thick_unc, porosity, por_unc,
                    oil_saturation, sat_unc, fvf, fvf_unc, ntg, ntg_unc, iterations):
    area_samples = np.random.normal(area_km2, area_km2 * area_unc/100, iterations)
    thick_samples = np.random.normal(thickness, thickness * thick_unc/100, iterations)
    por_samples = np.random.normal(porosity, porosity * por_unc/100, iterations)
    sat_samples = np.random.normal(oil_saturation, oil_saturation * sat_unc/100, iterations)
    fvf_samples = np.random.normal(fvf, fvf * fvf_unc/100, iterations)
    ntg_samples = np.random.normal(ntg, ntg * ntg_unc/100, iterations)
    
    # Clip values
    area_samples = np.clip(area_samples, 1, None)
    thick_samples = np.clip(thick_samples, 10, None)
    por_samples = np.clip(por_samples, 0.05, 0.4)
    sat_samples = np.clip(sat_samples, 0.2, 0.95)
    fvf_samples = np.clip(fvf_samples, 1.0, 2.0)
    ntg_samples = np.clip(ntg_samples, 0.1, 1.0)
    
    # STOIIP calculation
    stoiip_samples = (area_samples * thick_samples * por_samples * sat_samples * 
                      ntg_samples * barrel_conversion) / fvf_samples
    stoiip_bstb = stoiip_samples / 1_000_000_000
    
    # Calculate weights (relative contribution based on standard deviation)
    weights = {
        'Area': np.std(area_samples * thick_samples.mean() * por_samples.mean() * sat_samples.mean() * ntg_samples.mean() * barrel_conversion / fvf_samples.mean()),
        'Thickness': np.std(thick_samples * area_samples.mean() * por_samples.mean() * sat_samples.mean() * ntg_samples.mean() * barrel_conversion / fvf_samples.mean()),
        'Porosity': np.std(por_samples * area_samples.mean() * thick_samples.mean() * sat_samples.mean() * ntg_samples.mean() * barrel_conversion / fvf_samples.mean()),
        'Oil Sat': np.std(sat_samples * area_samples.mean() * thick_samples.mean() * por_samples.mean() * ntg_samples.mean() * barrel_conversion / fvf_samples.mean()),
        'NTG': np.std(ntg_samples * area_samples.mean() * thick_samples.mean() * por_samples.mean() * sat_samples.mean() * barrel_conversion / fvf_samples.mean()),
        'FVF': np.std(fvf_samples * area_samples.mean() * thick_samples.mean() * por_samples.mean() * sat_samples.mean() * ntg_samples.mean() * barrel_conversion / stoiip_samples)
    }
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    return stoiip_bstb, normalized_weights

# Run simulation
stoiip_samples_bstb, weights = run_monte_carlo(area_km2, area_unc, thickness, thick_unc,
                                              porosity, por_unc, oil_saturation, sat_unc,
                                              fvf, fvf_unc, ntg, ntg_unc, iterations)
p10 = np.percentile(stoiip_samples_bstb, 10)
p50 = np.percentile(stoiip_samples_bstb, 50)
p90 = np.percentile(stoiip_samples_bstb, 90)

# Create three plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# STOIIP Cases Bar Chart
bars = ax1.bar(['P10 (Low)', 'P50 (Base)', 'P90 (High)'], [p10, p50, p90], color=['red', 'blue', 'green'])
ax1.set_ylabel('Volume (BSTB)')
ax1.set_title(f'STOIIP Cases ({iterations} iterations)')
ax1.grid(True, alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:,.3f}', ha='center', va='bottom', fontsize=10)

# STOIIP Distribution Histogram
ax2.hist(stoiip_samples_bstb, bins=50, color='gray', alpha=0.7, density=True)
ax2.set_xlabel('Volume (BSTB)')
ax2.set_ylabel('Probability Density')
ax2.set_title('STOIIP Distribution')
ax2.axvline(p10, color='red', linestyle='--', label=f'P10: {p10:.3f}')
ax2.axvline(p50, color='blue', linestyle='--', label=f'P50: {p50:.3f}')
ax2.axvline(p90, color='green', linestyle='--', label=f'P90: {p90:.3f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Variable Weights Bar Chart
variables = list(weights.keys())
weight_values = list(weights.values())
bars = ax3.bar(variables, weight_values, color='purple')
ax3.set_ylabel('Relative Weight')
ax3.set_title('Variable Weights in STOIIP')
ax3.set_ylim(0, max(weight_values) * 1.2 if max(weight_values) > 0 else 1)
ax3.grid(True, alpha=0.3)
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10)
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

# Adjust layout and display
plt.tight_layout()
st.pyplot(fig)