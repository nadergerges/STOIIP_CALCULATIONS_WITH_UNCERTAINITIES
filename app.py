import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# Streamlit app title
st.title("STOIIP Calculator with Monte Carlo Simulation (Altair)")

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
    # Generate random samples
    area_samples = np.random.normal(area_km2, area_km2 * area_unc/100, iterations)
    thick_samples = np.random.normal(thickness, thickness * thick_unc/100, iterations)
    por_samples = np.random.normal(porosity, porosity * por_unc/100, iterations)
    sat_samples = np.random.normal(oil_saturation, oil_saturation * sat_unc/100, iterations)
    fvf_samples = np.random.normal(fvf, fvf * fvf_unc/100, iterations)
    ntg_samples = np.random.normal(ntg, ntg * ntg_unc/100, iterations)
    
    # Clip values so they don't go below/above physically reasonable limits
    area_samples = np.clip(area_samples, 1, None)
    thick_samples = np.clip(thick_samples, 10, None)
    por_samples = np.clip(por_samples, 0.05, 0.4)
    sat_samples = np.clip(sat_samples, 0.2, 0.95)
    fvf_samples = np.clip(fvf_samples, 1.0, 2.0)
    ntg_samples = np.clip(ntg_samples, 0.1, 1.0)
    
    # STOIIP calculation
    stoiip_samples = (area_samples * thick_samples * por_samples * 
                      sat_samples * ntg_samples * barrel_conversion) / fvf_samples
    stoiip_bstb = stoiip_samples / 1_000_000_000  # Convert to billions of STB
    
    # Calculate weights (relative contribution based on standard deviation approach)
    weights = {
        'Area': np.std(area_samples * thick_samples.mean() * por_samples.mean() * 
                       sat_samples.mean() * ntg_samples.mean() * barrel_conversion / fvf_samples.mean()),
        'Thickness': np.std(thick_samples * area_samples.mean() * por_samples.mean() * 
                            sat_samples.mean() * ntg_samples.mean() * barrel_conversion / fvf_samples.mean()),
        'Porosity': np.std(por_samples * area_samples.mean() * thick_samples.mean() * 
                           sat_samples.mean() * ntg_samples.mean() * barrel_conversion / fvf_samples.mean()),
        'Oil Sat': np.std(sat_samples * area_samples.mean() * thick_samples.mean() * 
                          por_samples.mean() * ntg_samples.mean() * barrel_conversion / fvf_samples.mean()),
        'NTG': np.std(ntg_samples * area_samples.mean() * thick_samples.mean() * 
                      por_samples.mean() * sat_samples.mean() * barrel_conversion / fvf_samples.mean()),
        'FVF': np.std(fvf_samples * area_samples.mean() * thick_samples.mean() * 
                      por_samples.mean() * sat_samples.mean() * ntg_samples.mean() * barrel_conversion / stoiip_samples)
    }
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    return stoiip_bstb, normalized_weights

# Run the Monte Carlo simulation
stoiip_samples_bstb, weights = run_monte_carlo(
    area_km2, area_unc, thickness, thick_unc,
    porosity, por_unc, oil_saturation, sat_unc,
    fvf, fvf_unc, ntg, ntg_unc, iterations
)

# Calculate P10, P50, and P90
p10 = np.percentile(stoiip_samples_bstb, 10)
p50 = np.percentile(stoiip_samples_bstb, 50)
p90 = np.percentile(stoiip_samples_bstb, 90)

# --- 1. Bar Chart of P10, P50, and P90 ---
cases_df = pd.DataFrame({
    'Case': ['P10 (Low)', 'P50 (Base)', 'P90 (High)'],
    'Volume (BSTB)': [p10, p50, p90]
})

cases_chart = (
    alt.Chart(cases_df)
    .mark_bar()
    .encode(
        x=alt.X('Case:N', title='', sort=None),
        y=alt.Y('Volume (BSTB):Q', title='Volume (BSTB')),
        tooltip=['Case:N', 'Volume (BSTB):Q']
    )
    .properties(title=f'STOIIP Cases ({iterations} iterations)')
)

# --- 2. Histogram of the STOIIP Distribution with lines for P10, P50, P90 ---
dist_df = pd.DataFrame({'STOIIP (BSTB)': stoiip_samples_bstb})

# Base histogram
hist = (
    alt.Chart(dist_df)
    .mark_bar(opacity=0.7)
    .encode(
        alt.X('STOIIP (BSTB):Q', bin=alt.Bin(maxbins=50), title='STOIIP (BSTB)'),
        alt.Y('count()', title='Count')
    )
    .properties(title='STOIIP Distribution')
)

# Vertical lines for P10, P50, P90 (layer on top of the histogram)
rule_p10 = alt.Chart(pd.DataFrame({'value': [p10]})).mark_rule(color='red').encode(x='value:Q')
rule_p50 = alt.Chart(pd.DataFrame({'value': [p50]})).mark_rule(color='blue').encode(x='value:Q')
rule_p90 = alt.Chart(pd.DataFrame({'value': [p90]})).mark_rule(color='green').encode(x='value:Q')

dist_chart = alt.layer(hist, rule_p10, rule_p50, rule_p90).interactive()

# --- 3. Bar Chart of variable weights ---
weights_df = pd.DataFrame({
    'Variable': list(weights.keys()),
    'Weight': list(weights.values())
})

weights_chart = (
    alt.Chart(weights_df)
    .mark_bar()
    .encode(
        x=alt.X('Variable:N', sort=None, title='Variable'),
        y=alt.Y('Weight:Q', title='Relative Weight'),
        tooltip=['Variable:N', 'Weight:Q']
    )
    .properties(title='Variable Weights in STOIIP')
)

# Display charts in Streamlit
st.altair_chart(cases_chart, use_container_width=True)
st.altair_chart(dist_chart, use_container_width=True)
st.altair_chart(weights_chart, use_container_width=True)

# Display the numeric values for P10, P50, P90
st.write(f"**P10 (Low):** {p10:,.3f} BSTB")
st.write(f"**P50 (Base):** {p50:,.3f} BSTB")
st.write(f"**P90 (High):** {p90:,.3f} BSTB")
