import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

# IMPORTANT: Ensure Plotly is installed: pip install plotly
import plotly.graph_objects as go

import math

# Streamlit app title
st.title("STOIIP Calculator (Acres) with Monte Carlo Simulation (Altair)")

# Display the STOIIP equation below the main header
st.latex(r"""
\text{STOIIP (STB)} = 7758 \times A(\text{acres}) \times h(\text{ft}) \times \phi \times S_o \times \frac{\text{NTG}}{B_o}
""")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")

area_acres = st.sidebar.slider("Area (acres)", 1.0, 1_000_000.0, 10000.0, step=1000.0)
area_unc = st.sidebar.slider("Area Uncertainty (±%)", 0.0, 50.0, 10.0, step=1.0)
thickness = st.sidebar.slider("Thickness (ft)", 1.0, 500.0, 100.0, step=1.0)
thick_unc = st.sidebar.slider("Thickness Uncertainty (±%)", 0.0, 50.0, 15.0, step=1.0)
porosity = st.sidebar.slider("Porosity (fraction)", 0.05, 0.4, 0.2, step=0.01)
por_unc = st.sidebar.slider("Porosity Uncertainty (±%)", 0.0, 50.0, 20.0, step=1.0)
oil_saturation = st.sidebar.slider("Oil Saturation (fraction)", 0.2, 0.95, 0.7, step=0.01)
sat_unc = st.sidebar.slider("Oil Saturation Uncertainty (±%)", 0.0, 50.0, 10.0, step=1.0)
fvf = st.sidebar.slider("Formation Volume Factor (Bo)", 1.0, 2.0, 1.2, step=0.05)
fvf_unc = st.sidebar.slider("FVF Uncertainty (±%)", 0.0, 50.0, 5.0, step=1.0)
ntg = st.sidebar.slider("Net-to-Gross (fraction)", 0.1, 1.0, 0.8, step=0.01)
ntg_unc = st.sidebar.slider("NTG Uncertainty (±%)", 0.0, 50.0, 10.0, step=1.0)
iterations = st.sidebar.number_input("Iterations", min_value=100, max_value=5000, value=1000, step=100)

########################################################
# Monte Carlo simulation function
########################################################
def run_monte_carlo(area_acres, area_unc, thickness, thick_unc, porosity, por_unc,
                    oil_saturation, sat_unc, fvf, fvf_unc, ntg, ntg_unc, iterations):
    # Generate random samples
    area_samples = np.random.normal(area_acres, area_acres * area_unc/100, iterations)
    thick_samples = np.random.normal(thickness, thickness * thick_unc/100, iterations)
    por_samples = np.random.normal(porosity, porosity * por_unc/100, iterations)
    sat_samples = np.random.normal(oil_saturation, oil_saturation * sat_unc/100, iterations)
    fvf_samples = np.random.normal(fvf, fvf * fvf_unc/100, iterations)
    ntg_samples = np.random.normal(ntg, ntg * ntg_unc/100, iterations)
    
    # Clip values
    area_samples = np.clip(area_samples, 1.0, None)
    thick_samples = np.clip(thick_samples, 1.0, None)
    por_samples = np.clip(por_samples, 0.05, 0.4)
    sat_samples = np.clip(sat_samples, 0.2, 0.95)
    fvf_samples = np.clip(fvf_samples, 1.0, 2.0)
    ntg_samples = np.clip(ntg_samples, 0.1, 1.0)
    
    # STOIIP calculation (in STB)
    stoiip_samples = (
        7758.0
        * area_samples
        * thick_samples
        * por_samples
        * sat_samples
        * ntg_samples
        / fvf_samples
    )
    
    # Convert to billions of STB (BSTB)
    stoiip_bstb = stoiip_samples / 1_000_000_000
    
    # Variable weights (std dev while holding other parameters at means)
    weights = {
        'Area': np.std(
            7758.0
            * area_samples
            * thick_samples.mean()
            * por_samples.mean()
            * sat_samples.mean()
            * ntg_samples.mean()
            / fvf_samples.mean()
        ),
        'Thickness': np.std(
            7758.0
            * area_samples.mean()
            * thick_samples
            * por_samples.mean()
            * sat_samples.mean()
            * ntg_samples.mean()
            / fvf_samples.mean()
        ),
        'Porosity': np.std(
            7758.0
            * area_samples.mean()
            * thick_samples.mean()
            * por_samples
            * sat_samples.mean()
            * ntg_samples.mean()
            / fvf_samples.mean()
        ),
        'Oil Sat': np.std(
            7758.0
            * area_samples.mean()
            * thick_samples.mean()
            * por_samples.mean()
            * sat_samples
            * ntg_samples.mean()
            / fvf_samples.mean()
        ),
        'NTG': np.std(
            7758.0
            * area_samples.mean()
            * thick_samples.mean()
            * por_samples.mean()
            * sat_samples.mean()
            * ntg_samples
            / fvf_samples.mean()
        ),
        'FVF': np.std(
            7758.0
            * area_samples.mean()
            * thick_samples.mean()
            * por_samples.mean()
            * sat_samples.mean()
            * ntg_samples.mean()
            / fvf_samples
        ),
    }
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    return stoiip_bstb, normalized_weights

########################################################
# Run the Monte Carlo simulation
########################################################
stoiip_samples_bstb, weights = run_monte_carlo(
    area_acres, area_unc, thickness, thick_unc,
    porosity, por_unc, oil_saturation, sat_unc,
    fvf, fvf_unc, ntg, ntg_unc, iterations
)

# Calculate P10, P50, and P90
p10 = np.percentile(stoiip_samples_bstb, 10)
p50 = np.percentile(stoiip_samples_bstb, 50)
p90 = np.percentile(stoiip_samples_bstb, 90)

########################################################
# Altair Charts
########################################################

# 1) Bar Chart of P10, P50, P90
cases_df = pd.DataFrame({
    'Case': ['P10 (Low)', 'P50 (Base)', 'P90 (High)'],
    'Volume (BSTB)': [p10, p50, p90]
})

color_scale = alt.Scale(
    domain=['P10 (Low)', 'P50 (Base)', 'P90 (High)'],
    range=['blue', 'green', 'red']
)

cases_chart_bars = (
    alt.Chart(cases_df)
    .mark_bar()
    .encode(
        x=alt.X('Case:N', title='', sort=None),
        y=alt.Y('Volume (BSTB):Q', title='Volume (BSTB)'),
        color=alt.Color('Case:N', scale=color_scale, legend=None),
        tooltip=['Case:N', 'Volume (BSTB):Q']
    )
    .properties(title=f'STOIIP Cases ({iterations} iterations)')
)

cases_chart_text = (
    alt.Chart(cases_df)
    .mark_text(dy=-5)
    .encode(
        x=alt.X('Case:N', sort=None),
        y=alt.Y('Volume (BSTB):Q'),
        text=alt.Text('Volume (BSTB):Q', format=",.3f"),
        color=alt.value('black')
    )
)

cases_chart = cases_chart_bars + cases_chart_text

# 2) Histogram with vertical lines at P10, P50, P90
dist_df = pd.DataFrame({'STOIIP (BSTB)': stoiip_samples_bstb})

hist = (
    alt.Chart(dist_df)
    .mark_bar(opacity=0.7)
    .encode(
        alt.X('STOIIP (BSTB):Q', bin=alt.Bin(maxbins=50), title='STOIIP (BSTB)'),
        alt.Y('count()', title='Count')
    )
    .properties(title='STOIIP Distribution')
)

rule_p10 = alt.Chart(pd.DataFrame({'value': [p10]})).mark_rule(color='blue').encode(x='value:Q')
rule_p50 = alt.Chart(pd.DataFrame({'value': [p50]})).mark_rule(color='green').encode(x='value:Q')
rule_p90 = alt.Chart(pd.DataFrame({'value': [p90]})).mark_rule(color='red').encode(x='value:Q')

dist_chart = alt.layer(hist, rule_p10, rule_p50, rule_p90).interactive()

# 3) Bar Chart of variable weights
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

# Display Altair charts
st.altair_chart(cases_chart, use_container_width=True)
st.altair_chart(dist_chart, use_container_width=True)
st.altair_chart(weights_chart, use_container_width=True)

# Display numeric STOIIP values
st.write(f"**P10 (Low):** {p10:,.3f} BSTB")
st.write(f"**P50 (Base):** {p50:,.3f} BSTB")
st.write(f"**P90 (High):** {p90:,.3f} BSTB")

########################################################
# 3D Visualization of the reservoir volume (Plotly)
########################################################
st.header("3D Reservoir Volume Visualization")

# Convert area (acres) to square meters
area_m2 = area_acres * 4046.8564224
# Convert thickness (ft) to meters
thickness_m = thickness * 0.3048

# We define cubic cells of 200 m in the XY direction
cell_size_xy = 200.0
cell_size_z = 200.0

import math

Nx = int(math.sqrt(area_m2 / (cell_size_xy**2)))
if Nx < 1:
    Nx = 1
Ny = Nx  # keep it square for simplicity

Nz = int(thickness_m / cell_size_z)
if Nz < 1:
    Nz = 1

Lx = Nx * cell_size_xy
Ly = Ny * cell_size_xy
Lz = Nz * cell_size_z

st.write(f"**Number of cells**: Nx={Nx}, Ny={Ny}, Nz={Nz}")
st.write(f"**Model dimensions** (approx): {Lx:,.1f} m × {Ly:,.1f} m × {Lz:,.1f} m")

def build_3d_wireframe(Nx, Ny, Nz, dx, dy, dz):
    """
    Build a list of Plotly Scatter3d objects representing the wireframe
    of a 3D grid with Nx, Ny, Nz cells in X, Y, Z directions.
    """
    lines = []
    
    # Lines in X-direction
    for j in range(Ny+1):
        for k in range(Nz+1):
            x0, y0, z0 = 0, j * dy, k * dz
            x1, y1, z1 = Nx * dx, j * dy, k * dz
            lines.append(
                go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False
                )
            )
    
    # Lines in Y-direction
    for i in range(Nx+1):
        for k in range(Nz+1):
            x0, y0, z0 = i * dx, 0, k * dz
            x1, y1, z1 = i * dx, Ny * dy, k * dz
            lines.append(
                go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False
                )
            )
    
    # Lines in Z-direction
    for i in range(Nx+1):
        for j in range(Ny+1):
            x0, y0, z0 = i * dx, j * dy, 0
            x1, y1, z1 = i * dx, j * dy, Nz * dz
            lines.append(
                go.Scatter3d(
                    x=[x0, x1],
                    y=[y0, y1],
                    z=[z0, z1],
                    mode='lines',
                    line=dict(color='black', width=1),
                    showlegend=False
                )
            )
    
    return lines

wireframe_lines = build_3d_wireframe(Nx, Ny, Nz, cell_size_xy, cell_size_xy, cell_size_z)
fig = go.Figure(data=wireframe_lines)

fig.update_layout(
    scene=dict(
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        zaxis_title='Z (m)',
        aspectmode='manual',
        aspectratio=dict(x=1, y=1, z=0.5)
    ),
    margin=dict(l=10, r=10, b=10, t=30),
    showlegend=False
)

# Display the Plotly figure
st.plotly_chart(fig, use_container_width=True)
