import streamlit as st

# Streamlit App Title
st.title("Reservoir STOIIP Calculator")

# Introduction
st.markdown("""
This interactive app calculates the Stock Tank Original Oil In Place (STOIIP) using the standard volumetric equation:

\[ STOIIP = \frac{7758 \times A \times h \times \phi \times (1 - S_w)}{B_{oi}} \]

Where:
- **STOIIP** is the Stock Tank Original Oil In Place (STB)
- **A** is the reservoir area (acres)
- **h** is the reservoir net pay thickness (ft)
- **φ (phi)** is the reservoir porosity (fraction)
- **S₍w₎** is the water saturation (fraction)
- **B₍oi₎** is the formation volume factor (reservoir barrels/stock tank barrel)
""")

# Interactive sliders for parameters
st.sidebar.header("Input Parameters")

area = st.sidebar.slider("Reservoir Area (acres)", min_value=10, max_value=10000, value=1000, step=10)
thickness = st.sidebar.slider("Net Pay Thickness (ft)", min_value=1, max_value=500, value=50, step=1)
porosity = st.sidebar.slider("Porosity (fraction)", min_value=0.05, max_value=0.35, value=0.20, step=0.01)
water_saturation = st.sidebar.slider("Water Saturation (fraction)", min_value=0.05, max_value=0.80, value=0.25, step=0.01)
formation_volume_factor = st.sidebar.slider("Formation Volume Factor (Boi)", min_value=1.0, max_value=2.5, value=1.2, step=0.01)

# STOIIP calculation function
def calculate_stoiip(A, h, phi, Sw, Boi):
    stoiip = (7758 * A * h * phi * (1 - Sw)) / Boi
    return stoiip

# Perform STOIIP calculation
stoiip_result = calculate_stoiip(area, thickness, porosity, water_saturation, formation_volume_factor)

# Display result
st.header("Calculated STOIIP")
st.success(f"STOIIP = {stoiip_result:,.2f} Stock Tank Barrels (STB)")