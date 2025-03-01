import streamlit as st

def main():
    # Title
    st.title("Reservoir STOIIP Calculator")

    # Introduction
    st.markdown("""
    This Streamlit application calculates the Stock Tank Original Oil In Place (STOIIP) for a reservoir using the following standard formula:

    \\[
    \\text{STOIIP} = \\frac{7758 \\times A \\times h \\times \\phi \\times (1 - S_{w})}{B_{oi}}
    \\]

    Where:
    - **STOIIP**: Stock Tank Original Oil in Place (in STB)
    - **A**: Reservoir area (acres)
    - **h**: Net pay thickness (ft)
    - **φ (phi)**: Reservoir porosity (fraction)
    - **S₍w₎**: Water saturation (fraction)
    - **B₍oi₎**: Formation volume factor (bbls/stock tank barrel)
    """)

    # Sidebar input parameters
    st.sidebar.header("Input Parameters")

    area = st.sidebar.slider(
        "Reservoir Area (acres)",
        min_value=10,
        max_value=10000,
        value=1000,
        step=10
    )

    thickness = st.sidebar.slider(
        "Net Pay Thickness (ft)",
        min_value=1,
        max_value=500,
        value=50,
        step=1
    )

    porosity = st.sidebar.slider(
        "Porosity (fraction)",
        min_value=0.05,
        max_value=0.35,
        value=0.20,
        step=0.01
    )

    water_saturation = st.sidebar.slider(
        "Water Saturation (fraction)",
        min_value=0.05,
        max_value=0.80,
        value=0.25,
        step=0.01
    )

    formation_volume_factor = st.sidebar.slider(
        "Formation Volume Factor (Boi)",
        min_value=1.0,
        max_value=2.5,
        value=1.2,
        step=0.01
    )

    # Calculation function
    def calculate_stoiip(A, h, phi, Sw, Boi):
        """
        This function calculates STOIIP using the standard formula:
        STOIIP (STB) = (7758 * A * h * phi * (1 - Sw)) / Boi
        """
        stoiip_val = (7758 * A * h * phi * (1 - Sw)) / Boi
        return stoiip_val

    # Calculate STOIIP
    stoiip_result = calculate_stoiip(
        area,
        thickness,
        porosity,
        water_saturation,
        formation_volume_factor
    )

    # Display the result
    st.header("Calculated STOIIP")
    st.success(f"STOIIP = {stoiip_result:,.2f} STB")

if __name__ == '__main__':
    main()
