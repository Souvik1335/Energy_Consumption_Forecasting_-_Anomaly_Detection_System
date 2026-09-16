import streamlit as st


# Page Configuration
st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🚨",
    layout="wide"
)


# Page Title
st.title("🚨 Energy Usage Check")

st.write(
    "Check whether the current household energy consumption "
    "is within its expected range."
)


# Input Section
st.subheader("⚡ Enter Consumption Details")

col1, col2 = st.columns(2)

with col1:
    current_usage = st.number_input(
        "Current Energy Consumption (kW)",
        min_value=0.0,
        value=2.50
    )

with col2:
    expected_usage = st.number_input(
        "Expected Energy Consumption (kW)",
        min_value=0.0,
        value=2.48
    )


# Check Button
if st.button("🚨 Check Energy Usage", use_container_width=True):

    difference = abs(current_usage - expected_usage)

    # Simple Display
    st.subheader("📊 Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Usage",
            f"{current_usage:.3f} kW"
        )

    with col2:
        st.metric(
            "Expected Usage",
            f"{expected_usage:.3f} kW"
        )

    with col3:
        st.metric(
            "Difference",
            f"{difference:.3f} kW"
        )


    # Anomaly Status
    if difference > 0.143895:
        st.error(
            "🚨 Unusual energy consumption detected."
        )

        st.write(
            "The current energy usage is significantly different "
            "from the expected usage."
        )

    else:
        st.success(
            "🟢 Energy consumption is within the normal range."
        )

        st.write(
            "The current energy usage is close to the expected usage."
        )


# Information
st.markdown("---")

st.info(
    "This page helps identify unusually high or unexpected "
    "energy consumption."
)