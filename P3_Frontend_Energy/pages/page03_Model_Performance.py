import streamlit as st


# Page Configuration
st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)


# Page Title
st.title("📊 Model Performance")

st.write(
    "Performance of the energy consumption prediction system "
    "on previously unseen data."
)


# Model Information
st.subheader("Prediction Model")

st.write(
    "The system uses a Random Forest model to predict "
    "household energy consumption."
)


# Test Performance
st.subheader("Test Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "R² Score",
        "0.9432"
    )

with col2:
    st.metric(
        "MAE",
        "0.0768"
    )

with col3:
    st.metric(
        "RMSE",
        "0.2027"
    )

with col4:
    st.metric(
        "MSE",
        "0.0411"
    )


# What The Metrics Mean
st.subheader("What do these results mean?")

st.write(
    """
    **R² Score:** Shows how well the system explains changes
    in energy consumption. A value closer to 1 indicates
    stronger predictive performance.

    **MAE:** Shows the average difference between predicted
    and actual energy consumption.

    **RMSE:** Shows the typical prediction error while giving
    more importance to larger errors.

    **MSE:** Measures the squared prediction error.
    """
)


# Anomaly Detection
st.subheader("Anomaly Detection")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Detection Method",
        "Residual Analysis"
    )

with col2:
    st.metric(
        "Detection Threshold",
        "0.144"
    )


st.info(
    "The system compares predicted and actual energy usage "
    "to identify unusually different consumption."
)


# Final Status
st.success(
    "✅ The prediction and anomaly detection system "
    "is ready for use."
)