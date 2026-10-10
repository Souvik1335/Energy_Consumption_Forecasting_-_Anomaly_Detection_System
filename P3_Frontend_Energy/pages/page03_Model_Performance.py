import streamlit as st


st.set_page_config(
    page_title="Model Performance",
    page_icon="📊",
    layout="wide"
)


st.title("📊 Model Performance")

st.write(
    "Performance of the energy consumption prediction system "
    "on previously unseen test data."
)


st.subheader("Prediction Model")

st.write(
    "The system uses a Random Forest regression model to "
    "predict household energy consumption."
)


st.subheader("Test Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "R² Score",
        "0.9349"
    )

with col2:

    st.metric(
        "MAE",
        "0.0878"
    )

with col3:

    st.metric(
        "RMSE",
        "0.2170"
    )

with col4:

    st.metric(
        "MSE",
        "0.0471"
    )


st.subheader("Validation Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "R² Score",
        "0.9410"
    )

with col2:

    st.metric(
        "MAE",
        "0.1023"
    )

with col3:

    st.metric(
        "RMSE",
        "0.2533"
    )

with col4:

    st.metric(
        "MSE",
        "0.0642"
    )


st.subheader("What do these results mean?")

st.write(
    """
    **R² Score:** Shows how well the model explains variations
    in energy consumption. A value closer to 1 indicates
    stronger predictive performance.

    **MAE:** Shows the average absolute difference between
    predicted and actual energy consumption.

    **RMSE:** Measures prediction error while giving more
    importance to larger errors.

    **MSE:** Measures the average squared prediction error.
    """
)


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
        "Model Configuration"
    )


st.info(
    "The system compares actual and predicted energy usage. "
    "The absolute residual is then compared with the configured "
    "anomaly threshold to identify unusual consumption."
)


st.markdown("---")


st.success(
    "✅ The energy forecasting and anomaly detection "
    "system is ready for use."
)