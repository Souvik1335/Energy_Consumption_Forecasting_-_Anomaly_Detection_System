import streamlit as st


st.set_page_config(
    page_title="Energy Monitoring",
    page_icon="⚡",
    layout="wide"
)


st.title("⚡ Energy Consumption Monitoring")

st.write(
    "Monitor household energy consumption, predict upcoming usage, "
    "and identify unusual consumption patterns."
)


st.markdown("---")


st.subheader("What can you do?")


col1, col2, col3 = st.columns(3)


with col1:

    st.info(
        "### 🔮 Energy Forecast\n\n"
        "Predict household energy consumption "
        "using Machine Learning."
    )


with col2:

    st.warning(
        "### 🚨 Anomaly Detection\n\n"
        "Identify unusual energy consumption "
        "patterns."
    )


with col3:

    st.success(
        "### 📊 Model Performance\n\n"
        "Explore the performance of the "
        "Machine Learning model."
    )


st.markdown("---")


st.subheader("Getting Started")

st.write(
    "Choose an option from the sidebar to explore "
    "the Energy Forecast, Anomaly Detection, or "
    "Model Performance sections."
)


st.markdown("---")


st.caption(
    "Energy Consumption Forecasting & Anomaly Detection System"
)