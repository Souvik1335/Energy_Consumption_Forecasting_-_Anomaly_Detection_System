import streamlit as st


# Page Configuration
st.set_page_config(
    page_title="Energy Monitoring",
    page_icon="⚡",
    layout="wide"
)


# Main Title
st.title("⚡ Energy Consumption Monitoring")


# Introduction
st.write(
    "Monitor household energy consumption, predict upcoming usage, "
    "and identify unusual consumption patterns."
)


# Quick Overview
st.subheader("What can you do?")


col1, col2, col3 = st.columns(3)

with col1:
    st.info(
        "### 🔮 Predict Energy\n"
        "Estimate household energy consumption."
    )

with col2:
    st.warning(
        "### 🚨 Detect Unusual Usage\n"
        "Find consumption patterns that look unusual."
    )

with col3:
    st.success(
        "### 📊 View Results\n"
        "Understand predictions and system results."
    )


# Getting Started
st.subheader("Getting Started")

st.write(
    "Choose an option from the sidebar to begin."
)


# Footer
st.markdown("---")

st.caption(
    "Energy Consumption Monitoring System"
)