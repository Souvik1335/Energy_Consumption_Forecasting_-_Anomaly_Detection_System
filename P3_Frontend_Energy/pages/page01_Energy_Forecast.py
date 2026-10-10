import streamlit as st
import requests


st.set_page_config(
    page_title="Energy Forecast",
    page_icon="🔮",
    layout="wide"
)


BACKEND_URL = "https://energy-forecasting-api-k930.onrender.com"


st.title("🔮 Energy Forecast")

st.write(
    "Enter the household energy information below to estimate "
    "energy consumption."
)


st.subheader("📅 Date & Time")

col1, col2, col3, col4 = st.columns(4)

with col1:

    hour = st.number_input(
        "Hour",
        min_value=0,
        max_value=23,
        value=18
    )

with col2:

    day = st.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=20
    )

with col3:

    day_of_week = st.number_input(
        "Day of Week",
        min_value=0,
        max_value=6,
        value=5
    )

with col4:

    month = st.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=11
    )


year = st.number_input(
    "Year",
    min_value=2000,
    max_value=2030,
    value=2009
)


st.subheader("🏠 Household Information")

col1, col2 = st.columns(2)

with col1:

    is_weekend = st.selectbox(
        "Is it a weekend?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

with col2:

    is_peak_hour = st.selectbox(
        "Is it a peak usage hour?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )


st.subheader("⚡ Previous Energy Usage")

col1, col2, col3 = st.columns(3)

with col1:

    lag_1 = st.number_input(
        "Recent Usage",
        value=2.45
    )

with col2:

    lag_2 = st.number_input(
        "Previous Usage",
        value=2.38
    )

with col3:

    lag_3 = st.number_input(
        "Earlier Usage",
        value=2.41
    )


col1, col2, col3, col4 = st.columns(4)

with col1:

    lag_24 = st.number_input(
        "Usage 24 Steps Ago",
        value=2.17
    )

with col2:

    lag_48 = st.number_input(
        "Usage 48 Steps Ago",
        value=2.31
    )

with col3:

    lag_60 = st.number_input(
        "Usage 60 Steps Ago",
        value=2.52
    )

with col4:

    lag_1440 = st.number_input(
        "Usage 1440 Steps Ago",
        value=2.08
    )


lag_10080 = st.number_input(
    "Usage 10080 Steps Ago",
    value=2.26
)


st.subheader("📈 Usage Statistics")

col1, col2, col3 = st.columns(3)

with col1:

    rolling_mean_15 = st.number_input(
        "Short-Term Average",
        value=2.43
    )

with col2:

    rolling_mean_60 = st.number_input(
        "Medium-Term Average",
        value=2.39
    )

with col3:

    rolling_mean_1440 = st.number_input(
        "Daily Average",
        value=2.21
    )


col1, col2, col3 = st.columns(3)

with col1:

    rolling_std_15 = st.number_input(
        "Short-Term Variation",
        value=0.18
    )

with col2:

    rolling_std_60 = st.number_input(
        "Medium-Term Variation",
        value=0.24
    )

with col3:

    rolling_std_1440 = st.number_input(
        "Daily Variation",
        value=0.51
    )


st.subheader("⚡ Current Consumption")

actual_value = st.number_input(
    "Current Energy Consumption",
    min_value=0.0,
    value=2.50
)


st.markdown("---")


if st.button(
    "🔮 Predict Energy",
    use_container_width=True
):

    input_data = {
        "Hour": int(hour),
        "Day": int(day),
        "DayOfWeek": int(day_of_week),
        "Month": int(month),
        "Year": int(year),

        "IsWeekend": int(is_weekend),
        "IsPeakHour": int(is_peak_hour),

        "Lag_1": float(lag_1),
        "Lag_2": float(lag_2),
        "Lag_3": float(lag_3),
        "Lag_24": float(lag_24),
        "Lag_48": float(lag_48),
        "Lag_60": float(lag_60),
        "Lag_1440": float(lag_1440),
        "Lag_10080": float(lag_10080),

        "Rolling_Mean_15": float(rolling_mean_15),
        "Rolling_Mean_60": float(rolling_mean_60),
        "Rolling_Mean_1440": float(rolling_mean_1440),

        "Rolling_Std_15": float(rolling_std_15),
        "Rolling_Std_60": float(rolling_std_60),
        "Rolling_Std_1440": float(rolling_std_1440),

        "Global_active_power": float(actual_value)
    }


    try:

        with st.spinner("Generating prediction..."):

            response = requests.post(
                f"{BACKEND_URL}/predict",
                json=input_data,
                timeout=60
            )


        response.raise_for_status()

        result = response.json()


        st.success(
            "Prediction completed successfully!"
        )


        st.subheader("📊 Prediction Result")


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(
                "Current Usage",
                f"{result['actual_global_active_power']:.3f} kW"
            )


        with col2:

            st.metric(
                "Predicted Usage",
                f"{result['predicted_global_active_power']:.3f} kW"
            )


        with col3:

            st.metric(
                "Difference",
                f"{result['absolute_residual']:.3f} kW"
            )


        st.write(
            f"**Residual:** "
            f"{result['residual']:.3f} kW"
        )


        if result["anomaly"] == 1:

            st.error(
                "🚨 Unusual energy consumption detected."
            )

        else:

            st.success(
                "🟢 Energy consumption is within "
                "the normal range."
            )


        st.write(
            f"**Status:** {result['anomaly_status']}"
        )


    except requests.exceptions.Timeout:

        st.error(
            "❌ The prediction server took too long to respond."
        )


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the prediction server. "
            "Please check the backend connection."
        )


    except requests.exceptions.HTTPError as error:

        st.error(
            f"❌ Prediction failed: {error}"
        )


    except requests.exceptions.RequestException as error:

        st.error(
            f"❌ An error occurred: {error}"
        )