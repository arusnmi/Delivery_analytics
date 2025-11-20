import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------
# 1. Page setup
# --------------------------------------------
st.set_page_config(
    page_title="Delay Analyzer Dashboard",
    page_icon="🚚",
    layout="centered"
)

st.title("🚚 Delay Analyzer Dashboard")
st.markdown("""
Explore how **Traffic** affects average delivery times under different **Weather** conditions.
Use the dropdown below to select a weather type and view the corresponding traffic distribution.
""")

# --------------------------------------------
# 2. Load data
# --------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("no_na_Last_mile_Delivery_Data.csv")
    return df

df = load_data()

# --------------------------------------------
# 3. Data preparation
# --------------------------------------------
delay_summary = (
    df.groupby(["Weather", "Traffic"])["Delivery_Time"]
      .mean()
      .reset_index()
      .rename(columns={"Delivery_Time": "Avg_Delivery_Time"})
)

# --------------------------------------------
# 4. Sidebar filter
# --------------------------------------------
weather_options = delay_summary["Weather"].unique()
selected_weather = st.selectbox("🌦️ Select Weather Condition:", sorted(weather_options))

# Filter for the selected weather
subset = delay_summary[delay_summary["Weather"] == selected_weather]

# --------------------------------------------
# 5. Interactive pie chart
# --------------------------------------------
fig = px.pie(
    subset,
    names="Traffic",
    values="Avg_Delivery_Time",
    title=f"Traffic Distribution of Average Delivery Time ({selected_weather} Weather)",
    color="Traffic",
    color_discrete_map={
        "Low": "#00CC96",
        "Medium": "#FFA15A",
        "High": "#FF6692",
        "Jam": "#EF553B"
    },
    hole=0.3
)

fig.update_traces(textinfo="percent+label")

# Display chart
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
# 6. Optional insights
# --------------------------------------------
st.markdown("### 📊 Summary Data")
st.dataframe(subset.style.format({"Avg_Delivery_Time": "{:.2f} min"}))

st.caption("Data source: Last Mile Delivery dataset — average delivery time grouped by traffic and weather.")
