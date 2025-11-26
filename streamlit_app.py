import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------
# 1. Page setup
# --------------------------------------------
st.set_page_config(
    page_title="Delay Analyzer Dashboard",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Delay Analyzer — Weather & Traffic Impact on Delivery Time")
st.markdown("""
This dashboard shows the **average Delivery Time** under different  
**Weather** and **Traffic** conditions.  
It helps managers quickly identify how rain, storms, or heavy traffic influence delivery delays.
""")

# --------------------------------------------
# 2. Load data
# --------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("delivery_with_delay_features.csv")
    return df

df = load_data()

# --------------------------------------------
# 3. Prepare summary data
# --------------------------------------------
delay_summary = (
    df.groupby(["Weather", "Traffic"])["Delivery_Time"]
      .mean()
      .reset_index()
      .rename(columns={"Delivery_Time": "Avg_Delivery_Time"})
)

# --------------------------------------------
# 4. Optional filter for weather (you can remove if not needed)
# --------------------------------------------
weather_options = sorted(delay_summary["Weather"].unique())
selected_weather = st.selectbox("🌦️ Select Weather Condition (Optional Filter):", ["All"] + weather_options)

if selected_weather != "All":
    filtered_df = delay_summary[delay_summary["Weather"] == selected_weather]
else:
    filtered_df = delay_summary

# --------------------------------------------
# 5. Bar Chart — Delay Analyzer
# --------------------------------------------
fig = px.bar(
    filtered_df,
    x="Weather",
    y="Avg_Delivery_Time",
    color="Traffic",
    barmode="group",
    title="Average Delivery Time by Weather and Traffic Conditions",
    labels={"Avg_Delivery_Time": "Avg Delivery Time (minutes)"}
)

fig.update_layout(
    xaxis_title="Weather Condition",
    yaxis_title="Average Delivery Time (minutes)",
    legend_title="Traffic Level",
    bargap=0.15
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------
# 6. Display Summary Table
# --------------------------------------------
st.markdown("### 📊 Summary Table")
st.dataframe(
    filtered_df.style.format({"Avg_Delivery_Time": "{:.2f} min"})
)

st.caption("Data source: Last-mile delivery dataset — grouped by Weather and Traffic.")