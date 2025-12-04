import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --------------------------------------------
# Page config
# --------------------------------------------
st.set_page_config(page_title="Delay Analyzer Dashboard", page_icon="🚚", layout="wide")

st.title("🚚 Delay Analyzer Dashboard — Weather, Traffic, Vehicle & Agent Insights")
st.markdown(
    "Explore how **Weather**, **Traffic**, **Vehicle**, **Agent** and **Area** affect average delivery times."
)

# --------------------------------------------
# Load data (cached)
# --------------------------------------------
@st.cache_data
def load_data(path="no_na_Last_mile_Delivery_Data.csv"):
    df = pd.read_csv(path)
    return df

df = load_data()

# --------------------------------------------
# Basic cleaning / derived columns
# --------------------------------------------
# Ensure numeric
df["Delivery_Time"] = pd.to_numeric(df["Delivery_Time"], errors="coerce")
df["Agent_Age"] = pd.to_numeric(df["Agent_Age"], errors="coerce")
df["Agent_Rating"] = pd.to_numeric(df["Agent_Rating"], errors="coerce")

# Drop rows missing critical values (or keep but warn)
df = df.dropna(subset=["Delivery_Time", "Weather", "Traffic", "Vehicle", "Area", "Category"])

# Create Agent Age groups
def age_group(age):
    if pd.isna(age):
        return "Unknown"
    if age < 25:
        return "<25"
    if 25 <= age <= 40:
        return "25-40"
    return "40+"

df["Agent_Age_Group"] = df["Agent_Age"].apply(age_group)

# Normalize categories (optional, helps grouping)
df["Weather"] = df["Weather"].astype(str).str.strip().str.title()
df["Traffic"] = df["Traffic"].astype(str).str.strip().str.title()
df["Vehicle"] = df["Vehicle"].astype(str).str.strip().str.title()
df["Area"] = df["Area"].astype(str).str.strip()
df["Category"] = df["Category"].astype(str).str.strip()

# --------------------------------------------
# Sidebar: choose visualization + filters
# --------------------------------------------
viz = st.sidebar.radio(
    "Choose visualization",
    (
        "Delay Analyzer — Weather & Traffic (Bar Chart)",
        "Vehicle Comparison (Bar Chart)",
        "Agent Performance (Scatter Plot)",
        "Area Heatmap (Avg Delivery Time)",
        "Category Visualizer (Boxplot)"
    )
)

st.sidebar.markdown("### Global filters (optional)")
weather_options = ["All"] + sorted(df["Weather"].dropna().unique().tolist())
traffic_options = ["All"] + sorted(df["Traffic"].dropna().unique().tolist())

selected_weather = st.sidebar.selectbox("Weather", weather_options, index=0)
selected_traffic = st.sidebar.selectbox("Traffic", traffic_options, index=0)

# Apply global filters to working df
work_df = df.copy()
if selected_weather != "All":
    work_df = work_df[work_df["Weather"] == selected_weather]
if selected_traffic != "All":
    work_df = work_df[work_df["Traffic"] == selected_traffic]

# --------------------------------------------
# 1) Delay Analyzer — Bar Chart (Weather × Traffic)
# --------------------------------------------
if viz == "Delay Analyzer — Weather & Traffic (Bar Chart)":
    st.header("Delay Analyzer — Average Delivery Time by Weather & Traffic")
    delay_summary = (
        work_df.groupby(["Weather", "Traffic"])["Delivery_Time"]
        .mean()
        .reset_index()
        .rename(columns={"Delivery_Time": "Avg_Delivery_Time"})
    )

    # If filtered to a single weather, show grouped bars across traffic for that weather
    if selected_weather != "All":
        fig = px.bar(
            delay_summary,
            x="Traffic",
            y="Avg_Delivery_Time",
            color="Traffic",
            title=f"Avg Delivery Time under Traffic conditions — Weather: {selected_weather}",
            labels={"Avg_Delivery_Time": "Avg Delivery Time (minutes)"},
        )
    else:
        fig = px.bar(
            delay_summary,
            x="Weather",
            y="Avg_Delivery_Time",
            color="Traffic",
            barmode="group",
            title="Avg Delivery Time by Weather and Traffic",
            labels={"Avg_Delivery_Time": "Avg Delivery Time (minutes)"},
        )

    fig.update_layout(bargap=0.12, legend_title="Traffic Level")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Data (averages)")
    st.dataframe(delay_summary.sort_values("Avg_Delivery_Time", ascending=False).style.format({"Avg_Delivery_Time":"{:.2f} min"}))

# --------------------------------------------
# 2) Vehicle Comparison (Bar Chart) — With Independent Filters
# --------------------------------------------
elif viz == "Vehicle Comparison (Bar Chart)":
    st.header("Vehicle Comparison — Average Delivery Time by Vehicle Type")

    st.sidebar.markdown("### Vehicle Comparison Filters")

    # Independent filter: Category
    category_options = ["All"] + sorted(df["Category"].unique().tolist())
    selected_category_v = st.sidebar.selectbox(
        "Filter by Category (Vehicle View)", 
        category_options,
        index=0
    )

    # Independent filter: Area
    area_options = ["All"] + sorted(df["Area"].unique().tolist())
    selected_area_v = st.sidebar.selectbox(
        "Filter by Area (Vehicle View)", 
        area_options,
        index=0
    )

    # Independent filter: Weather (optional)
    weather_options_v = ["All"] + sorted(df["Weather"].unique().tolist())
    selected_weather_v = st.sidebar.selectbox(
        "Filter by Weather (Vehicle View)", 
        weather_options_v,
        index=0
    )

    # Independent filter: Traffic (optional)
    traffic_options_v = ["All"] + sorted(df["Traffic"].unique().tolist())
    selected_traffic_v = st.sidebar.selectbox(
        "Filter by Traffic (Vehicle View)", 
        traffic_options_v,
        index=0
    )

    # Start with full df (NOT work_df — to avoid global filter interference)
    vehicle_df = df.copy()

    # Apply filters only for this visualization
    if selected_category_v != "All":
        vehicle_df = vehicle_df[vehicle_df["Category"] == selected_category_v]

    if selected_area_v != "All":
        vehicle_df = vehicle_df[vehicle_df["Area"] == selected_area_v]

    if selected_weather_v != "All":
        vehicle_df = vehicle_df[vehicle_df["Weather"] == selected_weather_v]

    if selected_traffic_v != "All":
        vehicle_df = vehicle_df[vehicle_df["Traffic"] == selected_traffic_v]

    # Aggregate delivery time by vehicle
    vehicle_summary = (
        vehicle_df.groupby("Vehicle")["Delivery_Time"]
        .mean()
        .reset_index()
        .rename(columns={"Delivery_Time": "Avg_Delivery_Time"})
    )

    # Sort by average delivery time
    vehicle_summary = vehicle_summary.sort_values("Avg_Delivery_Time")

    # Bar chart
    fig = px.bar(
        vehicle_summary,
        x="Vehicle",
        y="Avg_Delivery_Time",
        title=(
            "Average Delivery Time by Vehicle Type"
            f"{'' if selected_category_v == 'All' else f' — Category: {selected_category_v}'}"
            f"{'' if selected_area_v == 'All' else f' — Area: {selected_area_v}'}"
        ),
        labels={"Avg_Delivery_Time": "Avg Delivery Time (minutes)"}
    )

    fig.update_layout(
        xaxis_title="Vehicle Type",
        yaxis_title="Average Delivery Time (minutes)",
        bargap=0.15
    )
    
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Filtered Vehicle Summary")
    st.dataframe(
        vehicle_summary.style.format({"Avg_Delivery_Time": "{:.2f} min"})
    )


# --------------------------------------------
# 3) Agent Performance Scatter Plot
# --------------------------------------------
elif viz == "Agent Performance (Scatter Plot)":
    st.header("Agent Performance — Rating vs Delivery Time")
    st.markdown(
        "Scatter plot of `Agent_Rating` (x) vs `Delivery_Time` (y), colored by `Agent_Age_Group`. "
        "Use this to spot whether higher-rated or older/younger agents perform faster deliveries."
    )

    # Optionally filter rating range
    min_rating = int(work_df["Agent_Rating"].min()) if not work_df["Agent_Rating"].isna().all() else 0
    max_rating = int(work_df["Agent_Rating"].max()) if not work_df["Agent_Rating"].isna().all() else 5
    rating_range = st.sidebar.slider("Agent Rating range", min_value=min_rating, max_value=max_rating, value=(min_rating, max_rating))

    plot_df = work_df.dropna(subset=["Agent_Rating", "Delivery_Time"])
    plot_df = plot_df[(plot_df["Agent_Rating"] >= rating_range[0]) & (plot_df["Agent_Rating"] <= rating_range[1])]

    fig = px.scatter(
        plot_df,
        x="Agent_Rating",
        y="Delivery_Time",
        color="Agent_Age_Group",
        hover_data=["Order_ID", "Vehicle", "Area", "Weather", "Traffic"],
        title="Agent Rating vs Delivery Time (colored by Age Group)",
        labels={"Delivery_Time": "Delivery Time (minutes)", "Agent_Rating": "Agent Rating"}
    )
    fig.update_traces(marker=dict(size=9, opacity=0.8), selector=dict(mode="markers"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Scatter data preview")
    st.dataframe(plot_df[["Order_ID", "Agent_Rating", "Agent_Age", "Agent_Age_Group", "Delivery_Time", "Vehicle", "Area"]].head(200))

# --------------------------------------------
# 4) Area Heatmap (Avg Delivery Time per Area)
# --------------------------------------------
elif viz == "Area Heatmap (Avg Delivery Time)":
    st.header("Area Heatmap — Average Delivery Time by Area")
    st.markdown(
        "This heatmap shows average delivery times per Area. Areas are sorted by average delay so the vertical heat gradient highlights slow regions."
    )

    area_summary = (
        work_df.groupby("Area")["Delivery_Time"]
        .mean()
        .reset_index()
        .rename(columns={"Delivery_Time": "Avg_Delivery_Time"})
    )

    # Sort areas by avg time
    area_summary = area_summary.sort_values("Avg_Delivery_Time", ascending=False)
    # Create a 2D matrix with one column to visualize as heatmap (Area × 1)
    heat_vals = area_summary["Avg_Delivery_Time"].values.reshape(-1, 1)

    fig = px.imshow(
        heat_vals,
        labels=dict(x="", y="Area", color="Avg Delivery Time (min)"),
        x=["Avg Delivery Time"],
        y=area_summary["Area"],
        title="Heatmap — Average Delivery Time by Area",
        aspect="auto",
    )
    # Enhance colorbar label
    fig.update_coloraxes(colorbar_title="Avg Time (min)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Area averages")
    st.dataframe(area_summary.style.format({"Avg_Delivery_Time":"{:.2f} min"}))

# --------------------------------------------
# 5) Category Visualizer (Boxplot)
# --------------------------------------------
elif viz == "Category Visualizer (Boxplot)":
    st.header("Category Visualizer — Delivery Time Distribution by Category")
    st.markdown("Boxplots show the distribution of Delivery_Time for each product Category.")

    # Optionally let user choose top N categories by count to make plot readable
    top_n = st.sidebar.slider("Show top N categories by count", min_value=5, max_value=50, value=20)
    cat_counts = work_df["Category"].value_counts().nlargest(top_n).index.tolist()
    plot_df = work_df[work_df["Category"].isin(cat_counts)]

    fig = px.box(
        plot_df,
        x="Category",
        y="Delivery_Time",
        points="outliers",
        title=f"Delivery Time Distribution by Category (top {top_n} categories)",
        labels={"Delivery_Time": "Delivery Time (minutes)"}
    )
    fig.update_layout(xaxis_title="Category", yaxis_title="Delivery Time (minutes)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Category counts (top shown)")
    cat_summary = plot_df.groupby("Category")["Delivery_Time"].agg(["count", "mean", "median"]).reset_index()
    st.dataframe(cat_summary.sort_values("count", ascending=False).style.format({"mean":"{:.2f}", "median":"{:.2f}"}))



