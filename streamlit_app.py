"""
Streamlit dashboard for Last-mile Delivery — Stage 5 & 6

Usage:
    pip install streamlit pandas plotly
    streamlit run streamlit_last_mile_dashboard.py

This app reads the CSV (default path is /mnt/data/no_na_Last_mile_Delivery_Data.csv) or allows uploading a file.
It creates five compulsory visualizations and sidebar filters that update charts live.

Notes:
 - "Late delivery" is defined here as Delivery_Time > overall mean delivery time; change if you have a different business rule.
 - The Area heatmap shows average Delivery_Time by Area (rows) vs Day of Week (columns).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(layout="wide", page_title="Last-mile Delivery Dashboard")

DEFAULT_CSV = "delay_based_features.csv"

@st.cache_data
def load_data(path=None, uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv(path)

    # parse datetimes
    if 'Order_Date' in df.columns and 'Order_Time' in df.columns:
        df['Order_Datetime'] = pd.to_datetime(df['Order_Date'].astype(str) + ' ' + df['Order_Time'].astype(str), errors='coerce')
    else:
        # try parsing Order_Time alone
        if 'Order_Time' in df.columns:
            df['Order_Datetime'] = pd.to_datetime(df['Order_Time'], errors='coerce')
        else:
            df['Order_Datetime'] = pd.NaT

    if 'Pickup_Time' in df.columns:
        # assume same date as Order_Date if present
        if 'Order_Date' in df.columns:
            df['Pickup_Datetime'] = pd.to_datetime(df['Order_Date'].astype(str) + ' ' + df['Pickup_Time'].astype(str), errors='coerce')
        else:
            df['Pickup_Datetime'] = pd.to_datetime(df['Pickup_Time'], errors='coerce')
    else:
        df['Pickup_Datetime'] = pd.NaT

    # basic time features
    df['Order_Hour'] = df['Order_Datetime'].dt.hour.fillna(-1).astype(int)
    df['Pickup_Hour'] = df['Pickup_Datetime'].dt.hour.fillna(-1).astype(int)
    df['Order_DayOfWeek'] = df['Order_Datetime'].dt.dayofweek.fillna(-1).astype(int)
    df['Is_Weekend'] = df['Order_DayOfWeek'].isin([5,6]).astype(int)

    # order-to-pickup minutes
    df['Order_to_Pickup_Minutes'] = (df['Pickup_Datetime'] - df['Order_Datetime']).dt.total_seconds() / 60

    # agent age group
    def age_group(age):
        try:
            age = int(age)
        except:
            return 'Unknown'
        if age < 25:
            return '<25'
        elif age <= 40:
            return '25-40'
        else:
            return '40+'
    if 'Agent_Age' in df.columns:
        df['Agent_Age_Group'] = df['Agent_Age'].apply(age_group)
    else:
        df['Agent_Age_Group'] = 'Unknown'

    # ensure Delivery_Time numeric
    if 'Delivery_Time' in df.columns:
        df['Delivery_Time'] = pd.to_numeric(df['Delivery_Time'], errors='coerce')
    else:
        df['Delivery_Time'] = np.nan

    return df

# ---------- App layout ----------
st.title("Last-mile Delivery — Delay Analyzer Dashboard")

with st.sidebar:
    st.header("Data source & filters")
    uploaded = st.file_uploader("Upload a CSV (or use default dataset)", type=['csv'])
    df = load_data(DEFAULT_CSV, uploaded)

    # date range filter if Order_Datetime exists
    if df['Order_Datetime'].notna().any():
        min_date = df['Order_Datetime'].min().date()
        max_date = df['Order_Datetime'].max().date()
        date_range = st.date_input("Order date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    else:
        date_range = None

    # multiple filters
    weather_sel = st.multiselect("Weather", options=sorted(df['Weather'].dropna().unique().tolist()) if 'Weather' in df.columns else [], default=None)
    traffic_sel = st.multiselect("Traffic", options=sorted(df['Traffic'].dropna().unique().tolist()) if 'Traffic' in df.columns else [], default=None)
    vehicle_sel = st.multiselect("Vehicle", options=sorted(df['Vehicle'].dropna().unique().tolist()) if 'Vehicle' in df.columns else [], default=None)
    category_sel = st.multiselect("Category", options=sorted(df['Category'].dropna().unique().tolist()) if 'Category' in df.columns else [], default=None)
    area_sel = st.multiselect("Area", options=sorted(df['Area'].dropna().unique().tolist()) if 'Area' in df.columns else [], default=None)

    st.markdown("---")
    st.markdown("**Metrics options**")
    late_def = st.selectbox("Late delivery rule", options=['Above mean Delivery_Time', 'Above median Delivery_Time', 'Custom minutes'], index=0)
    custom_late_min = None
    if late_def == 'Custom minutes':
        custom_late_min = st.number_input('Custom late threshold (minutes)', min_value=1, value=60)

# ---------- Filtering ----------
filtered = df.copy()

if date_range and len(date_range) == 2:
    start, end = date_range
    # include entire end day
    filtered = filtered[(filtered['Order_Datetime'].dt.date >= start) & (filtered['Order_Datetime'].dt.date <= end)]

if weather_sel:
    filtered = filtered[filtered['Weather'].isin(weather_sel)]
if traffic_sel:
    filtered = filtered[filtered['Traffic'].isin(traffic_sel)]
if vehicle_sel:
    filtered = filtered[filtered['Vehicle'].isin(vehicle_sel)]
if category_sel:
    filtered = filtered[filtered['Category'].isin(category_sel)]
if area_sel:
    filtered = filtered[filtered['Area'].isin(area_sel)]

# ---------- Key metrics ----------
st.subheader("Key metrics")
col1, col2, col3 = st.columns(3)
with col1:
    mean_delivery = filtered['Delivery_Time'].mean()
    st.metric("Average delivery time (minutes)", f"{mean_delivery:.1f}" if not np.isnan(mean_delivery) else "N/A")
with col2:
    median_delivery = filtered['Delivery_Time'].median()
    st.metric("Median delivery time (minutes)", f"{median_delivery:.1f}" if not np.isnan(median_delivery) else "N/A")
with col3:
    # percent late
    if late_def == 'Above mean Delivery_Time':
        threshold = mean_delivery
    elif late_def == 'Above median Delivery_Time':
        threshold = median_delivery
    else:
        threshold = custom_late_min
    if np.isnan(threshold):
        pct_late = np.nan
    else:
        pct_late = (filtered['Delivery_Time'] > threshold).mean() * 100
    st.metric("% Late deliveries", f"{pct_late:.1f}%" if not np.isnan(pct_late) else "N/A")

st.markdown("---")

# ---------- Visualizations (main area) ----------
st.header("Planned Visualizations")

# 1) Delay Analyzer (Bar Chart): average Delivery_Time under different Weather and Traffic conditions
st.subheader("Delay Analyzer — Avg Delivery Time by Weather & Traffic")
if 'Weather' in filtered.columns and 'Traffic' in filtered.columns:
    grp = filtered.groupby(['Weather', 'Traffic'])['Delivery_Time'].mean().reset_index()
    if not grp.empty:
        fig1 = px.bar(grp, x='Weather', y='Delivery_Time', color='Traffic', barmode='group',
                      labels={'Delivery_Time':'Avg Delivery Time (min)'}, title='Avg Delivery Time by Weather and Traffic')
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.write("No data for selected filters.")
else:
    st.write("Required columns (Weather, Traffic) not present in data.")

# 2) Vehicle Comparison (Bar Chart): avg Delivery_Time by Vehicle
st.subheader("Vehicle Comparison — Avg Delivery Time by Vehicle")
if 'Vehicle' in filtered.columns:
    veh = filtered.groupby('Vehicle')['Delivery_Time'].mean().reset_index().sort_values('Delivery_Time')
    if not veh.empty:
        fig2 = px.bar(veh, x='Vehicle', y='Delivery_Time', labels={'Delivery_Time':'Avg Delivery Time (min)'}, title='Avg Delivery Time by Vehicle')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("No data for selected filters.")
else:
    st.write("Vehicle column not present.")

# 3) Agent Performance Scatter Plot
st.subheader("Agent Performance — Rating vs Delivery Time")
if all(c in filtered.columns for c in ['Agent_Rating', 'Delivery_Time', 'Agent_Age_Group']):
    scatter_df = filtered.dropna(subset=['Agent_Rating','Delivery_Time'])
    if not scatter_df.empty:
        fig3 = px.scatter(scatter_df, x='Agent_Rating', y='Delivery_Time', color='Agent_Age_Group',
                          labels={'Delivery_Time':'Delivery Time (min)', 'Agent_Rating':'Agent Rating'},
                          title='Agent Rating vs Delivery Time (colored by Age Group)', hover_data=['Order_ID'])
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.write("No data for selected filters.")
else:
    st.write("Required columns for this plot (Agent_Rating, Agent_Age or Agent_Age_Group, Delivery_Time) not present.")

# 4) Area Heatmap
st.subheader("Area Heatmap — Avg Delivery Time by Area and Day of Week")
if 'Area' in filtered.columns and 'Order_DayOfWeek' in filtered.columns:
    heat = filtered.groupby(['Area','Order_DayOfWeek'])['Delivery_Time'].mean().reset_index()
    if not heat.empty:
        pivot = heat.pivot(index='Area', columns='Order_DayOfWeek', values='Delivery_Time').fillna(0)
        # reorder day columns 0..6
        cols = [c for c in range(7) if c in pivot.columns]
        pivot = pivot[cols]
        fig4 = px.imshow(pivot, labels=dict(x='Day of week (0=Mon)', y='Area', color='Avg Delivery Time (min)'),
                         x=pivot.columns.astype(str), y=pivot.index, aspect='auto', title='Avg Delivery Time: Area vs Day of Week')
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.write("No data for selected filters.")
else:
    st.write("Required columns for heatmap (Area, Order_DayOfWeek) not present.")

# 5) Category Visualizer (Boxplot)
st.subheader("Category Visualizer — Delivery Time distribution by Category")
if 'Category' in filtered.columns:
    cat_df = filtered.dropna(subset=['Category','Delivery_Time'])
    if not cat_df.empty:
        fig5 = px.box(cat_df, x='Category', y='Delivery_Time', points='outliers',
                      labels={'Delivery_Time':'Delivery Time (min)'}, title='Delivery Time distribution by Category')
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.write("No data for selected filters.")
else:
    st.write("Category column not present.")

# ---------- Optional visuals (collapsible) ----------
with st.expander("Optional visuals"):
    st.markdown("**Monthly trends (avg Delivery_Time by month)**")
    if 'Order_Datetime' in filtered.columns and filtered['Order_Datetime'].notna().any():
        filtered['Order_Month'] = filtered['Order_Datetime'].dt.to_period('M').astype(str)
        monthly = filtered.groupby('Order_Month')['Delivery_Time'].mean().reset_index()
        if not monthly.empty:
            fig_m = px.line(monthly, x='Order_Month', y='Delivery_Time', markers=True, title='Monthly avg Delivery Time')
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.write('No monthly data')
    else:
        st.write('Order datetime missing; cannot compute monthly trends.')

    st.markdown("**Delivery time distribution histogram**")
    if filtered['Delivery_Time'].dropna().shape[0] > 0:
        fig_hist = px.histogram(filtered, x='Delivery_Time', nbins=50, title='Delivery Time distribution')
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.write('No delivery time data')

    st.markdown("**% late deliveries by Traffic**")
    if 'Traffic' in filtered.columns:
        if np.isnan(threshold):
            st.write('Late threshold undefined; select a late delivery rule in the sidebar.')
        else:
            late_by_traffic = filtered.groupby('Traffic').apply(lambda g: (g['Delivery_Time'] > threshold).mean()*100).reset_index(name='% Late')
            fig_l = px.bar(late_by_traffic, x='Traffic', y='% Late', title='% Late deliveries by Traffic')
            st.plotly_chart(fig_l, use_container_width=True)
    else:
        st.write('No Traffic column')

    st.markdown("**Agent count per Area**")
    if 'Agent_Age' in filtered.columns and 'Area' in filtered.columns:
        agents_area = filtered.groupby('Area')['Order_ID'].nunique().reset_index(name='Agent_count_or_orders')
        fig_a = px.bar(agents_area.sort_values('Agent_count_or_orders', ascending=False), x='Area', y='Agent_count_or_orders', title='Orders (unique Order_ID) per Area')
        st.plotly_chart(fig_a, use_container_width=True)
    else:
        st.write('Agent or Area columns missing')

st.markdown("---")
st.caption("Dashboard created from the uploaded dataset. Adjust filters in the sidebar to update all charts.")
