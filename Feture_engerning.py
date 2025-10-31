import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("no_na_Last_mile_Delivery_Data.csv")

# Convert datetime columns
df["Order_DateTime"] = pd.to_datetime(df["Order_Date"] + " " + df["Order_Time"])
df["Pickup_DateTime"] = pd.to_datetime(df["Order_Date"] + " " + df["Pickup_Time"])

# Calculate delays (in minutes)
df["pickup_delay_min"] = (df["Pickup_DateTime"] - df["Order_DateTime"]).dt.total_seconds() / 60
df["delivery_delay_min"] = df["Delivery_Time"] - df["pickup_delay_min"]

# Compute statistics for delivery time
mean_delivery = df["Delivery_Time"].mean()
std_delivery = df["Delivery_Time"].std()

# Define “late” deliveries (beyond mean + 1 std)
threshold = mean_delivery + std_delivery
df["is_late"] = df["Delivery_Time"] > threshold

# Create derived efficiency metrics
df["delivery_speed_index"] = df["Delivery_Time"] / df["pickup_delay_min"].replace(0, np.nan)
df["on_time_score"] = np.where(df["is_late"], 0, 1)

# Aggregate data for dashboard visuals
summary = df.groupby(["Weather", "Traffic", "Vehicle", "Area", "Category"]).agg(
    avg_delivery_time=("Delivery_Time", "mean"),
    avg_pickup_delay=("pickup_delay_min", "mean"),
    percent_late=("is_late", lambda x: 100 * x.mean()),
    deliveries=("Order_ID", "count")
).reset_index()

# Save results
df.to_csv("delivery_with_delay_features.csv", index=False)
summary.to_csv("delivery_summary_dashboard.csv", index=False)

print("✅ Feature engineering complete.")
print("Sample feature columns:", df[[
    "Order_ID", "pickup_delay_min", "delivery_delay_min", "is_late", "delivery_speed_index"
]].head())
print("\nDashboard summary preview:")
print(summary.head())
