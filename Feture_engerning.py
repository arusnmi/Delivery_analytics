import pandas as pd

# Load the data
df = pd.read_csv("no_na_Last_mile_Delivery_Data.csv")

# Combine Order_Date and Order_Time into datetime
df["Order_Datetime"] = pd.to_datetime(df["Order_Date"] + " " + df["Order_Time"])
df["Pickup_Datetime"] = pd.to_datetime(df["Order_Date"] + " " + df["Pickup_Time"])

# --- TIME-BASED FEATURES ---
# Time difference between order and pickup
df["Order_to_Pickup_Minutes"] = (df["Pickup_Datetime"] - df["Order_Datetime"]).dt.total_seconds() / 60

# Extract time-of-day features
df["Order_Hour"] = df["Order_Datetime"].dt.hour
df["Pickup_Hour"] = df["Pickup_Datetime"].dt.hour
df["Order_DayOfWeek"] = df["Order_Datetime"].dt.dayofweek  # Monday=0
df["Is_Weekend"] = df["Order_DayOfWeek"].isin([5, 6]).astype(int)

# --- DELAY-RELATED FEATURES ---
# Delay ratio (pickup time vs. total delivery time)
df["Pickup_to_Delivery_Ratio"] = df["Order_to_Pickup_Minutes"] / df["Delivery_Time"]

# Binary flag: is the pickup delay unusually high?
pickup_threshold = df["Order_to_Pickup_Minutes"].quantile(0.75)
df["Is_Pickup_Delayed"] = (df["Order_to_Pickup_Minutes"] > pickup_threshold).astype(int)

# Time bins (rush hours, etc.)
def time_bin(hour):
    if 7 <= hour < 10:
        return "Morning_Rush"
    elif 10 <= hour < 17:
        return "Daytime"
    elif 17 <= hour < 21:
        return "Evening_Rush"
    else:
        return "Night"

df["Order_Time_Bin"] = df["Order_Hour"].apply(time_bin)
df["Pickup_Time_Bin"] = df["Pickup_Hour"].apply(time_bin)

# --- OPTIONAL: One-hot encode categorical columns ---
df = pd.get_dummies(df, columns=["Traffic", "Area", "Category", "Order_Time_Bin", "Pickup_Time_Bin"], drop_first=True)

# Final feature dataframe
delay_features = df[[
    "Order_ID", "Agent_Age", "Agent_Rating", "Order_to_Pickup_Minutes",
    "Order_Hour", "Pickup_Hour", "Order_DayOfWeek", "Is_Weekend",
    "Pickup_to_Delivery_Ratio", "Is_Pickup_Delayed"
] + [col for col in df.columns if col.startswith(("Traffic_", "Area_", "Category_", "Order_Time_Bin_", "Pickup_Time_Bin_"))]]

print(delay_features.head())


# Save the features to a new CSV
delay_features.to_csv("delay_based_features.csv", index=False)
