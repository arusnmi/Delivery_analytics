import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("no_na_Last_mile_Delivery_Data.csv")

# Compute average delivery time by Weather and Traffic
delay_summary = (
    df.groupby(["Weather", "Traffic"])["Delivery_Time"]
      .mean()
      .reset_index()
)

# Get list of unique weather conditions
weather_conditions = delay_summary["Weather"].unique()

# Set up subplots — one pie per weather type
fig, axes = plt.subplots(
    nrows=1, 
    ncols=len(weather_conditions), 
    figsize=(6 * len(weather_conditions), 6)
)

# Ensure axes is iterable even if there’s only one weather type
if len(weather_conditions) == 1:
    axes = [axes]

# Plot one pie per weather condition
for ax, weather in zip(axes, weather_conditions):
    subset = delay_summary[delay_summary["Weather"] == weather]
    ax.pie(
        subset["Delivery_Time"],
        labels=subset["Traffic"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white"},
    )
    ax.set_title(f"{weather} Weather", fontsize=14)

plt.suptitle("Traffic Distribution of Average Delivery Time under Each Weather Condition", fontsize=16)
plt.tight_layout()
plt.show()
