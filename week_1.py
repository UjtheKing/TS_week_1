#%%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# %%

# Read the vic_elec.csv file
vic_elec = pd.read_csv('/Users/yudhajatmiko/Pacman/TS/Week 1/vic_elec.csv')

# If you want to quickly check the first few rows of the data, you can add:
print(vic_elec.head())

# Would you like me to add this code to your `week_1.py` file? If you want to parse dates or set an index column, let me know!
vic_elec
vic_elec.info
# %%

# Check column names
print(vic_elec.columns)

# Convert the 'Time' column to datetime
vic_elec['Time'] = pd.to_datetime(vic_elec['Time'])
vic_elec['Time']

# %%
vic_elec.set_index('Time', inplace=True)

# %%
# Analyze electricity demand trends over time
print("Dataset Info:")
print(f"Date range: {vic_elec.index.min()} to {vic_elec.index.max()}")
print(f"Total observations: {len(vic_elec)}")
print(f"Demand range: {vic_elec['Demand'].min():.2f} to {vic_elec['Demand'].max():.2f}")

# Create time-based features for analysis
vic_elec['Year'] = vic_elec.index.year
vic_elec['Month'] = vic_elec.index.month
vic_elec['Hour'] = vic_elec.index.hour
vic_elec['DayOfWeek'] = vic_elec.index.dayofweek

# %%
# 1. Electricity demand over time - Annual and monthly trends
plt.figure(figsize=(15, 10))

# Annual average demand
plt.subplot(2, 2, 1)
annual_demand = vic_elec.groupby('Year')['Demand'].mean()
plt.plot(annual_demand.index, annual_demand.values, marker='o')
plt.title('Average Annual Electricity Demand')
plt.xlabel('Year')
plt.ylabel('Demand (MW)')

# Monthly pattern
plt.subplot(2, 2, 2)
monthly_demand = vic_elec.groupby('Month')['Demand'].mean()
plt.plot(monthly_demand.index, monthly_demand.values, marker='o')
plt.title('Average Monthly Electricity Demand')
plt.xlabel('Month')
plt.ylabel('Demand (MW)')
plt.xticks(range(1, 13))

# Daily pattern
plt.subplot(2, 2, 3)
hourly_demand = vic_elec.groupby('Hour')['Demand'].mean()
plt.plot(hourly_demand.index, hourly_demand.values, marker='o')
plt.title('Average Hourly Electricity Demand')
plt.xlabel('Hour of Day')
plt.ylabel('Demand (MW)')

# Weekly pattern
plt.subplot(2, 2, 4)
weekly_demand = vic_elec.groupby('DayOfWeek')['Demand'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
plt.plot(range(7), weekly_demand.values, marker='o')
plt.title('Average Daily Electricity Demand')
plt.xlabel('Day of Week')
plt.ylabel('Demand (MW)')
plt.xticks(range(7), days)

plt.tight_layout()
plt.show()

# %%
# 2. Correlation between electricity demand and temperature
correlation = vic_elec['Demand'].corr(vic_elec['Temperature'])
print(f"\nCorrelation between Demand and Temperature: {correlation:.4f}")

plt.figure(figsize=(15, 5))

# Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(vic_elec['Temperature'], vic_elec['Demand'], alpha=0.1)
plt.xlabel('Temperature (°C)')
plt.ylabel('Demand (MW)')
plt.title(f'Demand vs Temperature (r = {correlation:.3f})')

# Add trend line
z = np.polyfit(vic_elec['Temperature'], vic_elec['Demand'], 1)
p = np.poly1d(z)
plt.plot(vic_elec['Temperature'].sort_values(), p(vic_elec['Temperature'].sort_values()), "r--", alpha=0.8)

# Temperature and demand over time (sample period)
plt.subplot(1, 2, 2)
sample_data = vic_elec['2012-01':'2012-03']
ax1 = plt.gca()
ax1.plot(sample_data.index, sample_data['Demand'], 'b-', label='Demand', alpha=0.7)
ax1.set_ylabel('Demand (MW)', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(sample_data.index, sample_data['Temperature'], 'r-', label='Temperature', alpha=0.7)
ax2.set_ylabel('Temperature (°C)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Demand and Temperature Over Time (Jan-Mar 2012)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %%
# 3. Seasonal patterns analysis
print("\nSeasonal Analysis:")

# Summer months (Dec, Jan, Feb)
summer_demand = vic_elec[vic_elec['Month'].isin([12, 1, 2])]['Demand'].mean()
# Winter months (Jun, Jul, Aug)  
winter_demand = vic_elec[vic_elec['Month'].isin([6, 7, 8])]['Demand'].mean()

print(f"Average Summer Demand: {summer_demand:.2f} MW")
print(f"Average Winter Demand: {winter_demand:.2f} MW")
print(f"Winter vs Summer difference: {winter_demand - summer_demand:.2f} MW ({((winter_demand - summer_demand)/summer_demand)*100:.1f}%)")

# Seasonal boxplot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
vic_elec['Season'] = vic_elec['Month'].map({12: 'Summer', 1: 'Summer', 2: 'Summer',
                                          3: 'Autumn', 4: 'Autumn', 5: 'Autumn',
                                          6: 'Winter', 7: 'Winter', 8: 'Winter',
                                          9: 'Spring', 10: 'Spring', 11: 'Spring'})

sns.boxplot(data=vic_elec, x='Season', y='Demand', order=['Summer', 'Autumn', 'Winter', 'Spring'])
plt.title('Seasonal Electricity Demand Distribution')
plt.ylabel('Demand (MW)')

# Temperature by season
plt.subplot(1, 2, 2)
sns.boxplot(data=vic_elec, x='Season', y='Temperature', order=['Summer', 'Autumn', 'Winter', 'Spring'])
plt.title('Seasonal Temperature Distribution')
plt.ylabel('Temperature (°C)')

plt.tight_layout()
plt.show()

# Summary statistics by season
print("\nSeasonal Summary Statistics:")
seasonal_stats = vic_elec.groupby('Season')[['Demand', 'Temperature']].agg(['mean', 'std', 'min', 'max'])
print(seasonal_stats)

# %%
