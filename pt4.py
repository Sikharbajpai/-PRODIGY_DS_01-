import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

# Load dataset
df = pd.read_csv('US_Accidents_Dec21_updated.csv')  # Or use your dataset path
df['Start_Time'] = pd.to_datetime(df['Start_Time'])

# Extract hour from time
df['Hour'] = df['Start_Time'].dt.hour
df['Day'] = df['Start_Time'].dt.dayofweek  # 0=Monday

# --- 1. Time of Day Analysis ---
plt.figure(figsize=(10, 5))
sns.histplot(df['Hour'], bins=24, kde=False, color='tomato')
plt.title('Accidents by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Number of Accidents')
plt.show()

# --- 2. Weather Conditions Analysis ---
top_weather = df['Weather_Condition'].value_counts().nlargest(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top_weather.index, y=top_weather.values, palette='coolwarm')
plt.title('Top 10 Weather Conditions during Accidents')
plt.xticks(rotation=45)
plt.ylabel('Number of Accidents')
plt.show()

# --- 3. Road Conditions (if available) ---
if 'Road_Condition' in df.columns:
    top_road = df['Road_Condition'].value_counts().nlargest(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_road.index, y=top_road.values, palette='magma')
    plt.title('Accidents by Road Condition')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Accidents')
    plt.show()

# --- 4. Accident Hotspots (Using HeatMap) ---
# Use a sample to reduce rendering time
map_df = df[['Start_Lat', 'Start_Lng']].dropna().sample(n=1000)

m = folium.Map(location=[map_df['Start_Lat'].mean(), map_df['Start_Lng'].mean()], zoom_start=6)
heat_data = [[row['Start_Lat'], row['Start_Lng']] for index, row in map_df.iterrows()]
HeatMap(heat_data).add_to(m)
m.save('accident_hotspots_map.html')
print("Heatmap saved as 'accident_hotspots_map.html'.")

# --- 5. Correlation Between Weather and Accidents ---
weather_hour = df.groupby(['Weather_Condition', 'Hour']).size().unstack(fill_value=0)
plt.figure(figsize=(14, 6))
sns.heatmap(weather_hour.head(10), cmap='Reds')
plt.title('Accidents by Weather and Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Weather Condition')
plt.show()
