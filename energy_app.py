import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime

st.set_page_config(page_title="Energy Forecast", layout="centered")
st.title("🔌 Daily Energy Consumption Tracker & Forecast")

st.write("Track your daily energy use and predict future consumption!")

# --- Manual Data Entry ---
st.subheader("Manual Entry")
num_days = st.number_input("How many days of data do you want to enter?", min_value=3, max_value=30, value=7)

dates = []
consumptions = []

for i in range(num_days):
    date = st.date_input(f"Date {i+1}", value=datetime.date.today() - datetime.timedelta(days=num_days - i), key=i)
    usage = st.number_input(f"Energy Consumption on {date} (kWh)", min_value=0.0, value=100.0, key=i+100)
    dates.append(date)
    consumptions.append(usage)

# --- Create DataFrame ---
df = pd.DataFrame({'Date': dates, 'Consumption': consumptions})
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# --- Display Data ---
st.subheader("Energy Data")
st.dataframe(df)

# --- Plotting ---
fig, ax = plt.subplots()
ax.plot(df['Date'], df['Consumption'], marker='o', linestyle='-')
ax.set_xlabel("Date")
ax.set_ylabel("Consumption (kWh)")
ax.set_title("Energy Consumption Over Time")
plt.xticks(rotation=45)
st.pyplot(fig)

# --- Forecasting ---
st.subheader("🔮 Forecast Future Consumption")

# Prepare data
df['Days'] = (df['Date'] - df['Date'].min()).dt.days
X = df[['Days']]
y = df['Consumption']

model = LinearRegression()
model.fit(X, y)

days_forward = st.slider("How many days ahead to predict?", min_value=1, max_value=14, value=3)
future_date = df['Date'].max() + datetime.timedelta(days=days_forward)
future_day_value = (future_date - df['Date'].min()).days

predicted_value = model.predict(np.array([[future_day_value]]))[0]
st.success(f"Estimated energy consumption on {future_date.strftime('%Y-%m-%d')}: **{predicted_value:.2f} kWh**")
