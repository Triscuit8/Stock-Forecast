import yfinance as yf
import streamlit as st
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from prophet import Prophet
#Utilized ChatGpt to help with syntax errors and removing timezone from date
#Utilized the Prophet forecasting model documentation
#Utilized Yahoo Finance API documentation
#Utilized Streamlit framework documentation
#Utilized UBC CPSC 330 lectures for reference, I used to refamiliarize myself with machine learning

st.title("Stock Forecast Application")

# Makes a slider listing out all the stocks that are available to see
stocks = ("AAPL", "ABNB", "AMZN", "GOOGL", "META", "MSFT", "NFLX",  "TSLA")
selected_stock = st.selectbox("Select dataset for prediction", stocks)
# year = st.slider("Year of prediction:", 0, 10)
year = st.number_input('Number of years from now')
year = int(year)

tickerData = yf.Ticker(selected_stock)

endDate = date.today().strftime("%Y-%m-%d")

# Searches up the history of the stock
tickerDf = tickerData.history(period='1d', start='2010-1-1', end = endDate)
tickerDf.reset_index(inplace=True)

#Plots raw data
st.subheader("Raw Data")
st.write(tickerDf.tail())
st.line_chart(tickerDf.Close)

#We only need the Date and Close column for the forecasting function
df_train = tickerDf[['Date', 'Close']]

#Renames Date column to ds and Close column to y
m = Prophet()
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})
df_train['ds'] = df_train['ds'].dt.tz_localize(None)
m.fit(df_train)

#Makes a future prediction using Prophet library, parameter takes in days
future = m.make_future_dataframe(periods=year * 365)
forecast = m.predict(future)

st.subheader("Prediction Data")
st.write(forecast.tail())

#Plots the forecasted data
fig = m.plot(forecast)
st.pyplot(fig)


