import streamlit as st
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import datetime
import numpy as np
import ta
import plotly.graph_objects as go

st.title('Global Stock Market Prediction App')

# Sidebar for user input
st.sidebar.header('User Input Parameters')

# Load list of stock tickers
@st.cache_data
def load_ticker_data():
    us_stocks = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    indian_stocks = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'HINDUNILVR.NS']
    return us_stocks, indian_stocks

us_stocks, indian_stocks = load_ticker_data()

market = st.sidebar.selectbox('Select Market', ('U.S. Market', 'Indian Market', 'Both'))
if market == 'U.S. Market':
    available_stocks = us_stocks
elif market == 'Indian Market':
    available_stocks = indian_stocks
else:
    available_stocks = us_stocks + indian_stocks

all_option = st.sidebar.checkbox('Use All Available Stocks')
if all_option:
    selected_stocks = available_stocks
else:
    selected_stocks = st.sidebar.multiselect('Select Stocks', available_stocks, default=available_stocks[0])

# Error handling when no stock is selected
if not selected_stocks:
    st.error("Please select at least one stock.")
    st.stop()

start_date = st.sidebar.date_input('Start Date', datetime.date(2015, 1, 1))
end_date = st.sidebar.date_input('End Date', datetime.date.today())

# Fetch data
@st.cache_data
def fetch_data(tickers, start, end):
    data_list = []
    for ticker in tickers:
        data = yf.download(ticker, start=start, end=end)
        if not data.empty:
            data['Ticker'] = ticker
            data_list.append(data)
    if data_list:
        combined_data = pd.concat(data_list)
        return combined_data.reset_index()
    else:
        return pd.DataFrame()

try:
    data = fetch_data(selected_stocks, start_date, end_date)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Check if data is available
if data.empty:
    st.error('No data available for the selected date range.')
    st.stop()
else:
    # Display data
    st.subheader('Historical Data')
    st.write(data.tail())

    # Preprocessing
    data['Date_ordinal'] = data['Date'].map(datetime.datetime.toordinal)
    data = data[['Date', 'Date_ordinal', 'Close', 'High', 'Low', 'Open', 'Volume', 'Ticker']]

    # Calculate Technical Indicators
    data.sort_values(['Ticker', 'Date'], inplace=True)

    # Moving Averages
    data['SMA_14'] = data.groupby('Ticker')['Close'].transform(lambda x: x.rolling(window=14).mean())
    data['EMA_14'] = data.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=14, adjust=False).mean())

    # RSI
    data['RSI_14'] = data.groupby('Ticker')['Close'].transform(
        lambda x: ta.momentum.RSIIndicator(close=x, window=14).rsi()
    )

    # MACD
    data['MACD'] = data.groupby('Ticker')['Close'].transform(
        lambda x: ta.trend.MACD(close=x).macd()
    )

    # Handle NaN values resulting from calculations
    data.fillna(method='bfill', inplace=True)
    data.fillna(method='ffill', inplace=True)

    # Show statistical summary
    st.subheader('Statistical Summary')
    stats = data.groupby('Ticker')['Close'].describe()
    st.write(stats)

    # Encode the 'Ticker' column
    le = LabelEncoder()
    data['Ticker_encoded'] = le.fit_transform(data['Ticker'])

    # Prepare features and target
    X = data[['Date_ordinal', 'Ticker_encoded', 'SMA_14', 'EMA_14', 'RSI_14', 'MACD']]
    y = data['Close']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Adjusted Future Predictions
    future_dates = pd.date_range(
        end_date + datetime.timedelta(1), periods=30, freq='B'
    )
    future_data_list = []
    for ticker in selected_stocks:
        ticker_data = data[data['Ticker'] == ticker]
        last_known_indicators = ticker_data.iloc[-1][['SMA_14', 'EMA_14', 'RSI_14', 'MACD']]
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Date_ordinal': future_dates.map(datetime.datetime.toordinal),
            'Ticker': ticker
        })
        future_df['Ticker_encoded'] = le.transform([ticker]*len(future_df))
        # Repeat the last known indicators
        for col in ['SMA_14', 'EMA_14', 'RSI_14', 'MACD']:
            future_df[col] = last_known_indicators[col]
        future_X = future_df[['Date_ordinal', 'Ticker_encoded', 'SMA_14', 'EMA_14', 'RSI_14', 'MACD']]
        future_pred = model.predict(future_X)
        future_df['Predicted_Close'] = future_pred
        future_data_list.append(future_df)
    future_combined = pd.concat(future_data_list)

    # Display future predictions
    st.subheader('Predicted Stock Prices for Next 30 Days')
    st.write(future_combined[['Date', 'Ticker', 'Predicted_Close']])

    # Plotting
    st.subheader('Closing Price vs Time Chart')

    fig = go.Figure()

    # Historical data
    for ticker in selected_stocks:
        ticker_data = data[data['Ticker'] == ticker]
        fig.add_trace(go.Scatter(
            x=ticker_data['Date'], y=ticker_data['Close'],
            mode='lines', name=f'{ticker} Historical'
        ))

    # Future predictions
    for ticker in selected_stocks:
        ticker_future = future_combined[future_combined['Ticker'] == ticker]
        fig.add_trace(go.Scatter(
            x=ticker_future['Date'], y=ticker_future['Predicted_Close'],
            mode='lines', name=f'{ticker} Predicted', line=dict(dash='dash')
        ))

    fig.update_layout(
        title='Closing Price vs Time',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        hovermode='x'
    )

    st.plotly_chart(fig)
