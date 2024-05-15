import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

ticker = input("Please input a ticker symbol:\n").upper()

def get_stock_data(ticker):
    # Fetch historical stock prices using Yahoo Finance
    try:
        stock_data = yf.download(ticker, start="2022-05-08", end="2024-05-08")  # Adjust the start date for the last 2 years
    except Exception as e:
        print(f"Error: Unable to fetch data for {ticker} from Yahoo Finance.")
        print(e)
        return None
    
    if stock_data.empty:
        print(f"Error: No data available for {ticker}.")
        return None
    
    # Calculate MACD
    stock_data['12_day_ema'] = stock_data['Close'].ewm(span=12, min_periods=12).mean()
    stock_data['26_day_ema'] = stock_data['Close'].ewm(span=26, min_periods=26).mean()
    stock_data['MACD'] = stock_data['12_day_ema'] - stock_data['26_day_ema']
    stock_data['MACD_signal'] = stock_data['MACD'].ewm(span=9, min_periods=9).mean()
    
    # Calculate RSI
    delta = stock_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    stock_data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate volume profile
    stock_data['Volume Profile'] = (stock_data['Volume'] - stock_data['Volume'].min()) / (stock_data['Volume'].max() - stock_data['Volume'].min())
    
    # Drop any rows with null values
    stock_data.dropna(inplace=True)
    
    return stock_data

stock_df = get_stock_data(ticker)
if stock_df is not None:
    print(stock_df.tail())

# Extract relevant features (excluding 'Close' and 'Adj Close')
features = ['Open', 'High', 'Low', 'Volume', '26_day_ema', 'MACD', 'MACD_signal', 'RSI', 'Volume Profile']
data = stock_df[features].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data[:, :])  # All features excluding the last column (Adj Close)
data_scaled = scaler.transform(data[:, :])  # All features excluding the last column (Adj Close)

target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(data[:, 4].reshape(-1, 1))  # Assuming 'Adj Close' price is the fifth column
target_scaled = target_scaler.transform(data[:, 4].reshape(-1, 1))

# Define sequence length
sequence_length = 30  # Use the last 30 days' data to predict the next day's price

# Create sequences and targets
X, y = [], []
for i in range(len(data_scaled) - sequence_length - 5):
    X.append(data_scaled[i:i+sequence_length])
    y.append(target_scaled[i+sequence_length:i+sequence_length+5])

X, y = np.array(X), np.array(y)

# Train-test split
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=5)  # Output layer predicts the next 5 Adj Close prices
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Example usage:
last_30_days = stock_df[-30:]
last_30_days_subset = last_30_days[features]  # Exclude the last column (Adj Close)
print(last_30_days_subset.head())

# Preprocess the data
last_30_days_scaled = scaler.transform(last_30_days_subset)

# Reshape the data
X_pred = last_30_days_scaled.reshape(1, 30, len(features))  # Exclude the last column (Adj Close)

predictions_scaled = model.predict(X_pred)

# Inverse scale the predictions
predictions = target_scaler.inverse_transform(predictions_scaled)

print("Predicted next 5 Adjusted Close prices:")
print(predictions)

# Average of predicted prices
average_pred = np.mean(predictions)

# Previous day's Adj Close
previous_close = last_30_days.iloc[-1]['Adj Close']

# Predict whether the average of predicted prices is higher or lower than the previous day's Adj Close
if average_pred > previous_close:
    print("Prediction: Higher")
else:
    print("Prediction: Lower")
