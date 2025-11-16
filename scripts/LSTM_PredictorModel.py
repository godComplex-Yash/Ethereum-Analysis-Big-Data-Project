import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

dataset_path = r"C:\Users\lenovo\PycharmProjects\Ethereum_Analysis\DataSets\Ethereum_Dataset.csv"
data = pd.read_csv(dataset_path)

data['time'] = pd.to_datetime(data['time'])
data = data.sort_values('time')
prices = data['PriceUSD'].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# (X = previous 60 days, y = next day)
sequence_length = 60
X, y = [], []

for i in range(sequence_length, len(scaled_prices)):
    X.append(scaled_prices[i-sequence_length:i, 0])
    y.append(scaled_prices[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # reshape for LSTM

# Split into train and test sets (80-20)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(100, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
print("Model summary:")
model.summary()

# train the model with set...
print("\n Training model (this may take 1–2 minutes)...")
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=1)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# pltted vs real
plt.figure(figsize=(10, 5))
plt.plot(real_prices, color='blue', label='Real Ethereum Prices')
plt.plot(predictions, color='red', label='Predicted Ethereum Prices')
plt.title('Ethereum Price Prediction (LSTM)')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
