import pandas as pd
import matplotlib.pyplot as plt

dataset_path = r"C:\Users\lenovo\PycharmProjects\Ethereum_Analysis\DataSets\Ethereum_Dataset.csv"
data = pd.read_csv(dataset_path)

data['time'] = pd.to_datetime(data['time'])

data = data.sort_values('time')

# Plot
plt.figure(figsize=(10, 5))
plt.plot(data['time'], data['PriceUSD'], color='blue', label='Ethereum Price (USD)')
plt.title('Real Ethereum Price Trend')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
