import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset_path = r"C:\Users\lenovo\PycharmProjects\Ethereum_Analysis\DataSets\Ethereum_Dataset.csv"
real = pd.read_csv(dataset_path)

real['time'] = pd.to_datetime(real['time'])
real = real.sort_values('time')
prices = real['PriceUSD'].values

np.random.seed(42)
noise = np.random.normal(0, prices.std()*0.314, size=len([prices]))
synthetic_prices = prices + noise

synthetic = pd.DataFrame({
    'time': real['time'],
    'SyntheticPriceUSD': synthetic_prices
})

synthetic.to_csv(r"C:\Users\lenovo\PycharmProjects\Ethereum_Analysis\DataSets\Synthetic_Ethereum.csv", index=False)

# Plot comparison
plt.figure(figsize=(10,5))
plt.plot(real['time'], prices, label='Real Ethereum Prices', color='blue')
plt.plot(synthetic['time'], synthetic['SyntheticPriceUSD'], label='Synthetic Prices', color='orange', alpha=0.7)
plt.title('Real vs Synthetic Ethereum Price Data')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()