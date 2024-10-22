import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import yfinance as yf
import plotly.graph_objects as go
import warnings
import mplfinance as mpf
warnings.filterwarnings("ignore")

# Fetching data using yfinance directly
prices = pd.DataFrame()
tickers = ['ITUB3.SA', 'BBDC3.SA', 'BBAS3.SA', 'SANB3.SA', '^BVSP']

for ticker in tickers:
    prices[ticker] = yf.download(ticker, start='2016-01-01')['Adj Close']
prices.head()

# Renaming columns for readability
prices.rename(columns={'ITUB3.SA':'ITUB', 'BBDC3.SA':'BBDC', 'BBAS3.SA':'BBAS', 'SANB3.SA':'SANB', '^BVSP':'IBOV'}, inplace=True)
prices['IBOV'] = prices['IBOV'] / 1000  # Scale IBOV index
prices.reset_index(inplace=True)
prices.dropna(subset=['IBOV'], inplace=True)
prices.IBOV.isnull().sum()

# 1) Cotação x tempo    
tickers = list(prices.drop(['Date'], axis=1).columns)
plt.figure(figsize=(16,6))

for i in tickers:
    plt.plot(prices['Date'], prices[i])
plt.legend(tickers)
plt.grid()
plt.title("Cotação x tempo", fontsize=25)
plt.show()

# ITUB3 moving averages
plt.figure(figsize=(16,6))
plt.plot(prices['Date'], prices['ITUB'].ewm(span=90).mean())
plt.plot(prices['Date'], prices['ITUB'], alpha=0.8)
plt.plot(prices['Date'], prices['ITUB'].ewm(span=365).mean())
plt.grid()
plt.title('Cotações diárias e médias móveis de ITUB3', fontsize=15)
plt.legend(['Média móvel trimestral', 'Cotação diária', 'Média móvel anual'])
plt.show()

# Correlation heatmap
sns.heatmap(prices.corr(), annot=True)
plt.show()

# 2) Retorno diário
returns = pd.DataFrame()
for i in tickers:
    returns[i] = prices[i].pct_change()
returns['Date'] = prices['Date']

sns.pairplot(returns)
plt.show()

returns.describe()

# Distribution plot for IBOV returns
sns.distplot(returns['IBOV'].dropna())

# 3) Retorno acumulado matplotlib 
return_sum = pd.DataFrame()
for ticker in tickers:
    return_sum[ticker] = (returns[ticker] + 1).cumprod()
return_sum['Date'] = returns['Date']

plt.figure(figsize=(16,6))
plt.plot(return_sum['Date'], return_sum.drop(['Date'], axis=1), alpha=0.9)
plt.legend(tickers)
plt.title("Retorno x tempo", fontsize=15)
plt.grid()
plt.show()
