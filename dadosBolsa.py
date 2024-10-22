import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.dates as mdates
import warnings
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
#plt.figure(figsize=(16,6))
#plt.plot(prices['Date'], prices['ITUB'].ewm(span=90).mean())
#plt.plot(prices['Date'], prices['ITUB'], alpha=0.8)
#plt.plot(prices['Date'], prices['ITUB'].ewm(span=365).mean())
#plt.grid()
#plt.title('Cotações diárias e médias móveis de ITUB3', fontsize=15)
#plt.legend(['Média móvel trimestral', 'Cotação diária', 'Média móvel anual'])
#plt.show()

#2)Dados de candlestick itub3
# Baixar os dados
itub = pd.DataFrame()
itub = yf.download('ITUB3.SA', start='2024-03-01')
itub.index = pd.to_datetime(itub.index)

# Renaming columns to simpler names
itub.columns = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

# Converting index to datetime
itub.index = pd.to_datetime(itub.index)

# Creating DateNum column
itub['DateNum'] = mdates.date2num(itub.index)

# Criando uma figura
plt.figure(figsize = (16, 10))

# Definir espessura do candle e da sombra
esp_candle = .7
esp_sombra = .1

# Filtrar candles de alta e de baixa
alta = itub[itub['Close'] >= itub['Open']]
baixa = itub[itub['Close'] < itub['Open']]

# Escolher cores
cor_alta = 'green'
cor_baixa = 'red'

# Plotar candles de alta
plt.bar(alta['DateNum'], alta['Close'] - alta['Open'], esp_candle, bottom=alta['Open'], color=cor_alta)
plt.bar(alta['DateNum'], alta['High'] - alta['Close'], esp_sombra, bottom=alta['Close'], color=cor_alta)
plt.bar(alta['DateNum'], alta['Low'] - alta['Open'], esp_sombra, bottom=alta['Open'], color=cor_alta)

# Plotar candles de baixa
plt.bar(baixa['DateNum'], baixa['Close'] - baixa['Open'], esp_candle, bottom=baixa['Open'], color=cor_baixa)
plt.bar(baixa['DateNum'], baixa['High'] - baixa['Open'], esp_sombra, bottom=baixa['Open'], color=cor_baixa)
plt.bar(baixa['DateNum'], baixa['Low'] - baixa['Close'], esp_sombra, bottom=baixa['Close'], color=cor_baixa)

# Adicionar Médias Móveis
itub['Close'].rolling(window=7).mean().plot(label='MMS = 7')
itub['Close'].ewm(span=7).mean().plot(label='MME = 7')

# Ajustar os eixos de data
plt.gca().xaxis_date()
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

plt.legend()
plt.show()

# Correlation heatmap
sns.heatmap(prices.corr(), annot=True)
plt.show()

# 3) Retorno diário
returns = pd.DataFrame()
for i in tickers:
    returns[i] = prices[i].pct_change()
returns['Date'] = prices['Date']

sns.pairplot(returns)
plt.show()

returns.describe()

# Distribution plot for IBOV returns
sns.distplot(returns['IBOV'].dropna())

# 4) Retorno acumulado matplotlib 
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