import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
import seaborn as sns
import matplotlib.pyplot as plt

# Classe para baixar e organizar os dados de ações
class StockData:
    def __init__(self, tickers, start_date='2016-01-01'):
        self.tickers = tickers
        self.start_date = start_date
        self.prices = None
        self.volume = None
        self.data = None
        self._download_data()
        self._clean_data()
        self._plot_candlestick()
        self._plot_correlation()
        self._plot_stocks_vs_ibov()
    
    def _download_data(self):
        """Baixa os dados das ações do Yahoo Finance."""
        print("Baixando os dados das ações...")
        self.data = yf.download(self.tickers, start=self.start_date)
        self.prices = self.data['Close'].copy()
        self.volume = self.data['Volume'].copy()
        self._rename_columns()
    
    def _rename_columns(self):
        """Renomeia as colunas para facilitar o uso."""
        rename_dict = {'ITUB3.SA': 'ITUB', 'BBDC3.SA': 'BBDC', 'BBAS3.SA': 'BBAS', 'SANB3.SA': 'SANB', '^BVSP': 'IBOV'}
        self.prices.rename(columns=rename_dict, inplace=True)
        self.volume.rename(columns=rename_dict, inplace=True)
    
    def _clean_data(self):
        """Realiza o tratamento dos dados, preenchendo valores ausentes e ajustando a escala."""
        print("Limpando os dados...")
        self.prices.loc[:, 'IBOV'] /= 1000  # Ajusta a escala do índice IBOV
        self.volume.loc[:, 'IBOV'] /= 1000
        self.prices.reset_index(inplace=True)
        self.volume.reset_index(inplace=True)
        self.prices.ffill(inplace=True)  # Preenchimento de valores ausentes
        self.volume.ffill(inplace=True)
    
    def get_stock(self, name):
        """Retorna um DataFrame com os preços e volume da ação selecionada."""
        df = pd.DataFrame({'Date': self.prices['Date'],
                           'Close': self.prices[name],
                           'Volume': self.volume[name]})
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        return df.dropna()
    
    def _plot_candlestick(self):
        """Plota gráfico de velas (Candlestick) para visualizar a cotação ao longo do tempo."""
        fig = go.Figure()
        for ticker in self.tickers:
            if ticker in self.data.columns.levels[1]:
                df = self.data.xs(ticker, level=1, axis=1)
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name=f'Candlestick {ticker}')
                )
        fig.update_layout(title='Cotação por Tempo - Candlestick',
                          xaxis_title='Data',
                          yaxis_title='Preço',
                          xaxis_rangeslider_visible=False)
        fig.show()
    
    def _plot_correlation(self):
        """Plota um heatmap de correlação entre as ações e o IBOV."""
        correlation_matrix = self.prices.corr()
        plt.figure(figsize=(10,6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlação entre ações e IBOV')
        plt.show()
    
    def _plot_stocks_vs_ibov(self):
        """Plota a evolução das ações e do IBOV no mesmo gráfico."""
        fig = px.line(self.prices, x=self.prices.index, y=self.prices.columns, 
                      labels={'value': 'Preço', 'index': 'Data'},
                      title='Evolução das Ações vs IBOV')
        fig.update_layout(hovermode='x unified')
        fig.show()

# Definição dos tickers
tickers = ['ITUB3.SA', 'BBDC3.SA', 'BBAS3.SA', 'SANB3.SA', '^BVSP']

# Inicializando classes
data_handler = StockData(tickers)
price_volume_df = data_handler.get_stock('ITUB')
