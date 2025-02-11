import pandas as pd
import yfinance as yf
import plotly.express as px
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge

# Classe para baixar e organizar os dados de ações
class StockData:
    def __init__(self, tickers, start_date='2019-01-01'):
        self.tickers = tickers
        self.start_date = start_date
        self.prices = None
        self.volume = None
        self._download_data()
        self._clean_data()
    
    def _download_data(self):
        print("Baixando os dados das ações...")
        data = yf.download(self.tickers, start=self.start_date)
        self.prices = data['Close'].copy()
        self.volume = data['Volume'].copy()
        self._rename_columns()
    
    def _rename_columns(self):
        rename_dict = {'ITUB3.SA': 'ITUB', 'BBDC3.SA': 'BBDC', 'VBBR3.SA': 'VBBR', 'SANB3.SA': 'SANB', '^BVSP': 'IBOV'}
        self.prices.rename(columns=rename_dict, inplace=True)
        self.volume.rename(columns=rename_dict, inplace=True)
    
    def _clean_data(self):
        print("Limpando os dados...")
        self.prices.loc[:, 'IBOV'] /= 1000
        self.volume.loc[:, 'IBOV'] /= 1000
        self.prices.reset_index(inplace=True)
        self.volume.reset_index(inplace=True)
        self.prices.ffill(inplace=True)
        self.volume.ffill(inplace=True)
    
    def get_stock(self, name):
        df = pd.DataFrame({'Date': self.prices['Date'],
                           'Close': self.prices[name],
                           'Volume': self.volume[name]})
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        return df.dropna()

# Classe para preparar os dados para o modelo
class Preprocessor:
    scaler = MinMaxScaler(feature_range=(0,1))
    
    def trading_window(data, n=1):
        print("Criando a variável de previsão...")
        data['Target'] = data[['Close']].shift(-n)
        return data.dropna()
    
    def normalize_data(data):
        print("Normalizando os dados...")
        scaled_values = Preprocessor.scaler.fit_transform(data.drop(columns=['Date']))
        return pd.DataFrame(scaled_values, columns=['Close', 'Volume', 'MA_5', 'MA_10', 'Target'])
    
    def split_data(scaled_df, train_size=0.7):
        print("Separando os dados em treino e teste...")
        X = scaled_df.iloc[:, :-1].values  # Features: Close, Volume, MA_5, MA_10
        y = scaled_df.iloc[:, -1].values  # Target: Future Close Price
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reformatando para LSTM
        return train_test_split(X, y, test_size=0.3, shuffle=False)

# Definição dos tickers
tickers = ['ITUB3.SA', 'BBDC3.SA', 'VBBR3.SA', 'SANB3.SA', '^BVSP']

# Inicializando classes
data_handler = StockData(tickers)
price_volume_df = data_handler.get_stock('VBBR')

# Pré-processamento
target_df = Preprocessor.trading_window(price_volume_df)
scaled_df = Preprocessor.normalize_data(target_df)
X_train, X_test, y_train, y_test = Preprocessor.split_data(scaled_df)

# Criando e treinando modelo LSTM
print("Criando modelo LSTM...")
inputs = keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
x = keras.layers.LSTM(150, return_sequences=True)(inputs)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.LSTM(150, return_sequences=True)(x)
x = keras.layers.Dropout(0.2)(x)
x = keras.layers.LSTM(150)(x)
outputs = keras.layers.Dense(1, activation='linear')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse')
model.summary()

print("Treinando modelo LSTM...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2
)

# Fazendo previsões
y_pred_lstm = model.predict(X_test)

# Desnormalizando previsões
y_test_original = Preprocessor.scaler.inverse_transform(
    np.column_stack([y_test, np.zeros((y_test.shape[0], 4))]))[:, 0]
predicted_price_original = Preprocessor.scaler.inverse_transform(
    np.column_stack([y_pred_lstm.flatten(), np.zeros((y_pred_lstm.shape[0], 4))]))[:, 0]

# Calculando o Erro Médio Absoluto (MAE)
lstm_mae = mean_absolute_error(y_test_original, predicted_price_original)
print(f'LSTM Model MAE: {lstm_mae}')

# Criando DataFrame com previsões
df_predicted = target_df.iloc[len(target_df) - len(X_test):].copy()
df_predicted['Prediction'] = predicted_price_original

# Criando modelo de regressão
ridge_model = Ridge()
ridge_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
y_pred_ridge = ridge_model.predict(X_test.reshape(X_test.shape[0], -1))

# Desnormalizando previsões da regressão
ridge_pred_original = Preprocessor.scaler.inverse_transform(
    np.column_stack([y_pred_ridge, np.zeros((y_pred_ridge.shape[0], 4))]))[:, 0]
ridge_mae = mean_absolute_error(y_test_original, ridge_pred_original)
print(f'Ridge Regression MAE: {ridge_mae}')

df_predicted['Ridge Prediction'] = ridge_pred_original

# Plotando previsões com gráfico interativo
def interactive_plot(data, title, y_columns):
    fig = px.line(data, x='Date', y=y_columns, labels={'value': 'Preço', 'Date': 'Data'},
                  title=title)
    fig.update_layout(hovermode='x unified')
    fig.show()

interactive_plot(df_predicted, 'Original Price vs. LSTM Predictions', ['Close', 'Prediction'])
interactive_plot(df_predicted, 'Original Price vs. Ridge Regression Predictions', ['Close', 'Ridge Prediction'])
