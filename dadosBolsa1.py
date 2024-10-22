#from datatime import datatime
from MetaTrader5 import *
import MetaTrader5 as mt5
import pandas as pd

# Connect to MT5
if not mt5.initialize():
    print("initialize() failed, error code =", mt5.last_error())
    quit()
else:
    print("MetaTrader5 package version:", mt5.version())
    
# Request data
while True:
    rates = mt5.copy_rates_from_pos("PETR4", mt5.TIMEFRAME_D1, 0, 200)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = df.drop(df.columns[[5, 6, 7]], axis=1)
    clear_output(wait=True)
    print(df[-1:])
    sleep(0.1)