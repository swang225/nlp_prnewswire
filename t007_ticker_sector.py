import pandas as pd

df1 = pd.read_csv("code/test001_nlp/sp500gics.csv")
sp500_gics = df1[['Symbol', 'GICS Sector']].set_index('Symbol')
sp500_gics.to_pickle('sp500_gics.pkl')

# ticker:sector, dataframe
