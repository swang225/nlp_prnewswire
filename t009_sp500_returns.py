import pandas as pd
import yfinance as yf

tickers = pd.read_csv('code/test002_tickers/sp500gics.csv')['Symbol'].values

history = []
for ticker in tickers:
    print(f'processing {ticker}')

    yt = yf.Ticker(ticker)

    res = yt.history(period='max')['Close']
    res.name = ticker
    res = pd.DataFrame(data=res)

    history.append(res)

prices = pd.concat(history, axis=1)
prices.to_pickle('sp500_prices.pkl')
#prices = pd.read_pickle('sp500_prices.pkl')

def daily_returns(prices):

    res = (prices/prices.shift(1) - 1.0)[1:]

    return res


dreturns = daily_returns(prices)
dreturns.to_pickle('sp500_daily_returns.pkl')
#dreturns = pd.read_pickle('sp500_daily_returns.pkl')


def cumulative_returns(returns):

    res = (returns + 1.0).cumprod()

    return res

creturns = cumulative_returns(dreturns)
creturns.to_pickle('sp500_cum_returns.pkl')

# files with notebooks
