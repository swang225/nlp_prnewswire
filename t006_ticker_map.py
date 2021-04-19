from nebula.util.common import read_pickle, write_pickle
import pandas as pd
from fuzzywuzzy import fuzz


df1 = pd.read_csv("code/test001_nlp/sp500gics.csv")
tickers = [(' '.join(ls).upper(), ls[0]) for ls in list(df1[['Symbol', 'Security']].values)]

def get_matches(ticker):

    match = []
    for test in tickers:

        score = fuzz.partial_ratio(test[0], ticker)

        if score > 80:
            match.append(test[1])

    return match


def create_ticker_map(subjects):

    ticker_map = {}
    total = len(subjects)
    count = 0
    success = 0
    failure = 0
    for subject in subjects:
        count = count + 1
        print(f"processing {count} out of {total}")
        ticker_map[subject] = get_matches(subject)

        if len(ticker_map[subject]) > 0:
            success = success + 1
        else:
            failure = failure + 1
        print(f"subject: {subject}, match: {ticker_map[subject]}")

        print(f"stat success {success}, failure {failure}")

    return ticker_map


org_map = read_pickle("code/test003_prnewswire_v01/data/org_map_complete.pkl")

orgs = set(org_map.values())

ticker_map = create_ticker_map(orgs)
write_pickle(ticker_map, "code/test003_prnewswire_v01/data/ticker_map.pkl")

# ticker_map is a dictionary of org: ticker
res = read_pickle("code/test003_prnewswire_v01/data/ticker_map.pkl")
