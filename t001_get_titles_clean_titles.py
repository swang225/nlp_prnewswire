from nebula.util.common import read_pickle
from nebula.nlp.nlp import NLPProcessor
import datetime as dt
import pandas as pd

# read in titles
res = read_pickle('code/prnews_title_dict.pkl')
total = len(res)
print(f"total: {total}")

nlpp = NLPProcessor()
titles = []
dates = []
count = 0
for k, v in res.items():

    if count % 10000 == 0:
        print(f"processed: {count} of {total}")
    count = count + 1

    if len(k) <= 0:
        continue

    earliest_date = dt.datetime.max
    for str_date in v['timestamps']:
        earliest_date = min(earliest_date, dt.datetime.strptime(str_date, '%Y%m%d%H%M%S'))

    title = nlpp.txtp.clean_digits(k)

    titles.append(title)
    dates.append(earliest_date)

res = pd.DataFrame(data={'date': dates, 'title': titles}).set_index('date')
res.to_pickle('prnews_title.pkl')
# creates a dataframe of date: title

