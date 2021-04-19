# nlp_prnewswire
Trading Strategy from historical data for PRNewswire from Waybackmachine


NLP on PRNewswire

# This is a first version of the codes used to extract historical data from wayback machine
# for PRNewswire website to construct a trading strategy for SP500

# many algorithms are a rudmentary version that needs improvement, namely, ticker mapping,
# vector clustering, etc.


1. download data
    1. t001_get_titles.py (prnews_title_dict.pkl)
        1. t001_clean_titles.py (prnews_title.pkl)
    2. t002_get_articles_multi.py (need update)
2. svo, ner, wv(average, concat) from titles
    1. t003_wv_average_remote.py (svo_wv_final.pkl)
    2. t004_wv_separate_remote.py (svo_sep_complete.pkl)
    3. t005_svo_org_ner_multi.py (subjects.pkl, org_map_complete.pkl)
    4. t008_stanford_to_gensim.py (using wv_100000.model)
3. ticker map from wiki
    1. t006_ticker_map.py (ticker_map.pkl)
    2. t007_ticker_sector.py (sp500_gics.pkl)
4. sp500 prices
    1. t009_sp500_returns.py(sp500_daily_returns.pkl, sp500_cum_returns.pkl)
5. kmean clusters
    1. n001_nlp_svo_average.ipynb
    2. aws0008_sep_kmean.ipynb (wv_sep_clustered.pkl is clustered df, didn’t scp down the kmean model)
    3. 0007_nlp_svo_sep.ipynb

TODO:
    1. clean up the data more
    2. use all words from word2vec model, use the other bigger model as well
    3. use rolling kmean clustering, potential implement a new one
    4. download all article contents
    5. do analysis on article content with tfidf matrix
    6. look at other archived data in wayback machine

# Also please note the nlp library used is in folder: nebula/util, which is actually a separate repo copied over for demonstration purposes.
# The actual model used for training/predictions are not included in this repo, since the sizes are too large.
