import requests
from nebula.util.common import write_pickle, read_pickle


def get_raw_list(base, page):
    # gets list of urls for a base url and the page number
    # there can be many pages for a base url

    URL = f"https://web.archive.org/cdx/search/xd?url={base}/*" \
          "&fl=timestamp,original&collapse=digest&gzip=false" \
          "&filter=statuscode:200" \
          f"&page={page}"

    r = requests.get(url=URL)
    res = r.text
    return res


def parse_raw_list(raw):
    # splits data for a page to list of timestamps and urls

    res = raw.split('\n')
    res = [r.split(' ') for r in res]

    return res


def parse_title(url):
    # parses title from the url of the article

    res = url.split('/')
    res = res[-1]

    res = res.split('.html')
    res = res[0]

    res = res.split('-')
    res = res[:-1]

    res = " ".join(res)
    res = " ".join(res.split())

    return res


from langdetect import detect_langs
def is_english(title):
    # checks if title is in english
    # not very accuracy given short titles

    res = detect_langs(title)

    for r in res:
        if r.lang == 'en':
            return True

    return False


def add_to_dict(raw, title_dict):
    # parses results for a page to title data

    raw_list = parse_raw_list(raw)
    for item in raw_list:
        if len(item) < 2:
            continue
        timestamp = item[0]
        url = item[1]
        title = parse_title(url)
        if title not in title_dict:
            title_dict[title] = dict(url=None, timestamps=set())
        if title_dict[title]['url'] is None:
              title_dict[title]['url'] = (timestamp, url)
        title_dict[title]['timestamps'].add(timestamp)


def get_titles(base):
    # loop through pages in wayback machine archive for the base url
    # get the list of urls from archive and parse title out of them
    # add to results dictionary

    title_dict = {}

    page = 0
    while True:
        print(f'Processing page: {page}')
        try:
            res = get_raw_list(base=base, page=page)
        except:
            print(f'invalid page: {page}')
            break

        if len(res) == 0:
            print(f'empty page: {page}')
            break

        add_to_dict(res, title_dict)
        page = page + 1

    return title_dict


res = get_titles(base='https://www.prnewswire.com/news-releases')
write_pickle(res, 'data/prnews_title_dict.pkl')

# prnews_title_dict description
# keys: article titles
# values: dict
# values-keys: url, timestamps
# values-url: tuple, (timestamp, url)
## note only kept one url
# values-timestamps: set, all timestamps for the title
