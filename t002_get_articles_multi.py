import pickle
import requests
from html.parser import HTMLParser
from langdetect import detect_langs
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from nltk.tokenize import word_tokenize, sent_tokenize

import multiprocessing as mp


# ----------------------

def read_pickle(file):
    with open(file, 'rb') as handle:
        b = pickle.load(handle)

    return b


def write_pickle(data, file):
    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


class TokenProcessor:

    def __init__(self):
        self._stemmer = None
        self._stop_words = None
        self._lemmatizer = None
        self._ner_tagger = None
        self._pos_tagger = None

    @property
    def stemmer(self):
        if self._stemmer is None:
            self._stemmer = PorterStemmer()
        return self._stemmer

    def stem(self, words):
        stemmer = self.stemmer

        res = []
        for w in words:
            rw = stemmer.stem(w)
            res.append(rw)

        return res

    @property
    def stop_words(self):
        if self._stop_words is None:
            self._stop_words = set(stopwords.words('english'))
        return self._stop_words

    def clean_stop_words(self, words):
        stop_words = self.stop_words

        res = []
        for w in words:
            w_test = w[0] if isinstance(w, tuple) else w
            if w_test.lower() not in stop_words:
                res.append(w)

        return res

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            self._lemmatizer = WordNetLemmatizer()

        return self._lemmatizer

    def lemmatize(self, words):
        lemmatizer = self.lemmatizer

        res = []
        for w in words:
            word = lemmatizer.lemmatize(w.lower())
            res.append(word)

        return res

    @staticmethod
    def clean_digits(words):
        return [w for w in words if not w.isdigit()]

    @staticmethod
    def upper(words):
        return [w.upper() for w in words]

    @staticmethod
    def lower(words):
        return [w.lower() for w in words]

    @staticmethod
    def sentence(words):
        return " ".join(words)


class TextProcessor:

    def __init__(self):
        pass

    @staticmethod
    def clean_punctuations(text):
        return text.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def to_words(text):
        return word_tokenize(text)

    @staticmethod
    def to_sentences(text):
        return sent_tokenize(text)


class NLPProcessor:

    def __init__(self, text_kwargs={}, token_kwargs={}):
        self.txtp = TextProcessor(**text_kwargs)
        self.tokp = TokenProcessor(**token_kwargs)

    def clean_words(self, sentence):
        txtp = self.txtp
        tokp = self.tokp

        res = txtp.clean_punctuations(sentence)

        words = txtp.to_words(res)
        words = tokp.clean_digits(words)
        words = tokp.lemmatize(words)
        words = tokp.upper(words)
        words = tokp.clean_stop_words(words)

        return words

    def clean_sentence(self, sentence):
        words = self.clean_words(sentence)

        return self.tokp.sentence(words)

# ----------------------


def is_english(title):
    # checks if article is english
    # not very accuracy given short titles

    res = detect_langs(title.lower())

    for r in res:
        if r.lang == 'en':
            return True

    return False


def get_archive_article(url, timestamp):
    # get article from wayback machine archive

    archive_url = f'https://web.archive.org/web/{timestamp}id_/{url}'

    # print(archive_url)

    r = requests.get(url=archive_url)

    res = r.text

    return res


class PRNewsHTMLParser(HTMLParser):
    # parses articles to get content

    def __init__(self):
        self.in_set = set()
        self.in_p = False
        self.no_parse_date = False
        self.levels = 0

        self.title = ""
        self.body = ""
        self.date = ""
        self.date_publish = ""
        self.date_create = ""
        self.date_modify = ""
        super().__init__()

    @property
    def res(self):
        result = {
            'date': self.date,
            'title': self.title,
            'body': self.body
        }

        return result

    @property
    def in_body(self):
        return 'body' in self.in_set

    @property
    def in_date(self):
        return 'date' in self.in_set

    @property
    def in_title(self):
        return 'title' in self.in_set

    def outside(self, tag):
        self.in_set.remove(tag)

    def inside(self, tag):
        self.in_set.add(tag)

    def try_get_date(self, attrs):

        date_type = None
        content = None

        for attr in attrs:
            if attr[1] is None:
                continue
            elif attr[0] == 'itemprop':
                date_type = attr[1]
            elif attr[0] == 'content':
                content = attr[1]

        if date_type == 'datePublished':
            self.date_publish = content
            return True
        elif date_type == 'dateCreated':
            self.date_create = content
            return True
        elif date_type == 'dateModified':
            self.date_modify = content
            return True

        return False

    def try_get_title(self, attrs):

        is_title = False
        content = None

        for attr in attrs:
            if attr[1] == 'og:title' or attr[1] == 'twitter:title':
                is_title = True
            elif attr[0] == 'content':
                content = attr[1]

        if is_title:
            self.title = content
            return True

        return False

    def handle_starttag(self, tag, attrs):
        if tag == 'br':
            return

        if tag == 'p' or tag == 'pre':
            self.in_p = True

        if tag == 'title':
            self.inside('title')

        res = self.try_get_title(attrs)
        if not res:
            for attr in attrs:
                if attr[1] == 'articleBody':
                    self.inside('body')
                if attr[1] == 'xn-chron' and not self.no_parse_date:
                    self.inside('date')

        if self.in_body:
            self.levels = self.levels + 1

    def handle_endtag(self, tag):

        if tag == 'p':
            self.in_p = False

        if self.levels > 0:
            self.levels = self.levels - 1

        if self.in_body and self.levels <= 0:
            self.outside('body')

    def handle_data(self, data):

        if '/PRNewswire' in data and self.in_p:
            self.no_parse_date = True
            if not self.in_body:
                self.inside('body')
                self.levels = self.levels + 4

        if self.in_title:
            self.title = data
            self.outside('title')
        elif self.in_date:
            self.date = data
            self.outside('date')
        elif self.in_body:
            self.body = self.body + data


def str_to_date(date_string):
    # date string to date

    import datetime
    import pytz
    et = pytz.timezone("US/Eastern")
    date = datetime.datetime.strptime(date_string, "%b %d, %Y,  %H:%M ET")

    return date


def process_article(title, info, nlpp):
    ts, url = info['url']
    raw = get_archive_article(url, ts)

    parser = PRNewsHTMLParser()
    parser.feed(raw)
    article = parser.res
    article['title'] = nlpp.clean_sentence(article['title'])
    article['body'] = nlpp.clean_sentence(article['body'])

    try:
        if not is_english(' '.join([article['title'], article['body']])):
            # print(f'article is not in english: {title}')
            return None
    except:
        # print(f'invalid article: {title}')
        # print(' '.join([article['title'], article['body']]))
        # print('skipping..')
        return 'invalid'

    return article


def process_articles(info_list, count, failc, invac):
    nlpp = NLPProcessor()
    articles = {}
    fail_art = {}
    inva_art = {}
    print(f'running for {len(info_list)} articles')
    for k, v in info_list:

        if len(k) > 0:
            try:

                lenk = len(k)
                lenk = min(10, lenk)

                print(f'processing: {k[:lenk]}...')

                res = process_article(k, v, nlpp)
                if res == 'invalid':
                    invac.value = invac.value + 1
                    inva_art[k] = v
                elif res is not None:
                    articles[k] = res
            except:
                failc.value = failc.value + 1
                fail_art[k] = v
                # print(f'failure processing {k}')

        count.value = count.value + 1
        if count.value % 500 == 0:
            print(f"COUNT: {count.value}")
            print(f"FAILC: {failc.value}")
            print(f"INVAC: {invac.value}")

    return articles, fail_art, inva_art


def multi_process_articles(info_list, articles, fail_art, inva_art, count, failc, invac):
    # one batch of articles processing

    res, fail, inva = process_articles(info_list, count, failc, invac)

    try:
        print(f'writing results {count.value}')
        write_pickle(res, f'data/{count.value}_res.pkl')
    except:
        print(f'failed to save intermediate results {count.value}')
    articles.update(res)
    fail_art.update(fail)
    inva_art.update(inva)


def multi_run(data, chunk=10000, max_tasks=None):
    # need to modify this, don't use shared variables
    # save to storage

    lists = []
    curr_list = []
    for k, v in data.items():

        if len(curr_list) >= chunk:
            lists.append(curr_list)
            curr_list = []

        if max_tasks is not None and len(lists) >= max_tasks:
            break

        curr_list.append((k, v))

    num_workers = 10
    pool = mp.Pool(num_workers)
    with mp.Manager() as manager:
        articles = manager.dict()
        fail_art = manager.dict()
        inva_art = manager.dict()
        count = manager.Value('i', 0)
        failc = manager.Value('i', 0)
        invac = manager.Value('i', 0)

        for ls in lists:
            pool.apply_async(multi_process_articles, args=(ls, articles, fail_art, inva_art, count, failc, invac))

        pool.close()
        pool.join()

        import copy
        articles = copy.deepcopy(articles)

    return articles, fail_art, inva_art

# need to update this

# read in titles
res = read_pickle('data/prnews_title_dict.pkl')

print('starting')
articles, fail, invalid = multi_run(res, chunk=10000, max_tasks=None)

print("saving to pickle")
write_pickle(articles, 'data/final_articles.pkl')
write_pickle(fail, 'data/failed_articles.pkl')
write_pickle(invalid, 'data/invalid_articles.pkl')
