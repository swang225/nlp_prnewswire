import pandas as pd
import multiprocessing as mp
import os

# ----------------------

import pickle


def write_pickle(data, file):
    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(file):
    with open(file, 'rb') as handle:
        b = pickle.load(handle)

    return b


import string
from nltk.tokenize import word_tokenize, sent_tokenize
from openie import StanfordOpenIE


class TextProcessor:

    def __init__(self):
        self._svo = None

    @staticmethod
    def clean_symbols(text, symbols):
        return text.translate(str.maketrans('', '', symbols))

    @staticmethod
    def clean_punctuations(text):
        return TextProcessor.clean_symbols(text, string.punctuation)

    @staticmethod
    def clean_digits(text):
        return TextProcessor.clean_symbols(text, string.digits)

    @staticmethod
    def to_words(text):
        return word_tokenize(text)

    @staticmethod
    def to_sentences(text):
        return sent_tokenize(text)

    @property
    def svo(self):
        if self._svo is None:
            self._svo = StanfordOpenIE()

        return self._svo

    def to_svo(self, text):
        return self.svo.annotate(text)

    @staticmethod
    def lower(text):
        return text.lower()


import os.path as osp

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag.stanford import StanfordNERTagger
from nltk.tag.stanford import StanfordPOSTagger


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

    @property
    def ner_tagger(self):
        if self._ner_tagger is None:
            jar = 'stanford-ner.jar'
            model = 'english.all.3class.distsim.crf.ser.gz'
            self._ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')

        return self._ner_tagger

    def tag_ner(self, words):
        return self.ner_tagger.tag(words)

    @property
    def pos_tagger(self):
        if self._pos_tagger is None:
            jar = 'stanford-postagger.jar'
            model = 'english-caseless-left3words-distsim.tagger'
            self._pos_tagger = StanfordPOSTagger(model, path_to_jar=jar)

        return self._pos_tagger

    def tag_pos(self, words, merge_nn=True):
        tagger = self.pos_tagger

        res_pos = tagger.tag(words)

        if merge_nn:

            def merge_nn(tg):
                word = ' '.join([t[0] for t in tg])
                pos = tg[-1][1]
                return word, pos

            res = []
            cur_tagged = []
            for tagged in res_pos:
                if tagged[1].startswith('NN'):
                    cur_tagged.append(tagged)
                else:
                    if len(cur_tagged) > 0:
                        res.append(merge_nn(cur_tagged)[0])
                        cur_tagged = []
                    res.append(tagged[0])

            res.append(merge_nn(cur_tagged)[0])

            res_pos = tagger.tag(res)

        return res_pos

    @staticmethod
    def nth_ne(words_ner, key, nth=0):

        for w in words_ner:
            if w[1] == key:
                nth = nth - 1
            if nth < 0:
                return w[0]
        return None

    @staticmethod
    def first_ne(words_ner, key):
        return TokenProcessor.nth_ne(words_ner, key, nth=0)

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


import os.path as osp
import pandas as pd
import gensim
from sklearn.cluster import KMeans


class NLPProcessor:

    def __init__(self, text_kwargs={}, token_kwargs={}):

        self.txtp = TextProcessor(**text_kwargs)
        self.tokp = TokenProcessor(**token_kwargs)

        self._wv_model = None

    def clean_words(self, sentence):
        txtp = self.txtp
        tokp = self.tokp

        res = txtp.clean_punctuations(sentence)

        words = txtp.to_words(res)
        words = tokp.clean_digits(words)
        words = tokp.lemmatize(words)
        words = tokp.clean_stop_words(words)

        return words

    def clean_sentence(self, sentence):
        words = self.clean_words(sentence)

        return self.tokp.sentence(words)

    def extract_svo(self, text, use_ner=False):

        svo_list = self.txtp.to_svo(text)

        if len(svo_list) == 0:
            return pd.Series(data={'s': None, 'v': None, 'o': None})

        if not use_ner:
            svo = svo_list[0]
            return pd.Series(data={'s': svo['subject'],
                                   'v': svo['relation'],
                                   'o': svo['object']})

        for svo in svo_list:
            words = self.txtp.to_words(svo['subject'])
            words_ner = self.tokp.tag_ner(words)
            org = self.tokp.first_ne(words_ner, key="ORGANIZATION")

            if org is not None:
                return pd.Series(data={'s': org,
                                       'v': svo['relation'],
                                       'o': svo['object']})

        return pd.Series(data={'s': None, 'v': None, 'o': None})

    @property
    def wv_model(self):
        if self._wv_model is None:
            # converted from glove.42B.300d.zip
            model_path = 'wv_100000.model'
            self._wv_model = gensim.models.KeyedVectors.load(model_path)
        return self._wv_model

    def calc_word_vec(self, text):

        text = self.clean_sentence(text)
        words = self.txtp.to_words(text.lower())

        count = 0
        sum = 0.0
        for w in words:
            if w in self.wv_model:
                sum = sum + self.wv_model[w]
                count = count + 1.0

        if count <= 0:
            return None

        return sum / count

    @staticmethod
    def kmean(n_clusters, train_data):
        km = KMeans(n_clusters=n_clusters)
        km.fit(train_data)

        return km


# ----------------------


def create_svo_wv(data):
    nlpp = NLPProcessor()

    print('extracting svo...')
    data = data['title'].apply(nlpp.extract_svo)
    data = data[~data['s'].isnull()]
    data['vo'] = data['v'] + ' ' + data['o']
    data = data[['s', 'v', 'o']]

    print('calc word vec...')
    data['wv_v'] = data['v'].apply(nlpp.calc_word_vec)
    data['wv_o'] = data['o'].apply(nlpp.calc_word_vec)
    data = data[~data['wv_v'].isnull() & ~data['wv_o'].isnull()]

    return data


def multi_create_svo_wv(data, output_name):
    print(f"processing {output_name}")
    data = create_svo_wv(data)

    print(f"saving {output_name}")
    data.to_pickle(output_name)


def multi_run(data,
              func=multi_create_svo_wv,
              prefix="svo_data/svo_sep_",
              chunk=3000):

    job_dict = {}
    total = len(data)
    count = int(total / chunk)
    for i in range(count):

        start = chunk * i
        end = chunk * (i + 1)

        print(f'{i}: {start} - {end}')
        if i == count - 1:
            print('last chunk')
            subset = data[start:]
        else:
            subset = data[start:end]

        output_name = f"{prefix}{i}.pkl"
        job_dict[output_name] = (subset, output_name)

    completed = []
    while True:
        keys = [k for k in job_dict.keys()]
        if len(keys) <= 0:
            break

        num_workers = mp.cpu_count()
        print(f'num workers: {num_workers}')
        pool = mp.Pool(num_workers)

        for k in keys:
            if os.path.exists(k):
                job_dict.pop(k)
                completed.append(k)
            else:
                pool.apply_async(func, args=job_dict[k])

        pool.close()
        pool.join()

    res = []
    for file in completed:
        file_res = pd.read_pickle(file)
        res.append(file_res)
    res = pd.concat(res, axis=0)
    pd.to_pickle(res, f"{prefix}complete.pkl")

    return res


if __name__ == '__main__':

    import platform, multiprocessing
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')

    data = pd.read_pickle('prnews_title.pkl')
    multi_run(data)

# output is svo_sep_complete.pkl
# dataframe date: s, v_wv, o_wv


