import os.path as osp
import pandas as pd
import gensim
from sklearn.cluster import KMeans

from .text import TextProcessor
from .token import TokenProcessor


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
            model_path = osp.join(osp.dirname(__file__), 'model', 'wv_100000.model')
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

