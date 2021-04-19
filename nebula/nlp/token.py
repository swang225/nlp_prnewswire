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
            jar = osp.join(osp.dirname(__file__), 'model', 'stanford-ner.jar')
            model = osp.join(osp.dirname(__file__), 'model', 'english.all.3class.distsim.crf.ser.gz')
            self._ner_tagger = StanfordNERTagger(model, jar, encoding='utf8')

        return self._ner_tagger

    def tag_ner(self, words):
        return self.ner_tagger.tag(words)

    @property
    def pos_tagger(self):
        if self._pos_tagger is None:
            jar = osp.join(osp.dirname(__file__), 'model', 'stanford-postagger.jar')
            model = osp.join(osp.dirname(__file__), 'model', 'english-caseless-left3words-distsim.tagger')
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


