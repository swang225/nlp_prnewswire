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
