import nltk
from unidecode import unidecode

nltk.download('wordnet')
nltk.download('stopwords')


def load_stop_words(path):
    with open(path, 'r') as f:
        stop_words = f.read().splitlines()
    return stop_words


class Tokenizer(object):

    def __init__(self, min_token_size=None, language=None, stemmer=None, lemmatizer=None):

        self.min_token_size = min_token_size

        self.stemmer = stemmer
        self.stemmer_cache = {}

        self.lemmatizer = lemmatizer
        self.lemmatizer_cache = {}

        # Note: Better keep stopwords to improve the performance for simple sentences (e.g. "How are you?")
        self.stopwords = nltk.corpus.stopwords.words(language) if language else None

    def tokenize(self, text: str):

        # Split on whitespace
        initial_tokens = text.split()

        # Keep track of possible words for each token
        token_words = {}

        for token in initial_tokens:

            # Remove punctuation
            token = token.strip('.,;:!?()[]{}')

            # Remove special characters
            if not token.isalnum():
                continue

            # Remove accents
            token = unidecode(token)

            # Ignore tokens that don't meet the minimum size
            if self.min_token_size and len(token) < self.min_token_size:
                continue

            # Remove stop words
            if self.stopwords and token in self.stopwords:
                continue

            # Normalize to lowercase
            token = token.lower()

            # Stem tokens
            if self.stemmer:

                # Dynamic Programming to speed up the stemmer
                if token in self.stemmer_cache:
                    token = self.stemmer_cache[token]
                else:
                    token = self.stemmer.stem(token)
                    self.stemmer_cache[token] = token

            # Lemmatize tokens
            if self.lemmatizer:

                # Dynamic Programming to speed up the lemmatizer
                if token in self.lemmatizer_cache:
                    token = self.lemmatizer_cache[token]
                else:
                    token = self.lemmatizer.lemmatize(token)
                    self.lemmatizer_cache[token] = token

            # Keep track of possible words for each token
            if token not in token_words:
                token_words[token] = []

            token_words[token].append(token)

        return token_words
    

