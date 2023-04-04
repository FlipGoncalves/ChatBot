import nltk


def load_stop_words(path):
    with open(path, 'r') as f:
        stop_words = f.read().splitlines()
    return stop_words


class Tokenizer(object):

    def __init__(self, stopwords_path, min_token_size=2):
        self.min_token_size = min_token_size

        self.stemmer = nltk.stem.PorterStemmer()
        self.stemmer_cache = {}

        # Note: Better keep stopwords to improve the performance for simple sentences (e.g. "How are you?")
        # self.stopwords = load_stop_words(stopwords_path)

    def tokenize(self, text):

        # Split on whitespace
        initial_tokens = text.split()

        tokens = []
        for token in initial_tokens:

            # Remove punctuation
            token = token.strip('.,;:!?()[]{}')

            # Remove special characters
            if not token.isalnum():
                continue

            # Ignore tokens that don't meet the minimum size
            if len(token) < self.min_token_size:
                continue

            # Remove stop words
            # if token in self.stopwords:
            #     continue

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

            tokens.append(token)

        return tokens
