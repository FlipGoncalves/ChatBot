import json

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

from tokenizer import Tokenizer


class Chatbot:

    def __init__(self, tokenizer, path):

        # Handle tokenizer
        self.tokenizer = tokenizer

        # Handle database
        self.database_path = path
        self.database = {}

        # Handle index
        self.terms = {}
        self.n_documents = 0

        # Handle vector representation (for machine learning)
        self.vector_representation = None

    def load_database(self):

        with open(self.database_path, 'r') as file:

            data = json.load(file)
            for entry in data:

                # Obtain entry id
                entry_id = entry['id']

                # Obtain question
                question = entry['question']

                # Tokenize question
                question_tokens = self.tokenizer.tokenize(question)

                # Add question to index
                for token in question_tokens:
                    self.add_term(token, entry_id)

                # Obtain annotations
                annotations = entry['annotations']

                for annotation in annotations:

                    # Check annotation type (singleAnswer)
                    if annotation['type'] == 'singleAnswer':

                        # Obtain answer
                        answer = annotation['answer']

                        # Convert answer to string
                        answer = '. '.join(answer)

                        # Add entry to database
                        self.add_database_entry(entry_id, question, answer)

                    # Check annotation type (multipleAnswer)
                    elif annotation['type'] == 'multipleQAs':

                        # Obtain answers
                        answers = annotation['qaPairs']

                        # Obtain first answer
                        answer = answers[0]

                        # Convert answer to string
                        answer = '. '.join(answer)

                        # Add entry to database
                        self.add_database_entry(entry_id, question, answer)

            # Sort index terms
            self.sort_terms()

            # Prepare training data in Vector Representation
            self.vector_representation = self.prepare_database()

    def add_term(self, term, entry_id):

        if term not in self.terms:
            self.terms[term] = set()
        self.terms[term].add(entry_id)

    def add_database_entry(self, entry_id, question, answer):
        self.database[entry_id] = [question, answer]

    def sort_terms(self):
        self.terms = dict(sorted(self.terms.items()))

    def prepare_database(self):

        corpus = [self.database[entry_id][0] for entry_id in self.database]

        tf_idf_model = TfidfVectorizer()
        tf_idf_vector = tf_idf_model.fit_transform(corpus)

        terms = tf_idf_model.get_feature_names_out()
        tf_idf_array = tf_idf_vector.toarray()

        df_tf_idf = pd.DataFrame(tf_idf_array, columns=terms)
        print(df_tf_idf)

        return tf_idf_vector


if __name__ == '__main__':

    # Prepare tokenizer
    tokenizer = Tokenizer(stopwords_path='stopwords.txt')

    # Prepare chatbot
    chatbot = Chatbot(tokenizer=tokenizer, path='datasets/small_ambigNQ.json')

    # Load database (training data)
    chatbot.load_database()