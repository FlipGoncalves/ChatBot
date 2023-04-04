import json

from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import nltk

from tokenizer import Tokenizer


class Chatbot:

    def __init__(self, tokenizer, path):

        # Handle tokenizer
        self.tokenizer = tokenizer

        # Handle dataset path
        self.dataset_path = path

        # Handle dataset
        self.questions = {}
        self.answers = {}

        # Handle all tokens
        self.tokens = {}

        # Handle number of documents
        # self.n_documents = 0

        # Handle vector representation (for machine learning)
        self.vector_representation = None

    def load_database(self):

        with open(self.dataset_path, 'r') as file:

            data = json.load(file)
            for entry in data:

                # Obtain intent
                intent = entry['intent']

                # Obtain questions
                questions = entry['questions']

                # Obtain english questions
                english_questions = questions['english']

                # Tokenize questions
                tokenized_questions = [token for question in english_questions
                                       for token in self.tokenizer.tokenize(question)]

                # Add questions to database
                self.add_questions(intent, tokenized_questions)

                # Keep track of question tokens
                for token in tokenized_questions:
                    self.add_token(intent, token)

                # Obtain answers
                answers = entry['answers']

                # Obtain english answers
                english_answers = answers['english']

                # Add answers to database
                self.add_answers(intent, english_answers)

                # TODO: Keep track of answer tokens

                # Update number of documents
                # self.n_documents += 1

            # Sort index terms
            self.sort_terms()

            # Prepare training data in Vector Representation
            # self.vector_representation = self.prepare_database()

    def add_token(self, intent, token):
        if token not in self.tokens:
            self.tokens[token] = set()
        self.tokens[token].add(intent)

    def add_questions(self, intent, questions):
        self.questions[intent] = questions

    def add_answers(self, intent, answers):
        self.answers[intent] = answers

    def sort_terms(self):
        self.tokens = dict(sorted(self.tokens.items()))

    def prepare_database(self):

        corpus = [question for intent in self.questions
                  for question in self.questions[intent]]

        tf_idf_model = TfidfVectorizer()
        tf_idf_vector = tf_idf_model.fit_transform(corpus)

        terms = tf_idf_model.get_feature_names_out()
        tf_idf_array = tf_idf_vector.toarray()

        df_tf_idf = pd.DataFrame(tf_idf_array, columns=terms)
        print(df_tf_idf)

        return tf_idf_vector

    def process_input(self, user_input):

        # Tokenize input
        tokenized_input = self.tokenizer.tokenize(user_input)

        # Check if tokens are in database
        for token in tokenized_input:

            # If token is not in database
            if token not in self.tokens.keys():
                print(f'Token "{token}" not found in database')

                # TODO: Check token length before checking for similar tokens

                # Check if token was misspelled
                for other_token in self.tokens.keys():

                    # If token is 60% similar to other token
                    if nltk.edit_distance(token, other_token) <= 0.4 * len(token):
                        print(f'"{token}": Did you mean "{other_token}"?')

            # If token is in database
            else:
                print(f'Token "{token}" found in database')

    def start(self):

        # Greet user
        print('Hello, I am a chatbot. How can I help you?')

        # Start chatbot
        while True:

            # Obtain user input
            user_input = input('> ')

            # Check if user wants to exit
            if user_input == 'exit':
                break

            # Process input
            self.process_input(user_input)


if __name__ == '__main__':

    # Prepare tokenizer
    tokenizer = Tokenizer(stopwords_path='stopwords.txt')

    # Prepare chatbot
    chatbot = Chatbot(tokenizer=tokenizer, path='datasets/sample_dataset.json')

    # Load database (training data)
    chatbot.load_database()

    # Start chatbot
    chatbot.start()