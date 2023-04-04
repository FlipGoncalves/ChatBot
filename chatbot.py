import itertools
import json
# import random

from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import nltk

from tokenizer import Tokenizer


def apply_corrections(tokens, is_question):

    for i, token in enumerate(tokens):
        print(f'[{i}] Did you mean: {" ".join(token)}{"?" if is_question else ""}')

    selected = False
    while not selected:

        # Obtain user input
        user_input = input('>> ')

        # If user input is a number
        if user_input.isdigit():

            # Convert user input to integer
            user_input = int(user_input)

            # If user input is a valid index
            if len(tokens) > user_input >= 0:

                # Update tokens
                tokens = tokens[user_input]

                # Update selected
                selected = True

            else:
                print('Invalid input. Please try again.')

        else:
            print('Invalid input. Please try again.')

    return tokens


class Chatbot:

    def __init__(self, tokenizer, path):

        # Handle tokenizer
        self.tokenizer = tokenizer

        # Handle dataset path
        self.dataset_path = path

        # # Handle english dataset
        # self.english_questions = {}
        # self.english_answers = {}
        #
        # # Handle portuguese dataset
        # self.portuguese_questions = {}
        # self.portuguese_answers = {}

        # Handle questions and answers
        self.questions = {}
        self.answers = {}

        # Handle all tokens
        # self.english_token_words = {}
        # self.portuguese_token_words = {}
        self.token_words = {}

        # Handle number of documents
        # self.n_documents = 0

        # Handle vector representation (for machine learning)
        self.vector_representation = None

    def load_database(self):

        # Open dataset
        with open(self.dataset_path, 'r', encoding='utf-8') as dataset:

            # Load dataset
            data = json.load(dataset)

            # Load all entries
            for entry in data:

                # Load all languages
                for language in entry['questions']:
                    self.load_entry(entry, language)

    def load_entry(self, entry: json, language: str):

        # Obtain intent
        intent = entry['intent']

        # Obtain questions
        questions = entry['questions'][language]

        # Obtain answers
        answers = entry['answers'][language]

        for question in questions:

            # Add questions to database
            self.add_question(intent, question, language)

            # Tokenize questions
            question_tokens_words = self.tokenizer.tokenize(question)

            # Add token words to database
            for token in question_tokens_words:
                self.add_token(token, question_tokens_words[token], language)

        # Tokenize answers
        for answer in answers:

            # Add answers to database
            self.add_answer(intent, answer, language)

            # TODO: Keep track of answer tokens

    def add_token(self, token, words, language):

        if language not in self.token_words:
            self.token_words[language] = {}

        if token not in self.token_words[language]:
            self.token_words[language][token] = set()

        for word in words:
            self.token_words[language][token].add(word)

    def add_question(self, intent, question, language):

        # if language == 'english':
        #     if intent not in self.english_questions:
        #         self.english_questions[intent] = []
        #     self.english_questions[intent].append(question)
        #
        # elif language == 'portuguese':
        #     if intent not in self.portuguese_questions:
        #         self.portuguese_questions[intent] = []
        #     self.portuguese_questions[intent].append(question)

        # Note: This approach allows for more languages later on
        if language not in self.questions:
            self.questions[language] = {}

        if intent not in self.questions[language]:
            self.questions[language][intent] = []

        self.questions[language][intent].append(question)

    def add_answer(self, intent, answer, language):

        # if language == 'english':
        #     if intent not in self.english_answers:
        #         self.english_answers[intent] = []
        #     self.english_answers[intent].append(answer)
        #
        # elif language == 'portuguese':
        #     if intent not in self.portuguese_answers:
        #         self.portuguese_answers[intent] = []
        #     self.portuguese_answers[intent].append(answer)

        # Note: This approach allows for more languages later on
        if language not in self.answers:
            self.answers[language] = {}

        if intent not in self.answers[language]:
            self.answers[language][intent] = []

        self.answers[language][intent].append(answer)

    def detect_language(self, tokens):

        languages_detected = {}
        for language in self.token_words:

            for token in tokens:
                if token in self.token_words[language]:
                    if language not in languages_detected:
                        languages_detected[language] = 0
                    languages_detected[language] += 1

        if len(languages_detected) == 0:
            print('[Language not detected]')
            return None

        return max(languages_detected, key=lambda key: languages_detected[key])

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

    def process_input(self, user_input, is_question):

        # Tokenize input
        tokenized_input = self.tokenizer.tokenize(user_input)

        # Detect language
        language = self.detect_language(tokenized_input)

        # If language was not detected
        if language is None:

            # Initialize tokens
            tokens = []

            # Attempt all languages
            for language_attempt in self.token_words:

                temp_tokens = self.obtain_input_tokens(tokenized_input, language_attempt)

                if len(temp_tokens) > len(tokens):

                    # Update tokens and language
                    tokens = temp_tokens
                    language = language_attempt

        else:

            # Obtain input tokens with spelling check
            tokens = self.obtain_input_tokens(tokenized_input, language)

        if len(tokens) == 1:
            tokens = tokens[0]

        elif len(tokens) > 1:
            tokens = apply_corrections(tokens, is_question)

        if len(tokens) != 0:
            print(f'Input for language {language.upper()}: {" ".join(tokens)}{"?" if is_question else ""}')
            print(f'Language detected: {language.upper() if language is not None else "None"}')

        return tokens

    def obtain_input_tokens(self, tokenized_input, language):

        # tokens = []  # Check if tokens are in database

        tokens = []
        for token in tokenized_input:

            # If token is in database
            if token in self.token_words[language]:
                # tokens.append(token)
                tokens.append([token])

            # If token is not in database
            elif token not in self.token_words[language]:

                print(f'Token "{token}" not found in database {language.upper()}')

                # Token must be at least 4 characters long for spelling check
                if len(token) < 3:

                    print(f' * "{token}": Token too short to check spelling')
                    continue

                # Initialize possible corrections (spelling check)
                possible_corrections = []

                # Check if token was misspelled
                for other_token in self.token_words[language]:

                    # If token is 60% similar to other token
                    if nltk.edit_distance(token, other_token) <= 0.4 * len(token):

                        # print(f' * "{token}": Did you mean any of the following?')
                        # print(f'   - {", ".join(self.token_words[language][other_token])}')

                        # Add possible correction
                        possible_corrections.append(other_token)

                # If there are no possible corrections
                if len(possible_corrections) == 0:
                    print(f' * "{token}": No possible corrections found')
                    continue

                # Add possible corrections to tokens
                tokens.append(possible_corrections)

                # Choose a random correction
                # correction = random.choice(possible_corrections)
                # tokens.append(correction)

        # Create combination of tokens
        tokens = list(itertools.product(*tokens))

        return tokens

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

            # Check if user input is empty
            if not user_input:
                continue

            # Process input
            self.process_input(user_input, user_input.endswith('?'))


if __name__ == '__main__':

    # Prepare tokenizer
    tokenizer = Tokenizer(stopwords_path='stopwords.txt')

    # Prepare chatbot
    chatbot = Chatbot(tokenizer=tokenizer, path='datasets/sample_dataset.json')

    # Load database (training data)
    chatbot.load_database()

    # Start chatbot
    chatbot.start()