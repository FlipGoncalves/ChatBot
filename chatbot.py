import itertools
import json
# import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

from tokenizer import Tokenizer

import numpy as np
import nltk
import random


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

        # Handle questions and answers
        self.questions = {}
        self.answers = {}
        self.extensions = {}

        # Likes
        self.likes = ['dogs', 'cats', 'money']

        # Handle intents
        self.intents = []

        # Model
        self.model = None
        self.tf_idf_model = TfidfVectorizer()

        # Handle all tokens
        self.token_words = {}

        # Handle vector representation (for machine learning)
        self.vector_representation = None

    def getLikes(self):
        return self.likes[0]

    def load_database(self):

        # Open dataset
        with open(self.dataset_path, 'r', encoding='utf-8') as dataset:

            # Load dataset
            data = json.load(dataset)

            # Load all entries
            for entry in data["intents"]:
                # Load all languages
                self.load_entry(entry)

    def load_entry(self, entry: json):

        # Obtain intent
        intent = entry['intent']

        # Obtain questions
        questions = entry['text']

        # Obtain answers
        answers = entry['responses']

        # Obtain answers
        extensions = entry['extension']

        for language in questions:
            for question in questions[language]:

                if intent not in self.intents: self.intents.append(intent)

                # Add questions to database
                self.add_question(intent, question, language)

                # Tokenize questions
                question_tokens_words = self.tokenizer.tokenize(question)

                # Add token words to database
                for token in question_tokens_words:
                    self.add_token(token, question_tokens_words[token], language)

        # Tokenize answers
        for language in answers:
            for answer in answers[language]:

                # Add answers to database
                self.add_answer(intent, answer, language)

                # TODO: Keep track of answer tokens

        for key in extensions:

            # Add extensions to database
            self.add_extension(intent, extensions[key], key)

    def add_token(self, token, words, language):

        # Note: This approach allows for more languages later on
        if language not in self.token_words:
            self.token_words[language] = {}

        if token not in self.token_words[language]:
            self.token_words[language][token] = set()

        for word in words:
            self.token_words[language][token].add(word)

    def add_question(self, intent, question, language):

        if language not in self.questions:
            self.questions[language] = {}

        if intent not in self.questions[language]:
            self.questions[language][intent] = []

        self.questions[language][intent].append(question)

    def add_answer(self, intent, answer, language):

        if language not in self.answers:
            self.answers[language] = {}

        if intent not in self.answers[language]:
            self.answers[language][intent] = []

        self.answers[language][intent].append(answer)

    def add_extension(self, intent, extension, key):

        if key not in self.extensions:
            self.extensions[key] = {}

        if intent not in self.extensions[key]:
            self.extensions[key][intent] = extension

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
    
    def get_response(self, intent, language):
        response = random.choice(self.answers[language][intent])
        
        # Check if there is an extension
        if self.extensions["function"][intent] != "":
            # New text for response
            extension_response = random.choice(self.extensions[language][intent])
            # Text that must be swap in reponse
            tag = self.extensions["tag"][intent]
            # Replace tag
            response = response.replace(tag, extension_response)

        return response

    def train_model(self):

        corpus = []
        tokens = []
        tags = []
        for language in self.questions:
            for intent in self.questions[language]:
                for question in self.questions[language][intent]:
                    corpus.append(question)
                    tokens.extend(self.tokenizer.tokenize(question))
                    tags.append(intent)

        self.tf_idf_model.fit_transform(tokens)
        X_train = self.tf_idf_model.transform(corpus)
        y_train = np.zeros((len(corpus), len(self.intents)))

        for i, tag in enumerate(tags):
            y_train[i][self.intents.index(tag)] = 1
        self.model = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='identity', solver='lbfgs', max_iter=500)
        self.model.fit(X_train, y_train)

    def predict_intent(self, message):

        tokens, lang = self.tokenize(message, message.endswith('?'))
        message_vector = self.tf_idf_model.transform([" ".join(tokens)])
        predicted_tag = self.intents[np.argmax(self.model.predict(message_vector))]
        return predicted_tag, lang
    
    def tokenize(self, user_input, is_question):

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

        # if len(tokens) != 0:
        #     print(f'Input for language {language.upper()}: {" ".join(tokens)}{"?" if is_question else ""}')
        #     print(f'Language detected: {language.upper() if language is not None else "None"}')

        return tokens, language

    def obtain_input_tokens(self, tokenized_input, language):

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

                # Choose a random correction and add it to tokens
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
                print("Goodbye !!")
                break

            # Check if user input is empty
            if not user_input:
                continue

            # Process input
            tag, language = self.predict_intent(user_input)

            response = self.get_response(tag,language)
            # response = random.choice(self.answers[language][tag])

            print(f"ChatBot: {response}")


if __name__ == '__main__':

    # Prepare tokenizer
    tokenizer = Tokenizer(stopwords_path='stopwords.txt')

    # Prepare chatbot
    chatbot = Chatbot(tokenizer=tokenizer, path='datasets/extension_dataset.json')

    # Load database (training data)
    chatbot.load_database()

    # train the model
    chatbot.train_model()

    # Start chatbot
    chatbot.start()