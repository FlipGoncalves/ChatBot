import itertools
import json
import time

import numpy as np
import nltk
import random
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

from nltk.stem import WordNetLemmatizer

from tokenizer import Tokenizer
from grammar_checker import GrammarChecker

# nltk.download('wordnet')
# nltk.download('stopwords')

import spacy

from colorama import Fore, Style

def apply_corrections(tokens, is_question):

    for i, token in enumerate(tokens):
        print(f'\t[{i}] Did you mean: ' + Fore.RED + f'{" ".join(token)}{"?" if is_question else ""}' + Style.RESET_ALL)

    selected = False
    while not selected:

        # Obtain user input
        user_input = input(Style.BRIGHT + Fore.BLUE + '>>> ' + Style.RESET_ALL)

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

    def __init__(self, tokenizer, grammar_checker, path):

        # Handle tokenizer
        self.tokenizer = tokenizer

        # Handle dataset path
        self.dataset_path = path

        # Handle questions and answers
        self.questions = {}
        self.answers = {}

        # Handle extensions
        self.extensions = {}

        # Handle intents
        self.intents = []

        # Handle entities for context
        self.entities = {}

        # Handle cache entities
        self.cacheEntities={}

        # Save token when it has not been identified as an entity, so that is saved in the entities when we have the correct intent
        self.forgottenEntity = None

        #Grammar Checker
        self.grammar_checker = grammar_checker

        #Tree from last sentence
        self.last_tree = ""

        # Model
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self.tf_idf_model = TfidfVectorizer()

        # Handle all tokens
        self.token_words = {}

        # Handle vector representation (for machine learning)
        self.vector_representation = None

        # handle error in model
        self.last_tag = ""
        self.last_question = ""
        self.last_lang = ""
        self.count = 0
        self.all_questions = []

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
        questions = entry['text'][0]

        # Obtain answers
        answers = entry['responses'][0]

        # Obtain extensions
        extensions = entry['extension']["function"]
        
        self.cacheEntities[intent] = entry['entities']

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

        # Add extensions to database
        self.add_extension(intent, extensions)

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

        question_tokens = self.tokenizer.tokenize(question)
        self.all_questions.append("".join([x[0] for x in question_tokens.values()]))

    def remove_question(self, intent, question, language):

        try:
            index = self.questions[language][intent].index(question)
            del self.questions[language][intent][index]
        except ValueError:
            pass

    def add_answer(self, intent, answer, language):

        if language not in self.answers:
            self.answers[language] = {}

        if intent not in self.answers[language]:
            self.answers[language][intent] = []

        
        self.answers[language][intent].append(answer)
        if intent in self.cacheEntities.keys() and self.cacheEntities[intent] != []:
            self.answers[language][intent+'Entity']=self.cacheEntities[intent]
            # print(intent+"Entity")

    def add_extension(self, intent, function):

        if intent not in self.extensions:
            self.extensions[intent] = function

    def get_time(self):
        current_time = time.localtime()
        return str(current_time[3]) + ":" + str(current_time[4]) 

    def detect_language(self, tokens):

        languages_detected = {}
        for language in self.token_words:

            for token in tokens:
                if token in self.token_words[language]:
                    if language not in languages_detected:
                        languages_detected[language] = 0
                    languages_detected[language] += 1

        if len(languages_detected) == 0:
            print(Style.BRIGHT + Fore.GREEN + 'Chatty' + Style.RESET_ALL + ': ' + Fore.RED + '[Language not detected]' + Style.RESET_ALL)
            return None

        return max(languages_detected, key=lambda key: languages_detected[key])

    def train_model(self, max_iter=3000):

        corpus = []
        tokens = []
        tags = []
        for language in self.questions:
            for intent in self.questions[language]:
                for question in self.questions[language][intent]:
                    corpus.append(question)
                    tokens.extend(self.tokenizer.tokenize(question))
                    tags.append(intent)

        tokens = [self.lemmatizer.lemmatize(word.lower()) for word in tokens if word not in string.punctuation]
        self.tf_idf_model.fit_transform(tokens)
        X_train = self.tf_idf_model.transform(corpus)
        y_train = np.zeros((len(corpus), len(self.intents)))

        for i, tag in enumerate(tags):
            y_train[i][self.intents.index(tag)] = 1
        self.model = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='identity', solver='lbfgs', max_iter=max_iter)
        self.model.fit(X_train, y_train)

        # all_words = []
        # documents = []
        # tags = []
        # for intent in self.intents:
        #     for pattern in self.questions[language]:
        #         words = self.tokenize(pattern)
        #         all_words.extend(words)
        #         documents.append(pattern)
        #         tags.append(intent)

        # all_words = [self.lemmatizer.lemmatize(word.lower()) for word in all_words if word not in string.punctuation]
        # self.tf_idf_model.fit_transform(all_words)
        # X_train = self.tf_idf_model.transform(documents)
        # y_train = np.zeros((len(documents), len(self.intents)))

        # for i, tag in enumerate(tags):
        #     y_train[i][self.intents.index(tag)] = 1
        # self.model = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=max_iter)
        # self.model.fit(X_train, y_train)

    def predict_intent(self, message):

        tokens, lang = self.tokenize(message, message.endswith('?'))
        message_vector = self.tf_idf_model.transform([" ".join(tokens)])
        predicted_tag = self.intents[np.argmax(self.model.predict(message_vector))]

        return predicted_tag, lang
    
    # Define a function to extract named entities from a text input using spaCy
    @staticmethod
    def extract_entities(text):
        nlp = spacy.load('en_core_web_sm')

        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities

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

        # Add token words to database
        i=0
        for token in tokenized_input:
            if tokens[i] != "<NULL>":
                self.add_token(token, tokens[i] if tokens[i] != "" else tokenized_input[token], language)
            i+=1

        return tokens, language

    def obtain_input_tokens(self, tokenized_input, language):

        # print(tokenized_input)

        tokens = []
        i=0
        for token in tokenized_input:
            i+=1
            # If token is in database
            if token in self.token_words[language]:
                # tokens.append(token)
                tokens.append([token])

            # If token is not in database
            elif token not in self.token_words[language]:

                # print(Style.BRIGHT + Fore.GREEN + 'Chatty' + Style.RESET_ALL + f': Token "{token}" ' + Fore.RED + 'not found' + Style.RESET_ALL + ' in database ' + Fore.RED + f'{language.upper()}' + Style.RESET_ALL)

                # Token must be at least 4 characters long for spelling check
                if len(token) < 3:

                    # print(f' * "{token}": Token too short to check spelling')
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
                    if i==len(tokenized_input):
                        print("entity")
                        self.forgottenEntity=tokenized_input[token][-1]
                        tokens.append(['<NULL>'])
                    else:
                        tokens.append([token])
                else:
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
        print('\n' + Style.BRIGHT + Fore.GREEN + 'Chatty' + Style.RESET_ALL + ': Hello, I am ' + Style.BRIGHT + Fore.GREEN + 'Chatty' + Style.RESET_ALL + '. How can I help you ?\n\tOlá, eu sou o ' + Style.BRIGHT + Fore.GREEN + 'Chatty' + Style.RESET_ALL + '. Como posso ajudar ?')

        # Start chatbot
        while True:

            # Obtain user input
            user_input = input(Style.BRIGHT + Fore.BLUE + '> ' + Style.RESET_ALL)

            # Check if user wants to exit
            if user_input == 'exit':
                print(Style.BRIGHT + Fore.GREEN + 'Chatty' + Style.RESET_ALL + ": Goodbye !\n\tAdeus !")
                self.saveDataset()
                break

            # Check if user input is empty
            if not user_input:
                continue

            potential_tree = self.grammar_checker.check_grammar(user_input)

            # Take entities from user input, only stays with the latest information
            # entity= self.extract_entities(user_input)

            # if potential_tree is None:
            #     #Bad grammar
            #     print(Style.BRIGHT + Fore.GREEN + 'Chatty' + Style.RESET_ALL + ": You should check your grammar!\n\t Devias verificar a tua gramática!")
            #     continue

            #print("Potential tree: " + str(potential_tree))
            #potential_tree.pretty_print()

            # Take entities from user input, only stays with the latest information
            # entity=self.extract_entities(user_input)
            # if entity:
            #     entity=[(entity[0][1],entity[0][0])]
            #     print(entity)
            #     self.entities.update(entity)
            #     user_input=user_input.replace(entity[0][1],'<NULL>')

            #print(self.entities)
            # Process input
            tag, language = self.predict_intent(user_input)

            #print("Tag: " + tag)
            
            # Check if forgottenEntity is not None
            if self.forgottenEntity is not None:
                print(self.forgottenEntity)
                # Add forgotten entity to entities
                # print("using entities")
                if tag+'Entity' in self.answers[language].keys():
                    self.entities[self.answers[language][tag+'Entity'][0]] = self.forgottenEntity
                # print(self.entities)
                # Reset forgotten entity
                self.forgottenEntity = None

            print(self.entities)
            print(tag)
            
            response = random.choice(self.answers[language][tag])

            if tag == "NotCorrect":
                self.remove_question(self.last_tag, self.last_question, self.last_lang)

                print(Style.BRIGHT + Fore.GREEN + 'Chatty' + Style.RESET_ALL + f": {response}{self.last_question}")

                # Obtain user input
                user_input = input(Style.BRIGHT + Fore.BLUE + '> ' + Style.RESET_ALL)

                # Check if user wants to exit
                if user_input == 'exit':
                    print(Style.BRIGHT + Fore.GREEN + 'Chatty' + Style.RESET_ALL + ": Goodbye !\n\tAdeus !")
                    break

                new_tag = "UserInput"+str(self.count)
                self.count += 1
                while new_tag in self.intents:
                    new_tag = "UserInput"+str(self.count)
                    self.count += 1

                question_tokens_words = self.tokenizer.tokenize(self.last_question)
                # Add token words to database
                for token in question_tokens_words:
                    self.add_token(token, question_tokens_words[token], self.last_lang)

                self.intents.append(new_tag)
                self.add_question(new_tag, self.last_question, self.last_lang)
                self.add_answer(new_tag, user_input, self.last_lang)
                self.last_tag = new_tag

                print(Style.BRIGHT + Fore.GREEN + 'Chatty' + Style.RESET_ALL + ": Thank you for your help / Obrigado pela ajuda !!")

                # train the model
                chatbot.train_model()

            elif tag == "SintaticTree":
                print(Style.BRIGHT + Fore.GREEN + 'Chatty' + Style.RESET_ALL + f": {response}{self.last_question}")
                self.last_tree.pretty_print()
                self.last_question = user_input
                self.last_tree = potential_tree

            else:
                # temp = response
                # Check if response has <> tags
                if '<' in response and '>' in response:
                    #Extract substring between <>
                    substring = response[response.find("<") + 1:response.find(">")]
                    if substring in self.entities.keys():
                        response=response.replace(f'<{substring}>', Style.BRIGHT + Fore.BLUE + self.entities[substring] + Style.RESET_ALL)
                    elif self.extensions[tag] != "":
                        response=response.replace(f'<{substring}>', getattr(self, self.extensions[tag])())
                    else:
                        response="I didn't understand what you said, please try with another phrasing\n\tEu não percebi o que querias dizer, tenta dizer de outra maneira"


                entity= None

                # print(language, tag)
                print(Style.BRIGHT + Fore.GREEN + 'Chatty' + Style.RESET_ALL + f": {response}")

                self.last_question = user_input
                self.last_tag = tag
                self.last_lang = language
                self.last_tree = potential_tree

                # question_tokens = self.tokenizer.tokenize(temp)

                # if question_tokens not in self.all_questions:
                #     self.add_question(tag, temp, language)

                # train the model
                chatbot.train_model()

    def saveDataset(self):
        with open("datasets/DataSetSave.json", "w") as f:
            data = {"intents": []}
            for intent in self.intents:
                data_intent = {"intent": intent, "text": [], "responses": [], "entities": [], "extension": {}}
                data_text = {}
                data_resp = {}
                for language in self.questions.keys():
                    if intent in self.questions[language]:
                        data_text[language] = self.questions[language][intent]
                    if intent in self.answers[language]:
                        data_resp[language] = self.answers[language][intent]
                data_intent['text'].append(data_text)
                data_intent['responses'].append(data_resp)
                if intent in self.cacheEntities.keys():
                    data_intent["entities"] = self.cacheEntities[intent]
                data["intents"].append(data_intent)
                if intent in self.extensions.keys():
                    data_intent["extension"]["function"] = self.extensions[intent]
                else:
                    data_intent["extension"]["function"] = ""

            json.dump(data, f)


if __name__ == '__main__':

    # Prepare tokenizer
    tokenizer = Tokenizer(min_token_size=2, stemmer=nltk.stem.PorterStemmer())

    # Alternative:
    # tokenizer = Tokenizer(lemmatizer=nltk.stem.WordNetLemmatizer())

    grammar_checker = GrammarChecker()

    # Prepare chatbot
    chatbot = Chatbot(tokenizer=tokenizer, grammar_checker=grammar_checker, path='datasets/DataSetSave.json')

    # Load database (training data)
    chatbot.load_database()

    # train the model
    chatbot.train_model()

    # Start chatbot
    chatbot.start()