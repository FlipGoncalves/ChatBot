import json
import math

from tokenizer import Tokenizer


def calculate_term_frequency(terms):
    # Initialize term frequency
    term_frequency = {}

    # Count term frequency
    for term in terms:
        if term not in term_frequency:
            term_frequency[term] = 0
        term_frequency[term] += 1

    return term_frequency


def create_tf_idf_matrix(term_frequency_matrix, inverse_document_frequency_vector):

    # Initialize tf-idf matrix
    tf_idf_matrix = {}

    # Calculate tf-idf for each document
    for entry_id in term_frequency_matrix:

        # Initialize tf-idf vector
        tf_idf_vector = {}

        # Calculate tf-idf for each term
        for term in term_frequency_matrix[entry_id]:

            # Calculate tf-idf
            tf_idf = term_frequency_matrix[entry_id][term] * inverse_document_frequency_vector[term]

            # Add tf-idf to tf-idf vector
            tf_idf_vector[term] = tf_idf

        # Add tf-idf vector to tf-idf matrix
        tf_idf_matrix[entry_id] = tf_idf_vector

    return tf_idf_matrix


def create_vector_representation(tf_idf_matrix):

    # Initialize vector representation
    vector_representation = {}

    # Create vector representation for each document
    for entry_id in tf_idf_matrix:

        # Initialize vector
        vector = []

        # Add tf-idf values to vector
        for term in tf_idf_matrix[entry_id]:
            vector.append(tf_idf_matrix[entry_id][term])

        # Add vector to vector representation
        vector_representation[entry_id] = vector

    return vector_representation


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

        # Create term frequency matrix
        term_frequency_matrix = self.create_term_frequency_matrix()

        # Create inverse document frequency vector
        inverse_document_frequency_vector = self.create_inverse_document_frequency_vector()

        # Create tf-idf matrix
        tf_idf_matrix = create_tf_idf_matrix(term_frequency_matrix, inverse_document_frequency_vector)

        # Create vector representation for each document
        vector_representation = create_vector_representation(tf_idf_matrix)

        return vector_representation

    def create_term_frequency_matrix(self):

        # Initialize term frequency matrix
        term_frequency_matrix = {}

        # Calculate term frequency for each document
        for entry_id in self.database:

            question, answer = self.database[entry_id]

            # Tokenize question
            question_tokens = self.tokenizer.tokenize(question)

            # Calculate term frequency
            question_term_frequency = calculate_term_frequency(question_tokens)

            terms_frequency = {}

            for term in self.terms.keys():

                # Initialize term frequency
                terms_frequency[term] = 0

                # Check if term is in question
                if term in question_term_frequency:
                    terms_frequency[term] = question_term_frequency[term]

            # Add term frequency vector to term frequency matrix
            term_frequency_matrix[entry_id] = terms_frequency

        return term_frequency_matrix

    def create_inverse_document_frequency_vector(self):

        # Initialize inverse document frequency vector
        inverse_document_frequency_vector = {}

        # Calculate inverse document frequency for each term
        for term in self.terms:
            inverse_document_frequency_vector[term] = len(self.terms[term])

        # Calculate total number of documents
        total_documents = len(self.database)

        # Calculate inverse document frequency
        for term in inverse_document_frequency_vector:
            inverse_document_frequency_vector[term] = math.log10(total_documents / inverse_document_frequency_vector[term])

        return inverse_document_frequency_vector


if __name__ == '__main__':

    # Prepare tokenizer
    tokenizer = Tokenizer(stopwords_path='stopwords.txt')

    # Prepare chatbot
    chatbot = Chatbot(tokenizer=tokenizer, path='datasets/small_ambigNQ.json')

    # Load database (training data)
    chatbot.load_database()

    print(chatbot.vector_representation)
