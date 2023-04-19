# ChatBot 

## Tokenizer

The Tokenizer is a class that takes a sentence and returns a list of tokens.
The ChatBot uses the Tokenizer to split the user's input into tokens, which are then used to find the best match for the user's input.

It supports the following features:

- Split a sentence into tokens
- Remove punctuation and special characters
- Remove accents
- Ignore words that don't meet a minimum length
- Remove stop words
- Normalize words to lowercase
- Stem words 
- Lemmatize words

## Setup
The program needs to run the following commands in a python compiler:
python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
