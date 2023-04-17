
import nltk

nltk.download('treebank')

#print(nltk.corpus.treebank.parsed_sents()[:10])
ruleset = set(rule for tree in nltk.corpus.treebank.parsed_sents()[:10] for rule in tree.productions())

for rule in ruleset:
    print(rule)