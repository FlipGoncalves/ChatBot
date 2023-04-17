import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from nltk import pos_tag, word_tokenize, RegexpParser

grammar = nltk.CFG.fromstring("""
    S -> NP VP
    S -> VP
    PP -> P NP
    NP -> DT NP | N PP | N | ADJ N | ADV ADJ N | ADV ADJ | DT N | P
    VP -> V NP | V PP | V NP PP | V
    ADJ -> 'ADJ'
    ADV -> 'ADV'
    DT -> 'DET'
    N -> 'NOUN'
    P -> 'PRON'
    V -> 'VERB'
    """)

sentence = "You are a very clever girl"

parser = nltk.parse.ChartParser(grammar=grammar)

sentence = sentence.strip('.,;:!?()[]{}')
tokenization = word_tokenize(sentence)
tagged = pos_tag(tokenization, tagset="universal")

print(tagged)

new_sentence = ""
for t in tagged:
    print(t[1])
    new_sentence = new_sentence + t[1] + " "
print(new_sentence)

parsed = parser.parse(new_sentence.split())

print("TESTE")
for p in parsed:
    print("ga")
    print(p)