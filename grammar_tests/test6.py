import nltk
import joblib

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from nltk import pos_tag, word_tokenize, RegexpParser

grammar = nltk.CFG.fromstring("""
    S -> NP VP
    S -> VP
    S -> S CONJ S
    PP -> P NP
    NP -> DT NP | N PP | N | ADJ N | ADV ADJ N | ADV ADJ | DT N | P | PREP N | N NP
    VP -> V NP | V PP | V NP PP | V
    ADJ -> 'ADJ'
    ADV -> 'ADV' | 'ADV-KS' | 'ADV-KS-REL'
    DT -> 'ART'
    N -> 'N' | 'NPROP'
    P -> 'PROADJ' | 'PRO-KS' | 'PROPESS' | 'PRO-KS-REL' | 'PROSUB'
    PREP -> 'PREP' | 'PREP|+'
    V -> 'V' | 'VAUX'
    CONJ -> 'KS' | 'KC'
    """)

sentence = "THIS is shit"

parser = nltk.parse.ChartParser(grammar=grammar)

sentence = sentence.strip('.,;:!?()[]{}')
tokenization = word_tokenize(sentence)
teste_tagger = joblib.load('POS_tagger_unigram.pkl')
tagged = teste_tagger.tag(word_tokenize(sentence))

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