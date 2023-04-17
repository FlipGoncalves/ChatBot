import nltk
import joblib

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from nltk import pos_tag, word_tokenize, RegexpParser

grammarEN = nltk.CFG.fromstring("""
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

grammarPT = nltk.CFG.fromstring("""
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

sentence = "I love chicken wings"

parserPT = nltk.parse.ChartParser(grammar=grammarPT)

parserEN = nltk.parse.ChartParser(grammar=grammarEN)

print()
print("EN")
print()

sentence = sentence.strip('.,;:!?()[]{}')
tokenization = word_tokenize(sentence)
tagged = pos_tag(tokenization, tagset="universal")

print(tagged)

new_sentence = ""
for t in tagged:
    print(t[1])
    new_sentence = new_sentence + t[1] + " "
print(new_sentence)

parsed = parserEN.parse(new_sentence.split())

print("TESTE")
for p in parsed:
    print("ga")
    print(p)

print()
print("PT")
print()

teste_tagger = joblib.load('POS_tagger_unigram.pkl')
tagged = teste_tagger.tag(word_tokenize(sentence))

print(tagged)

new_sentence = ""
for t in tagged:
    print(t[1])
    new_sentence = new_sentence + t[1] + " "
print(new_sentence)

parsed = parserPT.parse(new_sentence.split())