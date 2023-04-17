import json
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from nltk import pos_tag, word_tokenize, RegexpParser

with open("datasets\DataSet1.json", 'r', encoding='utf-8') as dataset:

    # Load dataset
    data = json.load(dataset)

    grammar = nltk.CFG.fromstring("""
    S -> NP VP
    PP -> P NP
    NP -> DT N | N PP | N | ADJ N | ADV ADJ N | ADV ADJ
    VP -> V NP | V PP | V NP PP | V
    Adj -> 'ADJ'
    Adv -> 'ADV'
    DT -> 'DET'
    N -> 'NOUN'
    P -> 'PRON'
    V -> 'VERB'
    """)

    parser = nltk.parse.ChartParser(grammar=grammar)

    for item in data["intents"]:
        print(item["intent"])
        for languages in item["text"]:
            for sentence in languages["english"]:
                print(sentence)
                sentence = sentence.strip('.,;:!?()[]{}')
                tokenization = word_tokenize(sentence)
                # Find all parts of speech in above sentence
                tagged = pos_tag(tokenization, tagset="universal")
                print(tagged)
                # for test in parser.parse(tokenization):
                #     print(test)
                print()
        print()