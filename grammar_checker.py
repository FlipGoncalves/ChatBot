import nltk
import joblib

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from nltk import pos_tag, word_tokenize, RegexpParser

class GrammarChecker(object):

    def __init__(self):

        self.grammarEN = nltk.CFG.fromstring("""
            S -> NP VP
            S -> VP
            PP -> P NP
            NP -> DT NP | N PP | N | ADJ N | ADV ADJ N | ADV ADJ | DT N | P | N NP
            VP -> V NP | V PP | V NP PP | V | V PRT NP | V VP
            ADJ -> 'ADJ'
            ADV -> 'ADV'
            DT -> 'DET'
            N -> 'NOUN'
            P -> 'PRON'
            V -> 'VERB'
            PRT -> 'PRT'
            X -> 'X'
            """)
        
        self.grammarPT = nltk.CFG.fromstring("""
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

        self.parserPT = nltk.parse.ChartParser(grammar=self.grammarPT)

        self.parserEN = nltk.parse.ChartParser(grammar=self.grammarEN)

        self.portuguese_tagger = joblib.load('POS_tagger_unigram.pkl')

    def check_grammar(self, sentence:str):
        
        #EN
        sentence = sentence.strip('.,;:!?()[]{}')
        tokenization = word_tokenize(sentence)
        taggedEN = pos_tag(tokenization, tagset="universal")

        new_sentenceEN = ""
        for t in taggedEN:
            new_sentenceEN = new_sentenceEN + t[1] + " "

        parsedEN = self.parserEN.parse(new_sentenceEN.split())

        for p in parsedEN:
            return p
        
        #PT

        taggedPT = self.portuguese_tagger.tag(tokenization)

        new_sentencePT = ""
        for t in taggedPT:
            new_sentencePT = new_sentencePT + t[1] + " "

        parsedPT = self.parserPT.parse(new_sentencePT.split())

        for p in parsedPT:
            return p
        
        return None
