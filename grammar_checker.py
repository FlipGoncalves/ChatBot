import nltk
import joblib

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from nltk import pos_tag, word_tokenize, RegexpParser

class GrammarChecker(object):

    def __init__(self):

        self.grammarEN = nltk.CFG.fromstring("""
            S -> N
            S -> ADJT N
            S -> NP VP
            S -> VP
            PP -> P NP
            NP -> DT NP | N PP | N | ADJT N | ADVB ADJT N | ADVB ADJT | DT N | P | N NP
            VP -> V NP | V PP | V NP PP | V | V PRTT NP | V VP
            ADJT -> 'ADJ'
            ADVB -> 'ADV'
            DT -> 'DET'
            N -> 'NOUN'
            P -> 'PRON'
            V -> 'VERB'
            PRTT -> 'PRT'
            X -> 'X'
            """)
        
        self.grammarPT = nltk.CFG.fromstring("""
            S -> NOUN
            S -> ADJT NOUN
            S -> NP VP
            S -> VP
            S -> S CONJ S
            PP -> P NP
            NP -> DT NP | NOUN PP | NOUN | ADJT NOUN | ADVB ADJT NOUN | ADVB ADJT | DT NOUN | P | PREPP NOUN | NOUN NP
            VP -> VERB NP | VERB PP | VERB NP PP | VERB
            ADJT -> 'ADJ'
            ADVB -> 'ADV' | 'ADV-KS' | 'ADV-KS-REL'
            DT -> 'ART'
            NOUN -> 'N' | 'NPROP'
            P -> 'PROADJ' | 'PRO-KS' | 'PROPESS' | 'PRO-KS-REL' | 'PROSUB'
            PREPP -> 'PREP' | 'PREP|+'
            VERB -> 'V' | 'VAUX'
            CONJ -> 'KS' | 'KC'
            """)

        self.parserPT = nltk.parse.ChartParser(grammar=self.grammarPT)

        self.parserEN = nltk.parse.ChartParser(grammar=self.grammarEN)

        self.portuguese_tagger = joblib.load('POS_tagger_unigram.pkl')

    def check_grammar(self, sentence:str):
        
        sentence = sentence.replace('<NULL>', '')
        #EN
        sentence = sentence.strip('.,;:!?()[]{}')
        tokenization = word_tokenize(sentence)
        taggedEN = pos_tag(tokenization, tagset="universal")
        
        new_sentenceEN = ""
        for t in taggedEN:
            new_sentenceEN = new_sentenceEN + t[1] + " "

        parsedEN = self.parserEN.parse(new_sentenceEN.split())

        for p in parsedEN:
            tree_string = str(p)
            for token in taggedEN:
                tree_string = tree_string.replace(token[1], token[0], 1)
            tree = nltk.tree.Tree.fromstring(tree_string)
            return tree
        
        #PT

        taggedPT = self.portuguese_tagger.tag(tokenization)

        new_sentencePT = ""
        for t in taggedPT:
            new_sentencePT = new_sentencePT + t[1] + " "

        parsedPT = self.parserPT.parse(new_sentencePT.split())

        for p in parsedPT:
            tree_string = str(p)
            for token in taggedPT:
                tree_string = tree_string.replace(token[1], token[0], 1)
            tree = nltk.tree.Tree.fromstring(tree_string)
            return tree
        
        return None
