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
            S -> ADJ N
            S -> N P 
            S -> NP VP
            S -> VP
            S -> ADV VP
            S -> V P ADP N
            S -> P VP
            PP -> P NP
            NP -> DT NP | N PP | N | ADJ N | ADV ADJ N | ADV ADJ | DT N | P | N NP | PP
            VP -> V NP | V PP | V NP PP | V | V PRT NP | V VP | P VP | V ADJ | ADP VP | PRT V | PRT V VP | V ADV VP | V ADP
            ADJ -> 'ADJ'
            ADV -> 'ADV'
            DT -> 'DET'
            N -> 'NOUN'
            P -> 'PRON'
            V -> 'VERB'
            PRT -> 'PRT'
            X -> 'X'
            ADP -> 'ADP'
            """)
        
        self.grammarPT = nltk.CFG.fromstring("""
            S -> NOUN
            S -> ADJ NOUN
            S -> NP VP
            S -> VP
            S -> NOUN P
            S -> S CONJ S
            S -> CONJ NP VP
            S -> CONJ VP
            S -> CONJ NP
            S -> ADV PCP
            S -> PCP VP
            S -> ADV NP
            PP -> P NP
            NP -> DT NP | NOUN PP | NOUN | ADJ NOUN | ADV ADJ NOUN | ADV ADJ | DT NOUN | P | PREP NOUN | NOUN NP | PP | NOUN ADV
            VP -> VERB NP | VERB PP | VERB NP PP | VERB | VERB VP | P VP | VERB ADJ | VERB PREP P | PREP VP | VP P | ADV VP
            ADJ -> 'ADJ'
            ADV -> 'ADV' | 'ADV-KS' | 'ADV-KS-REL'
            DT -> 'ART'
            NOUN -> 'N' | 'NPROP'
            P -> 'PROADJ' | 'PRO-KS' | 'PROPESS' | 'PRO-KS-REL' | 'PROSUB'
            PREP -> 'PREP' | 'PREP|+'
            VERB -> 'V' | 'VAUX'
            CONJ -> 'KS' | 'KC'
            EST -> 'N|EST'
            PCP -> 'PCP'
            """)

        self.parserPT = nltk.parse.ChartParser(grammar=self.grammarPT)

        self.parserEN = nltk.parse.ChartParser(grammar=self.grammarEN)

        self.portuguese_tagger = joblib.load('POS_tagger_unigram.pkl')

    def check_grammar(self, sentence:str):

        print("Sentence: " + sentence)
        
        sentence = sentence.replace('<NULL>', '')
        #EN
        sentence = sentence.strip('.,;:!?()[]{}')
        tokenization = word_tokenize(sentence)
        taggedEN = pos_tag(tokenization, tagset="universal")
        
        new_sentenceEN = ""
        for t in taggedEN:
            if t[0] not in ".,;:!?()[]{}":
                new_sentenceEN = new_sentenceEN + t[1] + " "

        print("EN sentence: " + new_sentenceEN)

        try:
            parsedEN = self.parserEN.parse(new_sentenceEN.split())
            for p in parsedEN:
                tree_string = str(p)
                for token in taggedEN:
                    tree_string = tree_string.replace(token[1] + ")", token[0] + ")", 1)
                tree = nltk.tree.Tree.fromstring(tree_string)
                return tree
        except:
            pass
        #PT

        taggedPT = self.portuguese_tagger.tag(tokenization)

        new_sentencePT = ""
        for t in taggedPT:
            if t[0] not in ".,;:!?()[]{}":
                new_sentencePT = new_sentencePT + t[1] + " "

        print("PT sentence: " + new_sentencePT)

        try:
            parsedPT = self.parserPT.parse(new_sentencePT.split())

            for p in parsedPT:
                tree_string = str(p)
                for token in taggedPT:
                    tree_string = tree_string.replace(token[1] + ")", token[0] + ")", 1)
                tree = nltk.tree.Tree.fromstring(tree_string)
                return tree
        except:
            pass
        
        return None
