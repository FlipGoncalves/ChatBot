import nltk

nltk.download('large_grammars')
grammar1 = nltk.data.load('grammars/large_grammars/atis.cfg')
grammar2 = nltk.data.load('grammars/large_grammars/commandtalk.cfg')

print(grammar2)

parser = nltk.parse.ChartParser(grammar=grammar2)
test = parser.chart_parse("OK thank you")
print(test)