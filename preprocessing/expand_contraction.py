from pycontractions import Contractions

# cont: Contractions = Contractions('../GoogleNews-vectors-negative300.bin.gz')
# cont.load_models()
def expand_contraction_preprocessing(sentence: str, cont: Contractions):
    return ' '.join(list(cont.expand_texts([sentence], precise=False)))

# print(expand_contraction_preprocessing("“I don’t think we’re emotionally compatible and u deserve someone with the maturity to handle u”\nDid my ex just call me a crazy bitch when he broke up with me?\n-", cont))