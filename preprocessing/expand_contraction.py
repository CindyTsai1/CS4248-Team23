from pycontractions import Contractions

def expand_contraction_preprocessing(sentence: str, cont: Contractions):
    return list(cont.expand_texts([sentence], precise=True))[0]

#print(expand_contraction_preprocessing("hi i'm Cindy's phone. You're ?"))