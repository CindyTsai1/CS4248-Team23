from pycontractions import Contractions

#cont: Contractions = Contractions('/Users/yuwen/Desktop/NUS/Year5Sem2/CS4248/Project/GoogleNews-vectors-negative300.bin.gz')
#cont.load_models()
def expand_contraction_preprocessing(sentence: str, cont: Contractions):
    return list(cont.expand_texts([sentence], precise=True))[0]

#print(expand_contraction_preprocessing("hi i'm Cindy's phone. You're ?", cont))