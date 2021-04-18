import spacy

def lemmatization_preprocessing(sentence: str, nlp:spacy.Language):
    # SpaCy lemmatization with stop words filtered first
    #nlp = spacy.load('en_core_web_sm',disable=['parser', 'ner'])
    #sentence = ' '.join([token.lemma_ for token in list(nlp(sentence)) if (token.is_stop==False)])
    sentence = ' '.join([token.lemma_ for token in list(nlp(sentence))])
    return sentence
