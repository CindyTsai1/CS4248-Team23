from re import *

def remove_non_english_preprocessing(sentence: str):
    # filter away non-english by substituting them with " "
    sentence = sub("([^\x00-\x7F])+", " ", sentence)
    return sentence
