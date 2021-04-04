from re import *

def remove_digit_preprocessing(sentence: str):
    # remove words with digits, numeric words, keeping possible nus module code
    sentence = sub(r'(\b\d+([\.\,]\d+)*\b)|(\b([a-z]?|[a-z][a-z][a-z][a-z]+)\d\w*\b)', '', sentence)
    return sentence
