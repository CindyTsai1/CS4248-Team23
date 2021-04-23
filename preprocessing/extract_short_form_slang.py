import pandas as pd
import re
import csv
import string
import nltk

def preprocessing(sentence: str):
    # remove text after last # if present (deletes the whole message if there is only 1 hashtag at the start)
    sentence = re.sub(r'’', "'", sentence)
    # sentence = sentence[0:sentence.rfind("#")] if sentence.rfind("#") != -1 else sentence
    sentence = re.sub(r'\bhttp.*\/\w*\b', ' ', sentence)
    sentence = re.sub(r'#\w+', ' ', sentence)
    sentence = sentence.replace("\n", " ")
    sentence = re.sub(r'(\b\d+([\.\,]\d+)*\b)|(\b([a-z]?|[a-z][a-z][a-z][a-z]+)\d\w*\b)', '', sentence)
    sentence = re.sub("([^\x00-\x7F])+", " ", sentence)
    regex = re.compile('[%s]' % re.escape(string.punctuation+"“”‘’"))
    return [t for word in sentence.split() for t in re.split(regex, word)]

def extract_slang():
    old_train: pd.DataFrame = pd.read_csv('data/v5.csv')
    old_train = old_train.dropna(axis = 0, subset=['text'], inplace=False)
    text = old_train['text'].values.tolist()
    text = [t for txt in text for t in preprocessing(txt)]
    lst = [word.upper() for txt in text for word in re.findall(r'^[^aeiou]+$', txt.lower()) if word != '']
    with open('preprocessing/slang.txt', 'r') as myCSVfile:
        short_form_dict:dict = dict([pair for pair in csv.reader(myCSVfile, delimiter="=")])
    st = set(lst) - set(short_form_dict.keys())
    return '\n'.join([word+'=' for word in st])

print(extract_slang())