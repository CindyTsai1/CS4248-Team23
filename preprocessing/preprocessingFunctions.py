import re
import string
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords

def convert_to_lowercase(sentence):
    lowercase = sentence.lower()
    return lowercase


def word_tokenizer(sentence):
    tokenized_word = word_tokenize(sentence)
    return tokenized_word

def sent_tokenizer(sentence):
    tokenized_sent = sent_tokenize(sentence)
    return tokenized_sent


regex = re.compile('[%s]' % re.escape(string.punctuation+"“”’"))

def punctuation_removal(tokens):
    tokenized_no_punc = [] #list of tokenized words
    for token in tokens:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            tokenized_no_punc.append(new_token)
    return tokenized_no_punc

def stopwords_removal(tokens):
    tokenized_no_stopwords = []
    for token in tokens:
        if not token.lower() in stopwords.words('english'):
            tokenized_no_stopwords.append(token)
    return tokenized_no_stopwords

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

porter = PorterStemmer()
wordnet = WordNetLemmatizer()

preprocessed_docs = []
def stemming(tokens):
    stemmed_tokens = []
    for token in tokens:
        stemmed_tokens.append(wordnet.lemmatize(token))
    return stemmed_tokens

def hashtag_extractions(sentence):
    hashtag_list = []
    for token in sentence.split():
        if token[0] == '#':
            hashtag_list.append(token)
    return hashtag_list
