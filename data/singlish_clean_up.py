import pandas as pd
import re
import string

regex = re.compile('[%s]' % re.escape(string.punctuation+"“”’"))

def punctuation_removal(tokens):
    tokenized_no_punc = [] #list of tokenized words
    for token in tokens:
        new_token = regex.sub(u'', token)
        if not new_token == u'':
            tokenized_no_punc.append(new_token)
    return tokenized_no_punc

COLUMN0: str = 'word'
COLUMN1: str = 'description'

def data_clean(sentence: str):
    return sentence.strip().lower()

def data_clean0(sentence: str):
    return sentence.split(' - ')[0].strip()

def data_clean1(sentence: str):
    if ' - ' in sentence:
        return sentence.split(' - ')[1].strip()
    else:
        return sentence.strip()

def duplicate_different_forms_of_same_word(df: pd.DataFrame):
    for index, row in df.iterrows():
        words = row[COLUMN0].split("/")
        if len(words) == 1:
            continue
        for i, word in enumerate(words):
            df.loc[index+(1/len(words))*i] = row.copy()
            df.loc[index+(1/len(words))*i][COLUMN0] = word.strip()
    df = df.sort_index().reset_index(drop=True)
    return df


def duplicate_words_with_extra_h(df: pd.DataFrame, noMoreBracket: bool):
    noMoreBracket = True
    for index, row in df.iterrows():
        count = row[COLUMN0].count("(")
        if count == 0:
            continue
        noMoreBracket = False
        print(row[COLUMN0])
        i = row[COLUMN0].index("(")
        j = row[COLUMN0].index(")")
        df.loc[index+0.5] = row.copy()
        df.loc[index+0.5][COLUMN0] = df.loc[index+0.5][COLUMN0][:i]+df.loc[index+0.5][COLUMN0][i+1:j]+df.loc[index+0.5][COLUMN0][j+1:]
        df.loc[index][COLUMN0] = df.loc[index][COLUMN0][:i]+df.loc[index][COLUMN0][j+1:]
    df = df.sort_index().reset_index(drop=True)
    return df, noMoreBracket

def remove_punc(df: pd.DataFrame):
    for index, row in df.iterrows():
        df.loc[index][COLUMN0] = ' '.join(punctuation_removal(df.loc[index][COLUMN0].split()))
    return df

if False:
    data: pd.DataFrame = pd.DataFrame(pd.read_csv("data/singlish_raw.csv"), columns=[COLUMN0])
    data = data.dropna(axis = 0, subset=[COLUMN0], inplace=False)
    data = data.drop(data[data[COLUMN0].map(str).map(lambda string: '[edit]' in string or ' - ' not in string)].index)
    data[COLUMN0] = data[COLUMN0].apply(data_clean)
    data[COLUMN1] = data[COLUMN0].copy()
    data[COLUMN0] = data[COLUMN0].apply(data_clean0)
    data[COLUMN1] = data[COLUMN1].apply(data_clean1)
    data = duplicate_different_forms_of_same_word(data)
    noMoreBracket: bool = False
    while not noMoreBracket:
        data, noMoreBracket = duplicate_words_with_extra_h(data,noMoreBracket)
    data.to_csv("data/singlish_1.csv", index=False)

data: pd.DataFrame = pd.DataFrame(pd.read_csv("data/singlish_1.csv"), columns=[COLUMN0, COLUMN1])
data['label'] = ''
data[COLUMN0] = data[COLUMN0].apply(data_clean)
data = duplicate_different_forms_of_same_word(data)
noMoreBracket: bool = False
while not noMoreBracket:
    data, noMoreBracket = duplicate_words_with_extra_h(data,noMoreBracket)
data = remove_punc(data)
data.to_csv("data/singlish_2.csv", index=False)
