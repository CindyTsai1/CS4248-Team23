import pandas as pd

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

if False:
    data: pd.DataFrame = pd.DataFrame(pd.read_csv("/Users/yuwen/Desktop/NUS/Year5Sem2/CS4248/Project/CS4248-Team23/data/singlish_raw.csv"), columns=[COLUMN0])
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
    data.to_csv("/Users/yuwen/Desktop/NUS/Year5Sem2/CS4248/Project/CS4248-Team23/data/singlish_1.csv", index=False)

data: pd.DataFrame = pd.DataFrame(pd.read_csv("/Users/yuwen/Desktop/NUS/Year5Sem2/CS4248/Project/CS4248-Team23/data/singlish_1.csv"), columns=[COLUMN0, COLUMN1])
data['label'] = ''
data.to_csv("/Users/yuwen/Desktop/NUS/Year5Sem2/CS4248/Project/CS4248-Team23/data/singlish_2.csv", index=False)
