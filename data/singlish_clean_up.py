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

def duplicate_different_forms_of_same_word(row: pd.Series):
    if row[COLUMN0].index("/") != -1:
        words = row[COLUMN0].split("/")
        lst = [row]
        for i, word in enumerate(words):
            if i != 0:
                lst.append(row.copy())
            lst[i][COLUMN0] = word.strip()
        return pd.concat(lst, axis=0)
    return row

def duplicate_words_with_extra_h(row: pd.Series):
    if row[COLUMN0].index("(") != -1:
        words = row[COLUMN0].split("")
        lst = [row]
        for i, word in enumerate(words):
            if i != 0:
                lst.append(row.copy())
            lst[i][COLUMN0] = word.strip()
        return pd.concat(lst, axis=0)
    return row
    
data: pd.DataFrame = pd.DataFrame(pd.read_csv("/Users/yuwen/Desktop/NUS/Year5Sem2/CS4248/Project/CS4248-Team23/data/singlish_raw.csv"), columns=[COLUMN0])
data = data.dropna(axis = 0, subset=[COLUMN0], inplace=False)
data = data.drop(data[data[COLUMN0].map(str).map(lambda string: '[edit]' in string)].index)
data[COLUMN0]= data[COLUMN0].apply(data_clean)
data[COLUMN1] = data[COLUMN0].copy()
data[COLUMN0] = data[COLUMN0].apply(data_clean0)
data[COLUMN1] = data[COLUMN1].apply(data_clean1)
data.to_csv("/Users/yuwen/Desktop/NUS/Year5Sem2/CS4248/Project/CS4248-Team23/data/singlish_1.csv", index=False)