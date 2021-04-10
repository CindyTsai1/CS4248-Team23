import pandas as pd

def singlish_feature(train: pd.Series):
    data: pd.DataFrame = pd.DataFrame(pd.read_csv("data/singlish_2.csv"), columns=['word', 'description', 'label'])
    data = data.drop(columns=['description'])
    dict_data:dict = data.set_index('word').to_dict()['label']
    #print(dict_data)
    return train.apply(singlish_score, args=(dict_data,))

def singlish_score(sentence: str, dict_data:dict):
    negativity: int = 0
    for word in sentence.split():
        negativity += int(dict_data.get(word, 0))
    return negativity

#print(singlish_feature(pd.Series(['4d ulu', 'lor abuden'])))