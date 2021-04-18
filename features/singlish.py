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
        # if word in dict_data: print(word, dict_data[word])
        negativity += int(dict_data.get(word.lower(), 0))
    # print(sentence)
    # print(negativity)
    return negativity

#print(singlish_feature(pd.DataFrame({'text':['c1010e post talk punishment go affect employment etc go affect employment whatsoever directly correct wrong punishment give prof go reflect directly transcript violation conduct right get 0 px1 module student able Px2 final come accept learn mistake work hard know b b bad right course choose entirely fuck know employer interpret fuck transcript idea extent damage AP surprise pretty severe significant quit give pathetic excuse threaten prof go work reflect badly human']})))