import csv
'''
with open('preprocessing/slang.txt', 'r') as myCSVfile:
    short_form_dict = dict([pair for pair in csv.reader(myCSVfile, delimiter="=")])
'''
def expand_short_form_preprocessing(sentence: str, short_form_dict):
    words = sentence.split(" ")
    for i, word in enumerate(words):
        if word.upper() in short_form_dict:
            words[i] = short_form_dict[word.upper()].lower()
    return ' '.join(words)

# print(expand_short_form_preprocessing('brb', short_form_dict))