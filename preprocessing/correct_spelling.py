from autocorrect import Speller

def correct_spelling_preprocessing(sentence: str, spell: Speller):
    return spell(sentence)

# print(fix_typos_preprocessing("I am veri angryyy."))