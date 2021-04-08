import re
def remove_link_preprocessing(sentence: str):
    # remove text after last # if present (deletes the whole message if there is only 1 hashtag at the start)
    sentence = re.sub(r'’', "'", sentence)
    # sentence = sentence[0:sentence.rfind("#")] if sentence.rfind("#") != -1 else sentence
    sentence = re.sub(r'\bhttp.*\/\w*\b', ' ', sentence)
    sentence = re.sub(r'#\w+', ' ', sentence)
    return sentence

#print(remove_link_preprocessing("[Copied from https://www.reddit.com/r/singapore/comments/kwzmno/crosspost_from_rsgexams_not_by_me_just_helping/]"))