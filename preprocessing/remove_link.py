def remove_link_preprocessing(sentence: str):
    # remove text after last # if present
    sentence = sentence[0:sentence.rfind("#")] if sentence.rfind("#") != -1 else sentence
    return sentence
