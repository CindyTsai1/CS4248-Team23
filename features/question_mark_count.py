def is_not_reply(sentence: str):
    question_mark_count: int = 0
    question_mark_count = sentence.count("??") + sentence.count("!?")
    return question_mark_count
