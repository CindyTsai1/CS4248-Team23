def is_not_reply(sentence: str):
    is_not_reply: int = 1
    if sentence.lower().startswith(("responding", "reply", "replying", "response", "to #", "to the op", "to op", "to post", "to poster")):
        is_not_reply = 0
    return is_not_reply
