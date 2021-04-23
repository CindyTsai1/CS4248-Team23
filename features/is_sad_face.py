def is_sad_face(sentence: str):
    sad_faces = [":(", ":-(", ";-(", ";(", "T.T", "T^T", ":<", ":c"]
    is_sad_face: int = 0
    if any(x in sentence for x in sad_faces):
        is_sad_face = 1
    return is_sad_face
