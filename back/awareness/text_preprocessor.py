import string

def preprocess_text(text):
    text = text.lower().strip()
    text = remove_punctuation(text)
    return text

def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))