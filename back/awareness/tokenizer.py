from nltk.tokenize import word_tokenize
import nltk

def tokenize_text(text):
    nltk.download("punkt", quiet=True)  # Descargar el tokenizador si no está presente
    tokens = word_tokenize(text)
    return tokens