import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np

# Asume que los datos de entrenamiento están en la forma de listas de pares mensaje-respuesta
data = [
    ("hello", "Hi there!"),
    ("how are you", "I'm fine, thank you."),
    # Más pares de mensaje-respuesta
]   

# Preprocesar y tokenizar los datos
input_texts, target_texts = zip(*data)
input_texts = [preprocess_text(text) for text in input_texts]
target_texts = [preprocess_text(text) for text in target_texts]

# Crear un tokenizador y ajustarlo a los datos de entrenamiento
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_texts + target_texts)
vocab_size = len(tokenizer.word_index) + 1

# Convertir las palabras a índices y rellenar las secuencias para tener la misma longitud
input_sequences = tokenizer.texts_to_sequences(input_texts)
target_sequences = tokenizer.texts_to_sequences(target_texts)
max_sequence_length = max(max(len(seq) for seq in input_sequences), max(len(seq) for seq in target_sequences))
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding="post")
target_sequences = pad_sequences(target_sequences, maxlen=max_sequence_length, padding="post")

# Dividir el conjunto de datos en conjuntos de entrenamiento y validación
input_train, input_val, target_train, target_val = train_test_split(input_sequences, target_sequences, test_size=0.2)

# Crear el modelo
model = Sequential()
model.add(Embedding(vocab_size, 256, input_length=max_sequence_length))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Dense(vocab_size, activation="softmax"))

# Compilar y entrenar el modelo
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()
history = model.fit(input_train, np.expand_dims(target_train, -1), validation_data=(input_val, np.expand_dims(target_val, -1)), epochs=50, batch_size=32)

# Guardar el modelo y el tokenizador
model.save("path/to/your/saved/model")
with open("path/to/your/saved/tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
