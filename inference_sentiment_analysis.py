import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

model = load_model('sentiment_analysis_model.h5')

input_text = input("Enter a movie review: ")

word_to_index = tokenizer.word_index
input_sequence = [word_to_index[word] if word in word_to_index else 0 for word in input_text.split()]
input_sequence = pad_sequences([input_sequence], maxlen=200)

prediction = model.predict(input_sequence)[0][0]

if prediction > 0.5:
    sentiment = "positive"
else:
    sentiment = "negative"

print("Predicted Sentiment:", sentiment)
print("Input Text:", input_text)