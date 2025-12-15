import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM

# --- MODIFICATION 1: NEW INPUT TEXT ---
# Changed the text to a more complex technical sentence.
# This proves the model can learn different patterns.
text = "The handsome boy whom I met last time is very intelligent also"

# Create character mappings
chars = sorted(list(set(text)))
char_to_index = {char: i for i, char in enumerate(chars)}
index_to_char = {i: char for i, char in enumerate(chars)}

# --- MODIFICATION 2: HYPERPARAMETERS ---
seq_length = 30 # Adjusted sequence length
sequences = []
labels = []

# Prepare the dataset
for i in range(len(text) - seq_length):
    seq = text[i:i + seq_length]
    label = text[i + seq_length]
    sequences.append([char_to_index[char] for char in seq])
    labels.append(char_to_index[label])

X = np.array(sequences)
y = np.array(labels)

# One-hot encoding
X_one_hot = tf.one_hot(X, len(chars))
y_one_hot = tf.one_hot(y, len(chars))

# Length of text to generate
text_len = 150

# --- MODIFICATION 3: ARCHITECTURE UPGRADE ---
model = Sequential()
# Replaced SimpleRNN with LSTM (128 units)
# LSTM is smarter at remembering context ("Long Short-Term Memory")
model.add(LSTM(128, input_shape=(seq_length, len(chars)))) 
model.add(Dense(len(chars), activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting Training (LSTM)...")
# Reduced epochs to 50 as LSTM learns faster than SimpleRNN
model.fit(X_one_hot, y_one_hot, epochs=400, verbose=1) 

# Test the model
start_seq = text[:seq_length]
print(f"Seed: {start_seq}")

generated_text = start_seq

for i in range(text_len):
    x = np.array([[char_to_index[char] for char in generated_text[-seq_length:]]])
    x_one_hot = tf.one_hot(x, len(chars))
    
    # Predict next character
    prediction = model.predict(x_one_hot)
    next_index = np.argmax(prediction)
    next_char = index_to_char[next_index]
    
    generated_text += next_char

print("Generated Text:")
print(generated_text)