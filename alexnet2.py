# -*- coding: utf-8 -*-
"""
Modified AlexNet for Assignment
Changes:
1. Added compilation and training loop.
2. Generating dummy data to run without external dataset.
3. Modified Dropout to 0.4 and Classes to 10.
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np

class AlexNet(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()

        # First Convolutional Layer
        self.add(Conv2D(96, kernel_size=(11, 11), strides=4, padding='valid', activation='relu', input_shape=input_shape))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

        # Second Convolutional Layer
        self.add(Conv2D(256, kernel_size=(5, 5), strides=1, padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

        # Third, Fourth, and Fifth Convolutional Layers
        self.add(Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.add(Conv2D(384, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.add(Conv2D(256, kernel_size=(3, 3), strides=1, padding='same', activation='relu'))
        self.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid'))

        # Flatten Layer
        self.add(Flatten())

        # Fully Connected Layers
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.4)) # CHANGE: Modified Dropout from 0.5 to 0.4
        self.add(Dense(4096, activation='relu'))
        self.add(Dropout(0.4)) # CHANGE: Modified Dropout from 0.5 to 0.4
        self.add(Dense(num_classes, activation='softmax')) # Output layer

# --- EXECUTION CODE (New) ---

# 1. Setup Parameters
input_shape = (224, 224, 3) 
num_classes = 10 # CHANGE: Changed from 1000 to 10 to simulate a smaller task

# 2. Instantiate Model
model = AlexNet(input_shape, num_classes)

# 3. Compile Model (Required to run)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Generate Dummy Data (To make it runnable instantly)
print("Generating dummy data...")
# Create 20 random images of size 224x224x3
X_train = np.random.random((20, 224, 224, 3)) 
# Create 20 random labels (integers 0-9)
Y_train_labels = np.random.randint(0, num_classes, (20,)) 
# Convert labels to one-hot encoding
Y_train = tf.keras.utils.to_categorical(Y_train_labels, num_classes)

# 5. Train the model
print("Starting training test...")
model.fit(X_train, Y_train, epochs=4, batch_size=2)

model.summary()