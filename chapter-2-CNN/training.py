#! /usr/bin/env python3

# Import Libraries
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Flatten
from keras.preprocessing.image import ImageDataGenerator


# Neuronal Network Initialization
classifier = Sequential()

# Convolution layer
classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1, activation="relu", input_shape=(64, 64, 3)))

# Polling layer
classifier.add(MaxPooling2D(pool_size=2))

# Add Second Convolution and Polling Layers
classifier.add(Convolution2D(filters=32, kernel_size=3, strides=1, activation="relu"))
classifier.add(MaxPooling2D(pool_size=2))


# Flattening layer
classifier.add(Flatten())

# Add fully connected layers
classifier.add(Dense(units=128, activation="relu"))
classifier.add(Dense(units=1, activation="sigmoid"))

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# Data Augmenting
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(
        'chapter-2-CNN/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'chapter-2-CNN/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=250,  # 8000/32
        epochs=20,
        validation_data=test_set,
        validation_steps=63,  # 2000/32
)

print(train_set.class_indices)
classifier.save('chapter-2-CNN/weight.h5')
