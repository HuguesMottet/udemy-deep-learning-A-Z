#! /usr/bin/env python3

# Import Libraries
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

classifier = load_model('chapter-2-CNN/weight.h5')

classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


test_image = image.load_img('chapter-2-CNN/dataset/single_prediction/cat_or_dog_2.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = classifier.predict(test_image)

if result[0][0] == 1:
    print('dog')
else:
    print('cat')


