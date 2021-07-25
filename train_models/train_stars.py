import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import pandas as pd
import numpy as np
import model
from tensorflow.keras.utils import to_categorical

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Dataset for star prediction
df = pd.read_csv(r'./dataset.csv', sep=';')
text = np.asarray(df['text'].values)
text = text.reshape(-1, 1)


# Output
stars = np.asarray(df['stars'].values).astype(np.float64)
stars = stars.reshape(-1, 1)
stars = to_categorical(stars)


def train(model, xtrain, ytrain, epochs):

    model.fit(xtrain,
              ytrain,
              epochs=epochs,
              steps_per_epoch=20,
              validation_split=0.25,
              batch_size=10
              )


print('text.shape', text.shape)

text = tf.convert_to_tensor(text, dtype=tf.string)

models = model.NLP_model(text)
model_stars = models.model_stars()

"""
TRAINING MODEL
"""
train(model_stars,
      text,
      tf.convert_to_tensor(stars),
      200)


model_path = './saved_models/stars/model_stars'
model_stars.save(model_path, save_format='tf')


"""
TESTING MODEL
"""

# model_stars = tf.keras.models.load_model('saved_models/stars/model_stars')
#
# inputs = 'Excelente aplicacion, me encanta, gran trabajo'
# inputs = tf.expand_dims(inputs, axis=0)
# print(inputs.shape)
#
# pred = model_stars.predict(inputs)
# print(np.argmax(pred))
