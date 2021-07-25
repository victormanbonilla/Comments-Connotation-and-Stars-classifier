import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import model

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Dataset for positive - negative model
df = pd.read_csv(r'./dataset.csv', sep=';')
x_train = np.array(df['text'].values); print(x_train.shape)
y_train = np.array(df['rate'].values); print(y_train.shape)

x_train = x_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)


def train(model, xtrain, ytrain, epochs):

    model.fit(xtrain,
              ytrain,
              epochs=epochs,
              steps_per_epoch=20,
              validation_split=0.2,
              batch_size=10
              )


models = model.NLP_model(x_train)
model_binary = models.model_positive_negative()

train(model_binary,
      x_train,
      y_train,
      50)

model_path = './saved_models/binary/model_binary'
model_binary.save(model_path, save_format='tf')



