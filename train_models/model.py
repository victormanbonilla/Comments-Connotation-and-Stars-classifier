import tensorflow as tf 
from tensorflow.keras import layers
from prepare_data import preprocessing


class NLP_model:
    
    def __init__(self, x_train):
        self.x_train = x_train
    
    max_vocab_length = 10000  # Max number of words to have un our vocabulary.
    max_length = 15  # Max length our sequences will be.   

    def model_stars(self):        
        
        embedding = layers.Embedding(input_dim=self.max_vocab_length,
                                     output_dim=128,
                                     input_length=self.max_length)
        
        obj_prepare = preprocessing(self.max_vocab_length,
                                    self.max_length)
        
        text_vectorizer = obj_prepare.preprocess_text(self.x_train)

        input2 = layers.Input(shape=(1,), name='Connotation')

        input1 = layers.Input(shape=(1,), dtype=tf.string, name='Text_Input')
        x = text_vectorizer(input1)
        x = embedding(x)
        x = layers.LSTM(units=512, return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(units=256)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)

        outputs = layers.Dense(6, activation='softmax')(x)

        model = tf.keras.Model(inputs=input1,
                               outputs=outputs,
                               name='LSTM_model2')

        model.compile(loss='categorical_crossentropy',
                      optimizer='Adam',
                      metrics=['acc']
                      )
        # model.summary()
        return model

    def model_positive_negative(self):        
    
        embedding = layers.Embedding(input_dim=self.max_vocab_length,
                                     output_dim=128,
                                     input_length=self.max_length)
        
        obj_prepare = preprocessing(self.max_vocab_length,
                                    self.max_length)
        
        text_vectorizer = obj_prepare.preprocess_text(self.x_train)
        
        inputs = layers.Input(shape=(1,), dtype=tf.string)
        x = text_vectorizer(inputs)
        x = embedding(x)
        x = layers.LSTM(units=512, return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(units=256, return_sequences=True)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(units=64)(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = tf.keras.Model(inputs, outputs, name='LSTM_model')
        
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=['acc']
                      )

        return model
