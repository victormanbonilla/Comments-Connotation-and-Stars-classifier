import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from os import path
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
import tensorflow as tf  # Tensorflow 2.3
import numpy as np
import pandas as pd


app = FastAPI(title="Modelo NLP Comentarios Victor Manuel Bonilla")


file_path = path.abspath(__file__)
dir_path = path.dirname(file_path)

model_1_dir = 'saved_models/binary/model_binary'
model_1_path = os.path.join(dir_path, model_1_dir)
model1 = tf.keras.models.load_model(model_1_path)

model_2_dir = 'saved_models/stars/model_stars'
model_2_path = os.path.join(dir_path, model_2_dir)
model2 = tf.keras.models.load_model(model_2_path)


class comments(BaseModel):
    comment: str


@app.get('/')
def index():
    return {'message': 'This is Comments Classification API!'}


# @app.post('/predict_connotation')
@app.get("/items/")
async def predict_qwery(start: int=0, limit: int=3):

    comment_path = os.path.join(dir_path, 'comentarios.txt')
    comment = pd.read_csv(comment_path, header=None, delimiter='\n')
    comment = np.array(comment.values)

    with tf.device("CPU:0"):

        prediction = model1.predict(comment.reshape(-1, 1))

    output = prediction[start:limit]
    preds = []

    
    for i in range(len(output)):
                
        if output[i] > 0.5:
            connt = 'Positivo'
        else:
            connt = 'Negativo'
        
        preds.append([comment[i+start, 0], connt])

    return {'Connotaciones': preds}


@app.post('/predict_stars_and_connotation')
def predict_stars_and_connotation(data: comments):
    """ FastAPI
    Args:
        data (Reviews): json file
    Returns:
        prediction: probability of review being positive
    """
    data = data.dict()
    comment = data['comment']
    comment = tf.expand_dims(comment, axis=0)

    with tf.device("CPU:0"):
        prediction_stars = model2.predict(comment)
        prediction_stars = np.argmax(prediction_stars)
        prediction_stars = prediction_stars.tolist()
        print(prediction_stars)

        prediction_conn = model1.predict(comment)
        print(prediction_conn)

        if prediction_conn > 0.5:
            prediction_conn = 'Comentario positivo'
        else:
            prediction_conn = 'Comentario negativo'

    return {
            'Prediccion estrellas': prediction_stars,
            'Connotacion del comentario': prediction_conn
            }


if __name__ == '__main__':
    uvicorn.run(app, port=8000, host='0.0.0.0')
  

