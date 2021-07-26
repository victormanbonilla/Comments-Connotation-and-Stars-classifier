# Comments-Connotation-and-Stars-classifier

This is the project for the test of EasyData.

## Procedimiento en localhost

Puede empaquetar la app con el docker usando "docker build -t comments-app ."

Y ejecutarla con "sudo docker run -p 8000:8000 comments-app"

O Puede bajar la imagen desde docker hub "sudo docker pull victormanbonilla/bonilla-comments-app"

Luego la ejecura con "sudo docker run -p 8000:8000 victormanbonilla/bonilla-comments-app"

Abra el navegador y escriba localhost:8000/docs.

Hay un método de obtención y un método de publicación.

### Método de obtención: Lee los datos históricos de los comentarios y calcula su connotación.

tiene dos quiery parammeters, el primero: Start; sirve para definir el inicio de donde se tomara el primer comentario que esta en el archivo "comentarios.txt" y el segundo: Limit, define hasta que comentario se toma.

El resultado de este metodo es un json con el comentario, seguido de su connotacion: Positiva o negativa.

### Método de publicación: A partir de un comentario dado, calcula la connotación y predice la calificación de estrellas.

En este metodo se inserta manuelmente un comentario en formato string y arroja una respuesta de la connotacion del mismo, ademas de un pronostico de la cantidad de estrellas que recibira el comentario.

Por ejemplo, si coloca: "Perfecta, la aplicacion funciona bien", recibira una connotacion positiva y 5 estrellas.

Si coloca: "Regular, pero puede mejorar", recibira una connotacion positiva y 3 estrellas.

Si coloca: "Malisima app, no me gusto para nada", recibira una connotacion negativa y 1 estrella.

Puede intentar con el comentario que quiera, pero se recomienda que sea superior a 4 palabras, debido a que el modelo se entreno con esa longitid de palabras.

## Procedure in localhost

Download image from docker hub "sudo docker pull victormanbonilla/bonilla-comments-app"

Run app with "sudo docker run -p 8000:8000 victormanbonilla/bonilla-comments-app"

open navigator and type localhost:8000/docs.

There is one get method and one post method.

### Get method: Reads the historical data of coments and calculates it's connotation.

### Post method: from a given comment, it calculates the connotation and predict the star qualification.




