from flask import Flask, render_template, request, jsonify
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "8"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)
IMG_HEIGHT = 180
IMG_WIDTH = 180

model = load_model("model_epoch_10.keras")

# Если модель включает информацию о классах, их можно автоматически получить
class_names = ["apple", "apricot", "avocado", "banana", "blackberry", "buffaloberry",
               "caimito", "cherry", "coconut", "crowberry", "dragonfruit", "durian",
               "grape", "kiwi", "lemon", "lime", "mango", "orange", "papaya", 
               "pear", "persimmon", "pineapple", "pomegranate", "strawberry", "watermelon"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Чтение изображения
        img_file = request.files["image"].read()
        img = Image.open(io.BytesIO(img_file))

        # Изменение размера изображения
        img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.NEAREST)

        # Преобразование изображения в массив
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        return jsonify({
            "predictions": "Это изображение {} с вероятностью {:.2F} процентов.".format(class_names[np.argmax(score)], 100 * np.max(score))
        })

if __name__ == "__main__": 
    app.run(debug=True)
