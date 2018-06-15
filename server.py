from flask import Flask, render_template, request
import numpy as np
import re
import base64
from PIL import Image
from scipy.misc import imsave, imread, imresize
from keras.models import load_model
from prepare_data import normalize
import json

app = Flask(__name__)

mlp = load_model("./models/mlp_94.h5")
conv = load_model("./models/conv_95.5.h5")
FRUITS = {0: "Apple", 1: "Banana", 2: "Grape", 3: "Pineapple"}


@app.route("/", methods=["GET", "POST"])
def ready():
    if request.method == "GET":
        return render_template("index1.html")
    if request.method == "POST":
        data = request.form["payload"].split(",")[1]
        net = request.form["net"]

        img = base64.decodestring(data)
        with open('temp.png', 'wb') as output:
            output.write(img)
        x = imread('temp.png', mode='L')
        # resize input image to 28x28
        x = imresize(x, (28, 28))

        if net == "MLP":
            model = mlp
            # invert the colors
            x = np.invert(x)
            # flatten the matrix
            x = x.flatten()

            # brighten the image a bit (by 60%)
            for i in range(len(x)):
                if x[i] > 50:
                    x[i] = min(255, x[i] + x[i] * 0.60)

        if net == "ConvNet":
            model = conv
            x = np.expand_dims(x, axis=0)
            x = np.reshape(x, (28, 28, 1))
            # invert the colors
            x = np.invert(x)
            # brighten the image by 60%
            for i in range(len(x)):
                for j in range(len(x)):
                    if x[i][j] > 50:
                        x[i][j] = min(255, x[i][j] + x[i][j] * 0.60)

        # normalize the values between -1 and 1
        x = normalize(x)
        val = model.predict(np.array([x]))
        pred = FRUITS[np.argmax(val)]
        classes = ["Apple", "Banana", "Grape", "Pineapple"]
        print pred
        print list(val[0])
        return render_template("index1.html", preds=list(val[0]), classes=json.dumps(classes), chart=True, putback=request.form["payload"], net=net)


app.run()
