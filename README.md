# doodle-classifier

Watch a neural network classify your fruit drawings.
GUI was made using Electron.js paired up with a Python backend running a flask server.

Watch the video here : https://youtu.be/O8Gvkq8y-qs

Google colab notebook : https://colab.research.google.com/drive/1pG7gbXyAq-8UL_Pj18FV6O-jU0MmYZze


<img src="/samples/1.png" width="50%" />

## Requirements
* Python
* Electron.js
* node.js
* Tensorflow
* Keras
* numpy
* scikit-learn
* PIL
* scipy
* Flask


Install electron (You'll need node.js)
```sh
$ npm install electron -g
```

Download and move into the project directory
```
$ git clone https://github.com/SouravJohar/doodle-classifier.git
$ cd doodle-classifier 
```

Download the dataset inside a 'data/' directory.
Dataset : https://github.com/googlecreativelab/quickdraw-dataset (get the numpy bitmaps)

```sh
$ python train.py
```

After training, link your saved model in `server.py` and then run the server

```sh
$ python server.py
```
After the server is up and running,

```sh
$ electron .
```
