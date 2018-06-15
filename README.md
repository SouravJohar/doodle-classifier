# doodle-classifier

Watch a neural network classify your fruit drawings

Watch the video here : https://youtu.be/O8Gvkq8y-qs

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

Download the dataset inside a 'data/' directory.
Dataset : https://github.com/googlecreativelab/quickdraw-dataset (get the numpy bitmaps)

```sh
$ python train.py
```

After training, the link your saved model in `server.py` and then run the server

```sh
$ python server.py
```
After the server is up and running,

```sh
$ npm start
```
