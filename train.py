from prepare_data import *
from sklearn.model_selection import train_test_split as tts
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from nets.MLP import mlp
from nets.conv import conv
from random import randint

# define some constants
N_FRUITS = 4
FRUITS = {0: "Apple", 1: "Banana", 2: "Grape", 3: "Pineapple"}

# number of samples to take in each class
N = 1000

# some other constants
N_EPOCHS = 20

# data files in the same order as defined in FRUITS
files = ["apple.npy", "banana.npy", "grapes.npy", "pineapple.npy"]

# images need to be 28x28 for training with a ConvNet
fruits = load("data/", files, reshaped=True)

# images need to be flattened for training with an MLP
# fruits = load("data/", files, reshaped=False)


# limit no of samples in each class to N
fruits = set_limit(fruits, N)

# normalize the values
fruits = map(normalize, fruits)

# define the labels
labels = make_labels(N_FRUITS, N)

# prepare the data
x_train, x_test, y_train, y_test = tts(fruits, labels, test_size=0.05)

# one hot encoding
Y_train = np_utils.to_categorical(y_train, N_FRUITS)
Y_test = np_utils.to_categorical(y_test, N_FRUITS)

# use our custom designed ConvNet model
model = conv(classes=N_FRUITS, input_shape=(28, 28, 1))

# use our custom designed MLP model
# model = mlp(classes=N_FRUITS)


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

raw_input("Type 'train' to start training: ")
print "Training commenced"

model.fit(np.array(x_train), np.array(Y_train), batch_size=32, epochs=N_EPOCHS, verbose=1)

print "Training complete"

print "Evaluating model"
preds = model.predict(np.array(x_test))

score = 0
for i in range(len(preds)):
    if np.argmax(preds[i]) == y_test[i]:
        score += 1

print "Accuracy: ", ((score + 0.0) / len(preds)) * 100

name = raw_input(">Enter name to save trained model: ")
model.save(name + ".h5")
print "Model saved"


def visualize_and_predict():
    "selects a random test case and shows the object, the prediction and the expected result"
    n = randint(0, len(x_test))
    visualize(denormalize(np.reshape(x_test[n], (28, 28))))
    pred = FRUITS[np.argmax(model.predict(np.array([x_test[n]])))]
    actual = FRUITS[y_test[n]]
    print "Actual:", actual
    print "Predicted:", pred


print "Testing mode"
visualize_and_predict()
