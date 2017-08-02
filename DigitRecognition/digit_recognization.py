# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#print X_train.shape[1]
# flatten 28*28 images to a 784 vector for each image
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
#print(X_train)
# print X_train.shape
# print X_train[0]
# print "hii"
# print y_train
# print y_train.shape
X_train /= 255
X_test /= 255

print y_test[0]
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
#print y_train[0]
#print y_test.shape[0]
num_classes = y_test.shape[1]
#print num_classes


def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = baseline_model()
# Fit the model

model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=3, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
# print model.metrics_names
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))
prediction = model.predict_classes(X_test)


Image = Image.open('images/5.bmp')
reducedImage = Image.resize((28,28))
#plt.imshow(reducedImage ,cmap=plt.get_cmap('gray'))
#Convert Image to numpy array
imageArray= np.array(reducedImage)
numPixels=imageArray.shape[0]*imageArray.shape[1]
imageArray=imageArray.reshape(1,numPixels).astype('float32')

prediction_new =model.predict(imageArray)
predicted_value=np_utils.probas_to_classes(prediction_new)
print "Digit: ",predicted_value
