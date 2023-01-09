from tensorflow import keras
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
import cv2 as cv
import streamlit as st
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

NAME = "NN-1024-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))

st.title("This is the MNIST trial")
st.write("You do know that there is a glitch in the matrix, don't you?")

st.header("You can select your own ingredients")
image_size = 784
num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model_nn = Sequential()

model_nn.add(Dense(units=1024, activation='relu', input_shape=(image_size,)))  # hidden layer 1
model_nn.add(Dropout(0.3))
model_nn.add(Dense(units=512, activation='relu'))
model_nn.add(Dense(units=256, activation='relu'))
model_nn.add(Dropout(0.2))
model_nn.add(Dense(units=64, activation='relu'))
model_nn.add(Dense(units=32, activation='relu'))
model_nn.add(Dense(units=16, activation='relu'))
model_nn.add(Dense(units=num_classes, activation='softmax'))  # output layer
model_nn.summary()

x_train = x_train.reshape(len(x_train), -1)
print(f'This is the x_train shape: {x_train.shape}')
print(x_train[0].shape)
y_train = y_train.reshape(len(y_train), -1)
#print(y_train[0])
x_test = x_test.reshape(len(x_test), -1)
#print(x_test[0])
y_test = y_test.reshape(len(y_test), -1)
#print(y_test[0])


x_train_norm = x_train / 255
x_test_norm = x_test / 255

print(f'This is the normalized shape: {x_train_norm.shape}')

model_nn.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
history = model_nn.fit(x_train_norm, y_train, batch_size=128, epochs=5, verbose=False, validation_split=.1,
                       callbacks=[tensorboard])
loss, accuracy = model_nn.evaluate(x_test_norm, y_test, verbose=True)

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['training', 'validation'], loc='best')
# plt.show()

options = ["Neural Network", "Convolutional Neural Network"]
st.header("Here is the menu")
arch_selected = st.selectbox("Pick your own poison", options = options)
st.write("Select box returns:", arch_selected)
if arch_selected == "Neural Network":

    image = st.file_uploader("Upload your photo", type=['jpg', 'png'])
    file = "D:/Streamlit projects/Weather Project/img_2.png"
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (28, 28))
    print(f'This is the resized image: {image.shape}')
    image = image.astype('float32')
    print(f'This is the user image shape: {image.shape}')
    image = image.reshape(-1, 784)
    print(f'This is the user image reshaped: {image.shape}')
    # image = 255-image
    image /= 255
    print(f'This is the normalized user image shape: {image.shape}')

    plt.imshow(image.reshape(28, 28),cmap='Greys')
    plt.show()
    st.write("This is the uploaded image")
    streamlit_image = st.image(image.reshape(28, 28), width=300)
    pred = model_nn.predict(image.reshape(-1, 784), batch_size=1, verbose=True)
    print(pred)
    print(np.argmax(pred))
    streamlit_result = st.write(f'This is the predicted result: {np.argmax(pred)}')

else:
    #CNN MODEL
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_train = x_train.astype('float32')/255

    x_test = x_test.reshape(10000, 28, 28, 1)
    x_test = x_test.astype('float32')/255

    model_cnn = Sequential()
    model_cnn.add(Conv2D(32,(3,3), activation='relu', input_shape = (28,28,1)))
    model_cnn.add(MaxPooling2D((2,2)))
    model_cnn.add(Conv2D(64,(3,3), activation='relu'))
    model_cnn.add(MaxPooling2D((2,2)))
    model_cnn.add(Conv2D(64,(3,3), activation='relu'))
    model_cnn.add(Flatten())
    model_cnn.add(Dense(64,activation = 'relu'))
    model_cnn.add(Dense(10, activation= 'softmax'))

    model_cnn.compile(optimizer = 'rmsprop',
                 loss = 'categorical_crossentropy',
                 metrics = ['accuracy'])

    model_cnn.fit(x_train, y_train, epochs=5, batch_size = 128)

    file_2 = "D:/Streamlit projects/Weather Project/img_2.png"
    image_2 = cv.imread(file_2, cv.IMREAD_GRAYSCALE)
    image_2 = cv.resize(image_2, (28, 28))
    print(f'This is the resized image: {image_2.shape}')
    # image_2 = image.astype('float32')
    # print(f'This is the user image shape: {image_2.shape}')
    image_2 = image_2.reshape(-1, 28, 28, 1)
    print(f'This is the user image reshaped: {image_2.shape}')
    # image_2 = 255-image_2
    # image_2 /= 255
    print(f'This is the normalized user image shape: {image_2.shape}')

    plt.imshow(image_2.reshape(28, 28),cmap='Greys')
    plt.show()
    st.write("This is the 2nd uploaded image")
    streamlit_image_2 = st.image(image_2.reshape(28, 28), width=300)
    pred_2 = model_cnn.predict(image_2.reshape(-1, 28, 28, 1), batch_size=1, verbose=True)
    print(pred_2)
    print(np.argmax(pred_2))
    streamlit_result_2 = st.write(f'This is the predicted result: {np.argmax(pred_2)}')