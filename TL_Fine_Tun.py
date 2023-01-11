import keras
from keras.layers import Dense, Flatten
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.callbacks import TensorBoard
import time

NAME = "CNN-Xception-unfrozen-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

imgWidth = 150
imgHeight = 150
batchSize = 16

trainImagesFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Train/"

train_datagen = ImageDataGenerator(rescale=1 / 255,
                                   shear_range=0.4,
                                   zoom_range=0.4,
                                   rotation_range=0.4,
                                   horizontal_flip=True)

train_data_set = train_datagen.flow_from_directory(trainImagesFolder,
                                                   target_size=(imgHeight, imgWidth),
                                                   batch_size=batchSize,
                                                   class_mode='categorical')

validationImagesFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Validation/"

valid_datagen = ImageDataGenerator(rescale=1 / 255)

valid_data_set = valid_datagen.flow_from_directory(validationImagesFolder,
                                                   target_size=(imgHeight, imgWidth),
                                                   batch_size=batchSize,
                                                   class_mode='categorical')

base_model = tf.keras.applications.Xception(input_shape=[imgHeight, imgWidth] + [3], weights='imagenet', include_top=False)
print(f'The model input shape is: {base_model.input_shape}')

classes = glob('D:/Streamlit projects/Weather Project/weather_dataset/Train/*')
print(classes)

classes_num = len(classes)
print('Number of Classes is : ')
print(classes_num)

# inputs = keras.Input(shape=(imgHeight, imgWidth, 3))
# x = base_model(inputs, training = False)

base_model.trainable = True

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(classes_num, activation='softmax'))

print(model.summary())

opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
model.compile(optimizer=opt,  # Very low learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping

best_model = ModelCheckpoint('D:/Streamlit projects/Weather Project/weather_dataset/Transfer_Learning_Xception_FT.h5',
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)

earlystop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)

result_2 = model.fit(train_data_set, validation_data=valid_data_set, epochs=15, verbose=1,
                   callbacks=[best_model, earlystop, tensorboard])

plt.plot(result_2.history['accuracy'], label='train accuracy')
plt.plot(result_2.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()

plt.plot(result_2.history['loss'], label='train loss')
plt.plot(result_2.history['val_loss'], label='val loss')
plt.legend()
plt.show()

model.save("D:/Streamlit projects/Weather Project/Transfer_Learning_Xception_FT.h5")