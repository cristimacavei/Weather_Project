import keras
from keras.layers import Dense, Flatten
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

import matplotlib.pyplot as plt
from glob import glob
from tensorflow.keras.callbacks import TensorBoard
import time

# NAME = "CNN-Xception-frozen-{}".format(int(time.time()))
# tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# except:
# # Invalid device or cannot modify virtual devices once initialized.
# pass

imgWidth = 160
imgHeight = 160
batchSize = 8

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

base_model = tf.keras.applications.MobileNetV2(input_shape=[imgHeight, imgWidth] + [3], weights='imagenet', include_top=False)
print(f'The model input shape is: {base_model.input_shape}')

base_model.trainable = False

for layer in base_model.layers:
    layer.trainable = False

classes = glob('D:/Streamlit projects/Weather Project/weather_dataset/Train/*')
print(classes)

classes_num = len(classes)
print('Number of Classes is : ')
print(classes_num)

# inputs = keras.Input(shape=(imgHeight, imgWidth, 3))
# x = base_model(inputs, training = False)

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(classes_num, activation='softmax'))

print(model.summary())

model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, LambdaCallback, ReduceLROnPlateau

best_model = ModelCheckpoint('D:/Streamlit projects/Weather Project/weather_dataset/Transfer_Learning_MobileNetV2.h5',
                             monitor='val_accuracy',
                             verbose=1,
                             save_best_only=True)

earlystop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)


result = model.fit(train_data_set, validation_data=valid_data_set, epochs=15, verbose=1,
                   callbacks=[best_model, earlystop])

plt.plot(result.history['accuracy'], label='train accuracy')
plt.plot(result.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()

plt.plot(result.history['loss'], label='train loss')
plt.plot(result.history['val_loss'], label='val loss')
plt.legend()
plt.show()

NAME = f"CNN-MobileNetV2-withLR-unfrozen-SGD1e_3-batchsize-{batchSize}-{format(int(time.time()))}"
tensorboard = TensorBoard(log_dir = 'logs/{}'.format(NAME))

base_model.trainable = True


def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr*0.9


lr_scheduler = LearningRateScheduler(scheduler, verbose=1)


lambda_callback = LambdaCallback(on_epoch_begin=lambda epoch, logs: print('in lambda:', epoch))

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                              patience=3, min_lr=0.001)

opt = tf.keras.optimizers.SGD(learning_rate=1e-3)
model.compile(optimizer=opt,  # Very low learning rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])

result_2 = model.fit(train_data_set, validation_data=valid_data_set, epochs=15, verbose=1,
                   callbacks=[best_model, earlystop, tensorboard, lr_scheduler, lambda_callback, reduce_lr])

plt.plot(result_2.history['accuracy'], label='train accuracy')
plt.plot(result_2.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()

plt.plot(result_2.history['loss'], label='train loss')
plt.plot(result_2.history['val_loss'], label='val loss')
plt.legend()
plt.show()

model.save("D:/Streamlit projects/Weather Project/Transfer_Learning_MobileNetV2.h5")