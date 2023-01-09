from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

imgWidth = 256
imgHeight = 256
batchSize = 16

TRAIN_DIR = "D:/Streamlit projects/Weather Project/weather_dataset/Train/"

train_datagetn = ImageDataGenerator(rescale=1 / 255.0,
                                    rotation_range=30,
                                    zoom_range=0.4,
                                    horizontal_flip=True)
train_data_set = train_datagetn.flow_from_directory(TRAIN_DIR,
                                                    batch_size=batchSize,
                                                    class_mode='categorical',
                                                    target_size=(imgHeight, imgWidth))

VALIDATION_DIR = "D:/Streamlit projects/Weather Project/weather_dataset/Validation/"
val_datagen = ImageDataGenerator(rescale=1 / 255.0)
val_data_set = val_datagen.flow_from_directory(VALIDATION_DIR,
                                               batch_size=batchSize,
                                               class_mode='categorical',
                                               target_size=(imgHeight, imgWidth))

callback = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
best_model_file_path = "D:/Streamlit projects/Weather Project/bestWeatherModel.h5"

best_model = ModelCheckpoint(best_model_file_path, monitor="val_accuracy", verbose=1, save_best_only=True)

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(imgHeight, imgWidth, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(11, activation='softmax')
])

print(model.summary())

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_data_set, epochs=30, verbose=1, validation_data=val_data_set, callbacks=[best_model])

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

fig = plt.figure(figsize=(14, 7))
plt.plot(epochs, acc, 'r', label='Train accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and validation accuracy')
plt.legend(loc='lower right')
plt.show()

fig = plt.figure(figsize=(14, 7))
plt.plot(epochs, loss, 'r', label='Train loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and validation loss')
plt.legend(loc='upper right')
plt.show()

model.save("D:/Streamlit projects/Weather Project/bestWeatherModel2.h5")