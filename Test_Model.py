import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import streamlit as st

st.title("This is the Weather data trial")
st.write("You do know that there is a glitch in the sky, don't you?")

categories = ["dew", "fogsmog", "frost", "glaze", "hail", "lightning",
              "rain", "rainbow", "rime", "sandstorm", "snow"]

model = tf.keras.models.load_model("D:/Streamlit projects/Weather Project/Transfer_Learning_Vgg19.h5")


def prepareImage(pathForImage):
    image = load_img(pathForImage, target_size=(224, 224))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult / 255.
    return imgResult


image = st.file_uploader("Upload your photo", type=['jpg', 'png'])
print(image)
testImagePath = "D:/Streamlit projects/Weather Project/Test/6101.jpg"

imgForModel = prepareImage(testImagePath)

resultArray = model.predict(imgForModel, verbose=1)

streamlit_image = st.image(imgForModel, width=600)

answer = np.argmax(resultArray, axis=1)
print(answer)

index = answer[0]

print("This image is : " + categories[index])

streamlit_result_2 = st.write(f'This image is: {categories[index]}')