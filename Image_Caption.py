import os
import numpy as np
from keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import streamlit as st
import tensorflow as tf
import spacy
import csv

# CSV file writer and header
header = ['Picture Number', 'Image Captioning', 'NER (word, label)', 'Classification']
f = open('image_interpretor.csv', 'w', newline='')
writer = csv.writer(f)
writer.writerow(header)

# Image caption model
captioning_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Chose running model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
captioning_model.to(device)

max_length = 16
num_beams = 4
# arguments for image captioning
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

# cathegories for cnn clasification
categories = ["dew", "fogsmog", "frost", "glaze", "hail", "lightning",
              "rain", "rainbow", "rime", "sandstorm", "snow"]

captioning = []

# nlp model
nlp1 = spacy.load(r"./output/model-best")  # load the best model

# cnn classification model
classification_model = tf.keras.models.load_model("D:/Streamlit projects/Weather Project/Transfer_Learning_Vgg19.h5")

st.title("Image Captioning using ViT + GPT2 or smth like this")
image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'], accept_multiple_files=True)

paths = []


# image captioning prediction
def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = captioning_model.generate(pixel_values, **gen_kwargs)
    pred_list = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    pred = ''.join(pred_list)
    return pred


def prepare_image(path_for_image):
    image = load_img(path_for_image, target_size=(224, 224))
    img_result = img_to_array(image)
    img_result = np.expand_dims(img_result, axis=0)
    img_result = img_result / 255.
    return img_result


for image in image_file:
    if image is not None:
        file_details = {"FileName": image.name, "FileType": image.type}
        st.write(file_details)
        img = load_img(image, target_size=(224, 224))
        st.image(img, width=600)
        with open(os.path.join("tempDir", image.name), "wb") as f:
            f.write(image.getbuffer())
        st.success("Saved File")
        prediction = predict_step([f'D:/Streamlit projects/Weather Project/tempDir/{image.name}'])
        st.write(f"This image describes: " + prediction)

        # NAME ENTITY RECOGNITION
        doc = nlp1(prediction)
        list = [(ent.text, ent.label_) for ent in doc.ents]
        str = "NLP sees in this description: "
        if list:
            for tuple in list:
                str = str + "(word :" + tuple[0] + " " + "label :" + tuple[1] + ") "
            st.write(str)
        else:
            list = "I can't tell what is in there"
            st.write("NLP sees in this description: I can't tell what is in there")

        # CLASSIFICATION
        imgForModel = prepare_image(f'D:/Streamlit projects/Weather Project/tempDir/{image.name}')
        resultArray = classification_model.predict(imgForModel, verbose=1)
        answer = np.argmax(resultArray, axis=1)
        index = answer[0]
        print("This image is : " + categories[index])
        st.write(f'This image fits into the category: {categories[index]}')

        # WRITE TO CSV
        data = [image.name, prediction, list, categories[index]]
        writer.writerow(data)

f.close()
