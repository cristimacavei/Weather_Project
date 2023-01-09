import os
import random
import shutil

dataOrgFolder = "D:/Streamlit projects/Weather Project/original_weather_dataset/"
dataBaseFolder = "D:/Streamlit projects/Weather Project/weather_dataset"

dataDirList = os.listdir(dataOrgFolder)
print(dataDirList)

split_Size = 0.85


def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    files = []

    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        print(file)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " has 0 length , will not copy this file !!")

    print(len(files))

    trainLength = int(len(files) * SPLIT_SIZE)
    validLength = int(len(files) - trainLength)

    suffleDataSet = random.sample(files, len(files))

    trainingSet = suffleDataSet[0:trainLength]
    validSet = suffleDataSet[trainLength:]

    for filename in trainingSet:
        f = SOURCE + filename
        dest = TRAINING + filename
        shutil.copy(f, dest)

    for filename in validSet:
        f = SOURCE + filename
        dest = VALIDATION + filename
        shutil.copy(f, dest)


dewSourceFolder = "D:/Streamlit projects/Weather Project/original_weather_dataset/dew/"
dewTrainFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Train/dew/"
dewValidFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Validation/dew/"

fogsmogSourceFolder = "D:/Streamlit projects/Weather Project/original_weather_dataset/fogsmog/"
fogsmogTrainFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Train/fogsmog/"
fogsmogValidFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Validation/fogsmog/"

frostSourceFolder = "D:/Streamlit projects/Weather Project/original_weather_dataset/frost/"
frostTrainFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Train/frost/"
frostValidFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Validation/frost/"

glazeSourceFolder = "D:/Streamlit projects/Weather Project/original_weather_dataset/glaze/"
glazeTrainFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Train/glaze/"
glazeValidFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Validation/glaze/"

hailSourceFolder = "D:/Streamlit projects/Weather Project/original_weather_dataset/hail/"
hailTrainFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Train/hail/"
hailValidFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Validation/hail/"

lightningSourceFolder = "D:/Streamlit projects/Weather Project/original_weather_dataset/lightning/"
lightningTrainFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Train/lightning/"
lightningValidFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Validation/lightning/"

rainSourceFolder = "D:/Streamlit projects/Weather Project/original_weather_dataset/rain/"
rainTrainFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Train/rain/"
rainValidFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Validation/rain/"

rainbowSourceFolder = "D:/Streamlit projects/Weather Project/original_weather_dataset/rainbow/"
rainbowTrainFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Train/rainbow/"
rainbowValidFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Validation/rainbow/"

rimeSourceFolder = "D:/Streamlit projects/Weather Project/original_weather_dataset/rime/"
rimeTrainFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Train/rime/"
rimeValidFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Validation/rime/"

sandstormSourceFolder = "D:/Streamlit projects/Weather Project/original_weather_dataset/sandstorm/"
sandstormTrainFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Train/sandstorm/"
sandstormValidFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Validation/sandstorm/"

snowSourceFolder = "D:/Streamlit projects/Weather Project/original_weather_dataset/snow/"
snowTrainFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Train/snow/"
snowValidFolder = "D:/Streamlit projects/Weather Project/weather_dataset/Validation/snow/"

split_data(dewSourceFolder, dewTrainFolder, dewValidFolder, split_Size)
split_data(fogsmogSourceFolder, fogsmogTrainFolder, fogsmogValidFolder, split_Size)
split_data(frostSourceFolder, frostTrainFolder, frostValidFolder, split_Size)
split_data(glazeSourceFolder, glazeTrainFolder, glazeValidFolder, split_Size)
split_data(hailSourceFolder, hailTrainFolder, hailValidFolder, split_Size)
split_data(lightningSourceFolder, lightningTrainFolder, lightningValidFolder, split_Size)
split_data(rainSourceFolder, rainTrainFolder, rainValidFolder, split_Size)
split_data(rainbowSourceFolder, rainbowTrainFolder, rainbowValidFolder, split_Size)
split_data(rimeSourceFolder, rimeTrainFolder, rimeValidFolder, split_Size)
split_data(sandstormSourceFolder, sandstormTrainFolder, sandstormValidFolder, split_Size)
split_data(snowSourceFolder, snowTrainFolder, snowValidFolder, split_Size)