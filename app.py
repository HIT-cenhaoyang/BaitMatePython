import io

from flask import Flask, jsonify, abort, request

from keras.src.layers import BatchNormalization
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

app = Flask(__name__)

def resize_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image)
    if image.shape[-1] != 3:
        image = np.stack((image,) * 3, axis=-1)
    return image

def predict(img):
    model = load_model('./static/FishModelClassifier_V6.h5', compile=False, custom_objects={'BatchNormalization': BatchNormalization})
    class_name = ['Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch', 'Fourfinger Threadfin',
                  'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 'Gourami', 'Grass Carp',
                  'Green Spotted Puffer',
                  'Indian Carp', 'Indo-Pacific Tarpon', 'Jaguar Gapote', 'Janitor Fish', 'Knifefish',
                  'Long-Snouted Pipefish',
                  'Mosquito Fish', 'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp',
                  'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia']
    img = resize_image(img)
    img = img_to_array(img)
    img = img / 255  # rgb is 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    x = list(np.argsort(answer[0])[::-1][:5])
    results = []

    for i in x:
        results.append(( int(i), class_name[i], float(answer[0][i]) * 100))
        print("{className}: {predVal:.2f}%".format(className=class_name[i], predVal=float(answer[0][i]) * 100))

    return results

@app.route('/image/predict', methods=['POST'])
def predict_fish():
    file = request.files['image']
    image_data = file.read()
    image = Image.open(io.BytesIO(image_data))
    result = predict(image)
    return jsonify(result)

if __name__ == '__main__':
    app.run()