import base64
import io

from flask import Flask, jsonify, request

from tensorflow.keras.layers import BatchNormalization
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
yolo_model = YOLO("yolov8n.pt")

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

def contains_fish(image_data):
    results = yolo_model(image_data)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            class_name = yolo_model.names[cls_id]
            if class_name == "fish":
                return True
    return False

@app.route('/api/image/predict', methods=['POST'])
def predict_fish():
    file = request.files['image']
    image_data = file.read()
    image = Image.open(io.BytesIO(image_data))
    result = predict(image)
    return jsonify(result)


@app.route('/api/image/check', methods=['POST'])
def check_image():
    if 'image' not in request.json:
        return jsonify({"error": "No image uploaded"}), 400

    image_b64 = request.json['image']
    image_data = base64.b64decode(image_b64)
    image = Image.open(io.BytesIO(image_data))

    if contains_fish(image):
        return jsonify({"status": "approved"})
    else:
        return jsonify({"status": "pending", "reason": "No Fish inside"})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)