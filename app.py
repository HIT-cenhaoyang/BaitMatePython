import base64
import io
import os

from flask import Flask, jsonify, request
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
from ultralytics import YOLO
import gdown
from gevent import pywsgi
import tensorflow as tf

app = Flask(__name__)
yolo_model = YOLO("yolov8n.pt")
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

MODEL_PATH = './static/FishModelClassifier_V6.tflite'
GOOGLE_DRIVE_URL = 'https://drive.google.com/uc?id=10GeQtSWnextXzLuIAfc9c81cykvLkEVj'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Downloading from Google Drive...")
        os.makedirs('./static', exist_ok=True)  # 确保 static 目录存在
        gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)
        print("Model file downloaded successfully.")
    else:
        print("Model file already exists.")

def load_model_with_fallback():
    try:
        # 加载 TFLite 模型
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        return None

def resize_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image)
    if image.shape[-1] != 3:
        image = np.stack((image,) * 3, axis=-1)
    return image

def predict(img):
    if not os.path.exists(MODEL_PATH):
        download_model()

    interpreter = load_model_with_fallback()
    if interpreter is None:
        return []

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = resize_image(img)
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]["index"], img)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])

    class_name = ['Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch', 'Fourfinger Threadfin',
                  'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 'Gourami', 'Grass Carp',
                  'Green Spotted Puffer',
                  'Indian Carp', 'Indo-Pacific Tarpon', 'Jaguar Gapote', 'Janitor Fish', 'Knifefish',
                  'Long-Snouted Pipefish',
                  'Mosquito Fish', 'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Barb', 'Silver Carp',
                  'Silver Perch', 'Snakehead', 'Tenpounder', 'Tilapia']
    x = list(np.argsort(output_data[0])[::-1][:5])
    results = []

    for i in x:
        results.append((int(i), class_name[i], float(output_data[0][i]) * 100))
        print("{className}: {predVal:.2f}%".format(className=class_name[i], predVal=float(output_data[0][i]) * 100))

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
    download_model()
    server = pywsgi.WSGIServer(('0.0.0.0', 5000), app)
    server.serve_forever()