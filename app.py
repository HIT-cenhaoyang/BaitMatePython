import io

import psycopg2
from flask import Flask, jsonify, abort

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

def get_image_from_db(image_id):
    try:
        conn = psycopg2.connect(
            dbname="postgres",
            host="localhost",
            user="postgres",
            password="123456"
        )
        cur = conn.cursor()
        cur.execute("SELECT fish_image FROM baitmate.fish WHERE id = %s", (image_id,))
        oid = cur.fetchone()
        lobj_manager = conn.lobject(oid=oid[0], mode="rb")
        # Read image data
        image_data = lobj_manager.read()
        lobj_manager.close()
        # Convert binary to image
        image = Image.open(io.BytesIO(image_data))
        cur.close()
        conn.close()
        if image_data is None:
            abort(404, description="Image not found")
        return image
    except psycopg2.OperationalError as e:
        print(f"Database connection error: {e}")
        abort(500, description="Database connection error")


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

    for i in x:
        print("{className}: {predVal:.2f}%".format(className=class_name[i], predVal=float(answer[0][i]) * 100))

    y_class = answer.argmax(axis=-1)
    # y = " ".join(str(x) for x in y_class)
    y = int(y_class[0])
    res = class_name[y]

    return res

@app.route('/image/<string:image_id>', methods=['GET'])
def predict_fish(image_id):
    image_data = get_image_from_db(image_id)
    result = predict(image_data)
    return jsonify(result)

if __name__ == '__main__':
    app.run()