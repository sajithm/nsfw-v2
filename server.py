from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K
from PIL import Image
import numpy as np
import flask
import io

MODEL_PATH = 'model.hdf5'
WEIGHTS_PATH = 'weights.hdf5'
IMAGE_DEPTH = 3
IMAGE_WIDTH = 192
IMAGE_HEIGHT = 192
IMAGE_SHAPE = (IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH)

if K.image_data_format() == 'channels_last':
    IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)

app = flask.Flask(__name__)
model = load_model(MODEL_PATH)
labels = ['nsfw-nude', 'nsfw-risque', 'nsfw-sex', 'nsfw-violence', 'sfw']

#The following line is required to avoid trouble.
#https://github.com/keras-team/keras/issues/2397#issuecomment-354061212
print('testing model:', model.predict(np.zeros((1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))))

def prepare_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255. 
    return img_tensor

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False, "predictions": {}}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            img = flask.request.files["image"].read()
            img = Image.open(io.BytesIO(img))
            img_tensor = prepare_image(img)
            pred_prob = model.predict(img_tensor)
            for index, prob in enumerate(pred_prob[0]):
                data["predictions"][labels[index]] = float(prob)
            data["success"] = True
            data["is_safe"] = bool(pred_prob[0][4] > 0.5)
    return flask.jsonify(data)

if __name__ == "__main__":
    app.run()