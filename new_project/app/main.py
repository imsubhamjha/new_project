from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# Load the trained model (best performing one)
model = load_model("models/unet_plus_plus.h5")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

    # Preprocess the image
    img = img / 255.0
    img = img.reshape((1, img.shape[0], img.shape[1], 1))

    # Predict using the model
    prediction = model.predict(img)
    return {"prediction": prediction.tolist()}