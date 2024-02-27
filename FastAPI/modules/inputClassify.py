import numpy as np
import cv2
import os
import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.backend import clear_session

# clear_session()
# inputValidityModel = load_model(os.path.join('modules/model_checkpoint', 'inputClassification.h5'))

def predict_input_validty(image_path, inputValidityModel):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    resize = tf.image.resize(img, (256, 256))
    input_data = np.expand_dims(resize / 255, 0)
    prediction_input_check = inputValidityModel.predict(input_data)
    
    if prediction_input_check > 0.5:
        
        # Delete the image
        os.remove(image_path)
        
        return {"error": "The image is not a valid sliced CBCT input. Please upload a valid image."}
    
    return {"message": "The image is a valid sliced CBCT input."}