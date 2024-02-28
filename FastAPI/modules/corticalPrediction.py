import cv2
import numpy as np
import os
import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.backend import clear_session

# clear_session()
# cortical_model = load_model(os.path.join('modules/model_checkpoint', 'cortiClassification.h5'))

def predict_cortical(cortical_model):
    image_path = os.path.join("output_images/enhanced_output/enhanced_final.jpg")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    resize = tf.image.resize(img, (256, 256))
    input_data = np.expand_dims(resize / 255, 0)
    prediction_corticilization_type = cortical_model.predict(input_data)

    if prediction_corticilization_type > 0.5:
        interruption_prediction = "Positive"
    else:
        interruption_prediction = "Negative"

    return interruption_prediction