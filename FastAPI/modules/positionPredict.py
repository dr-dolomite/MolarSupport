import os
import cv2
import tensorflow as tf
import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.backend import clear_session

# clear_session()
# position_model = load_model(os.path.join('modules/model_checkpoint', 'vgg16_checkpoint.h5'))

def predict_position(position_model):
    try:
        # Read the image from the specified path
        image_path = os.path.join("output_images/enhanced_output/enhanced_final.jpg")
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        resize = tf.image.resize(img, (224,224))
        yhat = position_model.predict(np.expand_dims(resize/255,0))
        
        # Find the index of the maximum value
        predicted_label_index = np.argmax(yhat)
        
        # Define labels based on your class order
        labels = ['apical', 'buccal', 'lingual', 'none']
        
        # Print the predicted label
        predicted_label = labels[predicted_label_index]
        print(predicted_label)
        
        if predicted_label == "apical":
            position_label = "Apical"
            
        elif predicted_label == "buccal":
            position_label = "Buccal"
            
        elif predicted_label == "lingual":
            position_label = "Lingual"
            
        else:
            position_label = "None"
        
        print("This is the position: " + position_label)
        return position_label

    except:
        print(f"Some error occurred while predicting the position of the tooth.")
