import cv2
import os
import numpy as np
from PIL import Image

def enhance_colors(image_path, saturation_factor=2.5):
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image from BGR to HSV
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Increase the saturation
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * saturation_factor, 0, 255)

    # Convert the image back to BGR
    enhanced_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    
    # Resize image shape
    #resized_img = cv2.resize(enchanced_img, "355, 355")
    
    # Save the enhanced image
    enhanced_img_path = os.path.join("output_images/enhanced_output/enhanced.jpg")
    
    # Create the folder if it does not exist
    if not os.path.exists("output_images/enhanced_output/"):
        os.makedirs("output_images/enhanced_output/")
    
    cv2.imwrite(enhanced_img_path, enhanced_img)
    
    return enhanced_img_path


def flat_violet_to_color(image_path, target_color, threshold=30):
    # Load the image
    
    
    img = Image.open(image_path)

    # Convert the image to a NumPy array
    img_array = np.array(img)

    # Define the violet color
    violet_color = np.array([64, 0, 58])

    # Define the threshold for identifying violet areas
    mask = np.sum(np.abs(img_array - violet_color), axis=-1) <= threshold

    # Apply the target color to the violet areas
    img_array[mask] = target_color

    # Save the result
    result_image = Image.fromarray(img_array)
    
    final_enhanced_img_path = os.path.join("output_images/enhanced_output/enhanced_final.jpg")
    
    # Create the folder if it does not exist
    if not os.path.exists("output_images/enhanced_output/"):
        os.makedirs("output_images/enhanced_output/")
        
    result_image.save(final_enhanced_img_path)