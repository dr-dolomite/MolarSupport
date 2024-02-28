import cv2
import os

def preprocess_input(image_path):
    # Create CLAHE object
    output_folder = "output_images/preprocess_input"
    filename = "initial_pre.jpg"
    
    # Create output folder if not present
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        clahe_img = clahe.apply(img)
        
        # Save the processed image to the output folder in color (RGB)
        output_path = os.path.join(output_folder, filename)
        
        cv2.imwrite(output_path, cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB))
        print(f"Preprocessed image saved to {output_path}")
        
        return output_path
    
    except:
        print(f"Error in preprocess_input")
    
