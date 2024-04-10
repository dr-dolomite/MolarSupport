# import torch
# import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
# from model import UNET
# from utils import load_checkpoint

import cv2
import os
# import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Header, Depends
from fastapi.responses import FileResponse, JSONResponse

# from fastapi_controllers import create_controller, controller, register_controllers_to_app

from starlette.middleware.cors import CORSMiddleware

# import tensorflow as tf
# from tensorflow.keras.models import load_model

from pydantic import BaseModel

# # Avoid Out Of Memory (OOM) errors by setting GPU Memory Consumption Growth
# # gpus = tf.config.experimental.list_physical_devices('GPU')
# # print(gpus)

# # for gpu in gpus:
# #     tf.config.experimental.set_memory_growth(gpu,True)

# class DistanceInput(BaseModel):
#     distance: float

# # global variable for distance
# distance_value = 0.0

# csv_file_path = 'results.xlsx'

# --------------- SQLITE AREA ---------------

# REFERENCE: https://blog.stackademic.com/how-to-build-a-crud-api-using-fastapi-python-sqlite-for-new-coders-2d056333ea20

#------------------ ROD CHANGES ------------------#

# --- Create a Database with SQLite ---
import sqlite3
from datetime import datetime # concatenate with filenames

# Class for the Molar Case
class MolarCaseCreate(BaseModel):
    mc_filename: str
    m3_filename: str
    generated_m3_mask_filename: str
    final_img_filename: str
    corticalization: str
    position: str
    distance: str
    final_image_distance: str
    relation: str
    risk: str

    # mc_file_upload: str
    # m3_file_upload: str
    # original_m3_mask: str
    # m3_mask_prediction: str
    # layered_image: str
    # corticalization: str
    # position: str
    # distance: float
    # image_with_distance: str
    # relation: str
    # risk: str

    

class MolarCase(MolarCaseCreate):
    id: int

def create_connection(): # establish connection
    connection = sqlite3.connect("molarcases.db")
    return connection

def create_table(): # create table for molar cases
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("""
                CREATE TABLE IF NOT EXISTS molarcases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                        mc_filename TEXT NOT NULL,
                        m3_filename TEXT NOT NULL,
                        generated_m3_mask_filename TEXT,
                        final_img_filename TEXT NOT NULL,
                        corticalization TEXT NOT NULL,
                        position TEXT NOT NULL,
                        distance FLOAT NOT NULL,
                        final_image_distance TEXT NOT NULL,
                        relation TEXT NOT NULL,
                        risk TEXT NOT NULL
                );
                    """)
    connection.commit()
    connection.close()

create_table()

def create_case(case: MolarCaseCreate): # (CRUD) Create molar case
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("INSERT INTO molarcases (mc_filename, m3_filename, generated_m3_mask_filename, final_img_filename, corticalization, position, distance, final_image_distance, relation, risk) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                   (case.mc_filename, case.m3_filename, case.generated_m3_mask_filename, case.final_img_filename, case.corticalization, case.position, case.distance, case.final_image_distance, case.relation, case.risk))
    connection.commit()
    connection.close()

# Function to get all molar cases from the SQLite database
def get_all_cases():
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM molarcases")
    rows = cursor.fetchall()
    connection.close()
    return rows

# Function to delete the molarcases table
def delete_table():
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS molarcases")
    connection.commit()
    connection.close()

# --------------- SQLITE AREA ---------------
    
#------------------ ROD CHANGES ------------------#

# # Load the model for input classification
# # model_input_check = load_model(os.path.join('model_checkpoint', 'inputClassification.h5'))
# # model_corticilization_type = load_model(os.path.join('model_checkpoint', 'cortiClassification.h5'))
# # model_position = load_model(os.path.join('model_checkpoint', 'vgg16_checkpoint.h5'))

# ------------------ Start of Segmentation Process Functions ------------------#
# def preprocess_input(image_path):
#     # Create CLAHE object
#     output_folder = "preprocess_input"
#     filename = "temp_input_image.jpg"
#     try:
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#         img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         clahe_img = clahe.apply(img)
        
#         # Save the processed image to the output folder in color (RGB)
#         output_path = os.path.join(output_folder, filename)
#         cv2.imwrite(output_path, cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB))
#         return output_path
#     except Exception as e:
#         print(f"Error in preprocess_input: {e}")
#         raise HTTPException(status_code=500, detail="Error preprocessing input image")


# def load_model_and_predict(input_image_path):
#     # Load the pre-trained model checkpoint
#     checkpoint_path = "model_checkpoint/my_checkpoint.pth.tar"
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = UNET(in_channels=3, out_channels=1)
#     load_checkpoint(torch.load(checkpoint_path), model)
#     model = model.to(device).eval()

#     # Load and preprocess the input image
#     image_path = preprocess_input(input_image_path)
    
#     image = Image.open(image_path).convert("RGB")
#     image = TF.resize(image, (400, 400), antialias=True)
#     image_tensor = TF.to_tensor(image).unsqueeze(0).to(device)

#     # Perform segmentation on the input image
#     with torch.no_grad():
#         prediction = model(image_tensor)
#         mask = torch.sigmoid(prediction) > 0.5
#         mask = mask.squeeze(0).cpu().numpy().astype(np.uint8)

#     # Create the colored segmentation mask
#     color_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)  # Modified line
#     color_mask[mask[0] == 1] = (57, 150, 81)  # Green color (RGB values)  # Modified line

#     # Convert the colored mask to PIL Image and save it
#     color_mask_image = Image.fromarray(color_mask)
    
#     # Save the segmented image
#     segmentation_output_path = os.path.join("output_assets/predicted_segmentation/", f"{os.path.splitext(os.path.basename(input_image_path))[0]}.jpg")
#     color_mask_image.save(segmentation_output_path)
#     return segmentation_output_path


# def remove_small_spots(segmented_image, min_size):
#     # Find connected components
#     _, labels, stats, _ = cv2.connectedComponentsWithStats(segmented_image, connectivity=4)

#     # Filter out small spots
#     for i, stat in enumerate(stats):
#         if stat[4] < min_size:  # Check the area (stat[4]) of the connected component
#             segmented_image[labels == i] = 0

#     return segmented_image

# def contour_filter(segmented_image, min_size_threshold):
#     # Convert to grayscale
#     gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

#     # Apply morphological operations
#     kernel = np.ones((5, 5), np.uint8)
#     morph_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
#     morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)

#     # Find contours on the morphological image
#     contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # Filter out small contours
#     for contour in contours:
#         area = cv2.contourArea(contour)
#         if area < min_size_threshold:
#             cv2.drawContours(segmented_image, [contour], 0, (0, 0, 0), -1)  # Fill contour with black (0, 0, 0)

#     return segmented_image


# def smoothen_edges(cleaned_image):
#     # Applying the filter 
#     bilateral = cv2.bilateralFilter(cleaned_image, 9, 75, 75)
    
#     return bilateral


# def process_image(original_segmented_image, min_size_threshold=5000):
#     # Load your segmented image in color mode
#     segmented_image = cv2.imread(original_segmented_image)

#     # Split the image into individual color channels
#     b, g, r = cv2.split(segmented_image)

#     # Remove small spots for each channel
#     cleaned_b = remove_small_spots(b, min_size_threshold)
#     cleaned_g = remove_small_spots(g, min_size_threshold)
#     cleaned_r = remove_small_spots(r, min_size_threshold)

#     # Smoothen image edges for each channel
#     smoothen_b = smoothen_edges(cleaned_b)
#     smoothen_g = smoothen_edges(cleaned_g)
#     smoothen_r = smoothen_edges(cleaned_r)

#     # Merge the channels back into a single image
#     smoothen_image = cv2.merge([smoothen_b, smoothen_g, smoothen_r])
    
#     # Apply contour filtering
#     filtered_image = contour_filter(smoothen_image, min_size_threshold)

#     # Add gaussian blur
#     #blur_image = cv2.GaussianBlur(filtered_image, (15, 15), 0)
    
#     # Save the processed image
#     preprocess_output_path = os.path.join("output_assets/preprocessed_segmentation/", f"{os.path.splitext(os.path.basename(original_segmented_image))[0]}.jpg")
#     # Save the processed image
#     cv2.imwrite(preprocess_output_path, filtered_image)
#     return preprocess_output_path


# def overlay_images(blur_image, original_image, alpha=0.5):
#     # Read the original image
#     original = cv2.imread(original_image)

#     # Check if the original image is loaded successfully
#     if original is None:
#         print(f"Error: Unable to load original image.")
#         return

#     # Try to read the segmented image
#     try:
#         segmented = cv2.imread(blur_image)

#         # Check if the segmented image is loaded successfully
#         if segmented is None:
#             print(f"Error: Unable to load segmented image.")
#             return

#         # Resize the segmented image to match the original image dimensions
#         segmented = cv2.resize(segmented, (original.shape[1], original.shape[0]))

#         # Blend the images using alpha blending
#         blended = cv2.addWeighted(original, 1 - alpha, segmented, alpha, 0)

#     except cv2.error as e:
#         print(f"OpenCV Error: {e}")
        
#     # Save the result
#     overlay_output_path = os.path.join("output_assets/overlayed_output/", f"{os.path.splitext(os.path.basename(original_image))[0]}.jpg")
#     cv2.imwrite(overlay_output_path, blended)
    
#     return overlay_output_path


# def overlay_result_mc(overlay_output, alpha=0.4):
    
#     with_mc = cv2.imread("uploaded_image/temp_input_image_mc.jpg")
#     overlayed = cv2.imread(overlay_output)
    
#     with_mc = cv2.resize(with_mc, (overlayed.shape[1], overlayed.shape[0]))
    
#     # blend the images
#     blended_final = cv2.addWeighted(overlayed, 1 - alpha, with_mc, alpha, 0)
    
#     # Save the result
#     final_overlayed_path = os.path.join("output_assets/overlayed_output/", f"{os.path.splitext(os.path.basename(overlay_output))[0]}final.jpg")
#     cv2.imwrite(final_overlayed_path, blended_final)
#     return final_overlayed_path

# def flat_violet_to_color(image_path, target_color, threshold=30):
#     # Load the image
    
    
#     img = Image.open(image_path)

#     # Convert the image to a NumPy array
#     img_array = np.array(img)

#     # Define the violet color
#     violet_color = np.array([64, 0, 58])

#     # Define the threshold for identifying violet areas
#     mask = np.sum(np.abs(img_array - violet_color), axis=-1) <= threshold

#     # Apply the target color to the violet areas
#     img_array[mask] = target_color

#     # Save the result
#     result_image = Image.fromarray(img_array)
#     result_image.save('output_assets/enhanced_output/enhanced.jpg')

# def enhance_colors(image_path, saturation_factor=2.5):
#     # Read the image
#     img = cv2.imread(image_path)

#     # Convert the image from BGR to HSV
#     hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     # Increase the saturation
#     hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * saturation_factor, 0, 255)

#     # Convert the image back to BGR
#     enhanced_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    
#     # Resize image shape
#     #resized_img = cv2.resize(enchanced_img, "355, 355")
    
#     return enhanced_img

# #------------------ End of Segmentation Process Functions ------------------#


# #- ----------------- Start of Classification Functions ------------------#

# def classify_relation():
    
#     # load excel file in dataframe
#     df = pd.read_excel("results.xlsx")
    
#     # get the values from the dataframe
#     distance = df.at[1, "Distance"].astype(float)
#     position = df.at[1, "Position"]
#     interruption = df.at[1, "Interruption"]
    
#     # If else statements for classification
    
#     if distance == 0.0 and interruption == "Negative" and position == "None":
#         relation = "Class 0"
        
#     elif position == "Buccal" or position == "Apical" and interruption == "Negative":
#         if distance > 2.0:
#             relation = "Class 1A"
#         else:
#             relation = "Class 1B"
    
#     elif position == "Lingual" and interruption == "Negative":
#         if distance > 2.0:
#             relation = "Class 2A"
#         else:
#             relation = "Class 2B"
    
#     elif position == "Buccal" or position == "Apical" and interruption == "Positive":
#        if distance > 1.0:
#            relation = "Class 3A"
#        else:
#            relation = "Class 3B"
           
#     elif position == "Lingual" and interruption == "Positive":
#         if distance > 1.0:
#             relation = "Class 4A"
#         else:
#             relation = "Class 4B"
    
#     else:
#         relation = "Unclassified Relation"
    
#     print(f"Relation: {relation}")
    
#     # Append the new value to the "Relation" column
#     df.at[1, "Relation"] = relation
    
#     # Save the updated DataFrame back to the Excel file
#     df.to_excel("results.xlsx", index=False)
    
#     return relation

# def classify_risk():
    
#     # load excel file in dataframe
#     df = pd.read_excel("results.xlsx")
    
#     # get the values from the dataframe
#     relation = df.at[1, "Relation"]
    
#     # If else statements for classification
#     if relation == "Class 0":
#         risk = "N.0 (Non-determinant)"
    
#     elif relation == "Class 1A" or relation == "Class 1B" or relation == "Class 2A" or relation == "Class 2B" or relation == "Class 4A":
#         risk = "N.1 (Low)"
    
#     elif relation == "Class 3A" or relation == "Class 3B":
#         risk = "N.2 (Medium)"
    
#     elif relation == "Class 4B":
#         risk = "N.3 (High)"
    
#     else:
#         risk = "Unclassified Risk"
    
#     print(f"Risk: {risk}")
    
#     # Append the new value to the "Risk" column
#     df.at[1, "Risk"] = risk
    
#     # Save the updated DataFrame back to the Excel file
#     df.to_excel("results.xlsx", index=False)
    
#     return risk

#------------------ End of Classification Functions ------------------#

    
    
#------------------ FastAPI End Points ------------------#
app = FastAPI(title="Molar Support with FastAPI")

# CORS (Cross-Origin Resource Sharing) configuration to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#------------------ Start of Check for Valid CBCT input ------------------#
from typing import Annotated    

# Create directory if it doesn't exist
if not os.path.exists("uploaded_images"):
    os.makedirs("uploaded_images")

@app.post("/check_valid_cbct") # https://fastapi.tiangolo.com/tutorial/request-forms-and-files/
async def check_valid_cbct(fileb: Annotated[UploadFile, File()] ):
    # Save the uploaded image to a temporary file
    input_image_path = f"uploaded_images/{fileb.filename}"
    with open(input_image_path, "wb") as temp_image:
        temp_image.write(fileb.file.read())

    # Check if the input image is a valid sliced CBCT input
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # resize = tf.image.resize(img, (256, 256))
    # input_data = np.expand_dims(resize / 255, 0)
    # prediction_input_check = model_input_check.predict(input_data)
    prediction_input_check = 0.4

    if prediction_input_check <= 0.5:
        return {"message": "The image is a valid sliced CBCT input."}
    else:
        # Delete the file if the image is not a valid CBCT image
        os.remove(input_image_path)
        return {"error": "The image is not a valid sliced CBCT input. Please upload a valid image."}

#------------------ End of Check for Valid CBCT input ------------------#

#------------------ Start of Checking for Valid MC CBCT Input------------------#
@app.post("/check_valid_cbct_mc")
async def check_valid_cbct(fileb: Annotated[UploadFile, File()] ):
    # Save the uploaded image to a temporary file
    input_image_path = f"uploaded_images/{fileb.filename}"
    with open(input_image_path, "wb") as temp_image:
        temp_image.write(fileb.file.read())

    # Check if the input image is a valid sliced CBCT input
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # resize = tf.image.resize(img, (256, 256))
    # input_data = np.expand_dims(resize / 255, 0)
    # prediction_input_check = model_input_check.predict(input_data)
    prediction_input_check = 0.4

    if prediction_input_check <= 0.5:
        return {"message": "The image is a valid sliced CBCT input."}
    else:
        # Delete the file if the image is not a valid CBCT image
        os.remove(input_image_path)
        return {"error": "The image is not a valid sliced CBCT input. Please upload a valid image."}
    
#------------------ End of Checking for Valid MC CBCT Input------------------#
    
#------------------ Start of Process Image for Segmentation Prediction ------------------#
@app.post("/process_image")
async def process_image_endpoint(file: Annotated[UploadFile, File()] ):
    # Save the uploaded image to a temporary file
    # input_image_path = "temp_input_image.jpg"
    input_image_path = file
    with open(input_image_path, "wb") as temp_image:
        temp_image.write(file.file.read())
    
    # Open the image again
    img = cv2.imread(input_image_path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    resize = tf.image.resize(img, (256, 256))
    input_data = np.expand_dims(resize / 255, 0)
    prediction_input_check = model_input_check.predict(input_data)

    if prediction_input_check > 0.5:
        return {"error": "The image is not a valid sliced CBCT input. Please upload a valid image."}

    # Perform segmentation, preprocessing, and overlay
    segmented_image = load_model_and_predict(input_image_path)
    preprocess_image = process_image(segmented_image)
    overlayed_image = overlay_images(preprocess_image, input_image_path)
    final_result = overlay_result_mc(overlayed_image)
    
    # for debugging
    print(f"Segmented Image: {segmented_image}")
    print(f"Preprocessed Image: {preprocess_image}")
    print(f"Overlayed Image: {overlayed_image}")
    print(f"Final Result: {final_result}")
    
    enhanced_image = enhance_colors(final_result, saturation_factor=2.5)

    # save the enchanced image
    enhanced_output_path = os.path.join("output_assets/enhanced_output/enhanced.jpg")
    
    cv2.imwrite(enhanced_output_path, enhanced_image)
    
    flat_violet_to_color(enhanced_output_path, (128, 47, 128))
    flatten_output_path = os.path.join("output_assets/enhanced_output/enhancedFinal.jpg")
    
    # for debugging
    print(f"Enhanced Image: {flatten_output_path}")
    
    return FileResponse(flatten_output_path, media_type="image/jpeg", filename="result.jpg")


# Add a new endpoint to serve the result image directly
@app.get("/result_image")
async def get_result_image():
    # You may need to replace "result.jpg" with the actual filename generated during processing
    result_image_path = "output_assets/distance_output/sample.jpg"
    return FileResponse(result_image_path, media_type="image/jpeg", filename="result.jpg")

#------------------ End of Process Image for Segmentation Prediction ------------------#

#------------------ Start of Classification Endpoints ------------------#
@app.post("/corticalization_type")
async def corticalization_type(fileb: Annotated[UploadFile, File()] ):
    try:
        # Read the image from the specified path
        image_path = "output_assets/enhanced_output/enhanced.jpg"
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        resize = tf.image.resize(img, (256, 256))
        input_data = np.expand_dims(resize / 255, 0)
        prediction_corticilization_type = model_corticilization_type.predict(input_data)

        if prediction_corticilization_type > 0.5:
            interruption_prediction = "Positive"
        else:
            interruption_prediction = "Negative"
        
        # Read the existing Excel file into a DataFrame
        df = pd.read_excel("results.xlsx")

        print(f"Prediction: {interruption_prediction}")
        # Append the new value to the "Interruption" column
        df.at[1, "Interruption"] = interruption_prediction
        

        # Save the updated DataFrame back to the Excel file
        df.to_excel("results.xlsx", index=False)

        return {"postInterruption": interruption_prediction}
    except Exception as e:
        print(f"Error in corticilization_type: {e}")
        raise HTTPException(status_code=500, detail="Error processing corticilization_type")

@app.post("/position_prediction")
def position(file: Annotated[UploadFile, File()]):
    try:
        # Read the image from the specified path
        # image_path = "output_assets/enhanced_output/enhanced.jpg"
        image_path = f"output_assets/enhanced_output/{file}"
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        resize = tf.image.resize(img, (224,224))
        yhat = model_position.predict(np.expand_dims(resize/255,0))
        
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

        # Read the existing Excel file into a DataFrame
        df = pd.read_excel("results.xlsx")

        # Append a new row to the DataFrame with the new value in the "Position" column
        df.at[1, "Position"] = position_label

        # Save the updated DataFrame back to the Excel file
        df.to_excel("results.xlsx", index=False)
        
        classify_relation()
        classify_risk()
        
        return {"postPosition": position_label}

    except Exception as e:
        print(f"Error in position: {e}")
        raise HTTPException(status_code=500, detail="Error processing position")


#------------------ End of Classification Endpoints ------------------#

#------------------ Start of Get Values ------------------#
@app.get("/getDistance")
def read_excel_value():

    # Calculate distance between M3 and MC using Euclidean Distance formula
    def calculate_distance(point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    '''
    - point1 and point2 are tuples representing the (x, y) coordinates of two points.
    - The function uses the Euclidean distance formula: distance = sqrt((x2 - x1)^2 + (y2 - y1)^2).
    '''

    def filter_color(image, lower, upper):
        mask = cv2.inRange(image, lower, upper)
        result = cv2.bitwise_and(image, image, mask=mask)
        return result

    '''
    - cv2.inRange(image, lower, upper) creates a binary mask where pixels within the specified color range are set to 1 and others to 0.
    - cv2.bitwise_and(image, image, mask=mask) applies the binary mask to the original image. It keeps only the pixels where the mask 
        is 1, effectively filtering out the colors outside the specified range.
    - The filter_color function is used to filter regions of the image containing the colors of 
        interest (purple and green) and create a combined mask (combined_regions) that represents both color regions. The subsequent 
        processing involves converting this combined mask to grayscale and detecting contours, ultimately leading to the identification 
        of objects based on their color characteristics.
    '''

    #169 - rgb(50,143,74) green , rgb(113,32,113) purple

    #169_predicted - 

    def detect_objects(image_path):
        # Load the image
        
        #dimensions = (355, 355)
        
        image = cv2.imread(image_path)
        
        #image = cv2.resize(image, dimensions)
        
        # Convert the image to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the purple color
        lower_purple = np.array([130, 50, 50])
        upper_purple = np.array([170, 255, 255])

        # Define the lower and upper bounds for the green color
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        # Filter out purple and green regions
        purple_regions = filter_color(hsv, lower_purple, upper_purple)
        green_regions = filter_color(hsv, lower_green, upper_green)

        # Combine the purple and green regions
        combined_regions = cv2.bitwise_or(purple_regions, green_regions)

        # Convert the combined image to grayscale
        gray = cv2.cvtColor(combined_regions, cv2.COLOR_BGR2GRAY)

        # Use a suitable method to detect objects, e.g., using contours
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours based on the topmost point of each contour
        contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])

        # Store points of each object in separate arrays
        object_points = []
        for contour in contours:
            points = np.array(contour[:, 0, :])
            object_points.append(points)

        # Calculate the least Euclidean distance between points of the two objects (this should be the true distance)
        min_distance = float('inf')
        point_a_min = None
        point_b_min = None
        # for point_a in object_points[0]:
        #     for point_b in object_points[1]:
        #         distance = calculate_distance(tuple(point_a), tuple(point_b))
        #         if distance < min_distance:
        #             min_distance = distance
        #             point_a_min = point_a
        #             point_b_min = point_b
        # Handle Exception
        try:
            for point_a in object_points[0]:
                for point_b in object_points[1]:
                    distance = calculate_distance(tuple(point_a), tuple(point_b))
                    if distance < min_distance:
                        min_distance = distance
                        point_a_min = point_a
                        point_b_min = point_b
        except IndexError as e:
            # print(f"An IndexError occurred: {e}")
            min_distance = 0

        # Display information about the detected objects
        min_distance *= 0.15510299643
        # print(min_distance - 2.7) # 8.462783060825256
        # min_distance -= 8.462783060825256
        min_distance = min_distance if min_distance > 0.5 else 0
        min_distance = round(min_distance, 2)

        # Draw a blue line connecting the closest points of the two objects
        if point_a_min is not None and point_b_min is not None:
            cv2.line(image, tuple(point_a_min), tuple(point_b_min), (255, 0, 0), 2)

            # Display the value of min_distance on top of the line
            text_position = ((point_a_min[0] + point_b_min[0]) // 2, (point_a_min[1] + point_b_min[1]) // 2)
            cv2.putText(image, f"{min_distance:.2f} mm", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        cv2.imwrite('output_assets/distance_output/distance.jpg', image)
        
        # Load the Excel file into a pandas DataFrame
        df = pd.read_excel("results.xlsx")
            
        # Append the new value to the "Distance" column
        df.at[1, "Distance"] = min_distance
            
        # Save the updated DataFrame back to the Excel file
        df.to_excel("results.xlsx", index=False)

        return min_distance
        

    detect_objects('output_assets/enhanced_output/enhanced.jpg')
    
    try:
        df = pd.read_excel("results.xlsx")
        # Get the value from the specified row and column
        value = df.at[1, "Distance"]

        # Return the value as a message
        return {"distance": f"{value}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/getPosition")
async def read_excel_value():
    
    # Load the Excel file into a pandas DataFrame
    df = pd.read_excel("results.xlsx")
    
    try:
        # Get the value from the specified row and column
        value = df.at[1, "Position"]

        # Return the value as a message
        return {"position": f"{value}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/getInterruption")
async def read_excel_value():
        
        # Load the Excel file into a pandas DataFrame
        df = pd.read_excel("results.xlsx")
        
        try:
            # Get the value from the specified row and column
            value = df.at[1, "Interruption"]
    
            # Return the value as a message
            return {"interruption": f"{value}"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/getRelation")
async def read_excel_value():
    
    # Load the Excel file into a pandas DataFrame
    df = pd.read_excel("results.xlsx")
    
    try:
        # Get the value from the specified row and column
        value = df.at[1, "Relation"]

        # Return the value as a message
        return {"relation": f"{value}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/getRisk")
async def read_excel_value():
    
    # Load the Excel file into a pandas DataFrame
    df = pd.read_excel("results.xlsx")
    
    try:
        # Get the value from the specified row and column
        value = df.at[1, "Risk"]

        # Return the value as a message
        return {"risk": f"{value}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


#------------------ End of Get Values ------------------#

#------------------ ROD CHANGES ------------------#

# FUNCTIONS

# START FOR CHECK IF VALID CBCT IMAGE ------------------

# params: image file
# returns boolean True or False
# Others: saves uploaded image if it is a valid CBCT image in /output_assets/uploaded_images/file_name.jpg

def check_valid_cbct(file: Annotated[UploadFile, File()] ):
    # Save the uploaded image to a temporary file
    input_image_path = f"uploaded_images/{file.filename}"
    with open(input_image_path, "wb") as temp_image:
        temp_image.write(file.file.read())

    # Check if the input image is a valid sliced CBCT input
    img = cv2.imread(input_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # resize = tf.image.resize(img, (256, 256))
    # input_data = np.expand_dims(resize / 255, 0)
    # prediction_input_check = model_input_check.predict(input_data)
    prediction_input_check = 0.4

    if prediction_input_check <= 0.5:
        # return {"message": "The image is a valid sliced CBCT input."}
        return True
    else:
        # Delete the file if the image is not a valid CBCT image
        os.remove(input_image_path)
        # return {"error": "The image is not a valid sliced CBCT input. Please upload a valid image."}
        return False

# END FOR CHECK IF VALID CBCT IMAGE ------------------
    
# START FOR PROCESSING CBCT IMAGE ------------------

# params: image file
# returns enhanced_output_path, flatten_output_path
# Others: performs segmentation, preprocessing, and overlay then enhancing the image and saves it in /output_assets/enhanced_output/file_name.jpg
def process_image(file: Annotated[UploadFile, File()] ):
    # Save the uploaded image to a temporary file
    # input_image_path = "temp_input_image.jpg"
    input_image_path = file
    with open(input_image_path, "wb") as temp_image:
        temp_image.write(file.file.read())
    
    # Open the image again
    img = cv2.imread(input_image_path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    resize = tf.image.resize(img, (256, 256))
    input_data = np.expand_dims(resize / 255, 0)
    prediction_input_check = model_input_check.predict(input_data)

    if prediction_input_check > 0.5:
        return {"error": "The image is not a valid sliced CBCT input. Please upload a valid image."}

    # Perform segmentation, preprocessing, and overlay
    segmented_image = load_model_and_predict(input_image_path)
    preprocess_image = process_image(segmented_image)
    overlayed_image = overlay_images(preprocess_image, input_image_path)
    final_result = overlay_result_mc(overlayed_image)
    
    # for debugging
    print(f"Segmented Image: {segmented_image}")
    print(f"Preprocessed Image: {preprocess_image}")
    print(f"Overlayed Image: {overlayed_image}")
    print(f"Final Result: {final_result}")
    
    enhanced_image = enhance_colors(final_result, saturation_factor=2.5)

    # get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    str_current_datetime = str(current_datetime)

    # create a file object along with extension
    enhanced_output_path = "output_assets/enhanced_output/enhanced_result-"+str_current_datetime+".jpg"

    # save the enchanced image
    # enhanced_output_path = os.path.join("output_assets/enhanced_output/enhanced.jpg")
    
    cv2.imwrite(enhanced_output_path, enhanced_image)
    
    flat_violet_to_color(enhanced_output_path, (128, 47, 128))

    # create a file object along with extension
    flatten_output_path = "output_assets/enhanced_output/enhanced_finalResult-"+str_current_datetime+".jpg"

    # flatten_output_path = os.path.join("output_assets/enhanced_output/enhancedFinal.jpg")
    
    # for debugging
    print(f"Enhanced Image: {flatten_output_path}")
    
    # return FileResponse(flatten_output_path, media_type="image/jpeg", filename="result.jpg")
    return enhanced_output_path, flatten_output_path


# END FOR PROCESSING CBCT IMAGE ------------------

# START FOR CORTICALIZATION TYPE ------------------

# params: image file
# returns interruption_prediction
# Others:
def corticalization_type(fileb: Annotated[UploadFile, File()] ):
    try:
        # Read the image from the specified path
        image_path = "output_assets/enhanced_output/enhanced.jpg"
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        resize = tf.image.resize(img, (256, 256))
        input_data = np.expand_dims(resize / 255, 0)
        prediction_corticilization_type = model_corticilization_type.predict(input_data)

        if prediction_corticilization_type > 0.5:
            interruption_prediction = "Positive"
        else:
            interruption_prediction = "Negative"
        
        # Read the existing Excel file into a DataFrame
        # df = pd.read_excel("results.xlsx")

        print(f"Prediction: {interruption_prediction}")
        # # Append the new value to the "Interruption" column
        # df.at[1, "Interruption"] = interruption_prediction
        

        # # Save the updated DataFrame back to the Excel file
        # df.to_excel("results.xlsx", index=False)

        # return {"postInterruption": interruption_prediction}
        return interruption_prediction
    except Exception as e:
        print(f"Error in corticilization_type: {e}")
        raise HTTPException(status_code=500, detail="Error processing corticilization_type")
    
# END FOR CORTICALIZATION TYPE ------------------

# START FOR POSITION ------------------

# params: image file
# returns interruption_prediction
# Others:
def position(file: Annotated[UploadFile, File()]):
    try:
        # Read the image from the specified path
        # image_path = "output_assets/enhanced_output/enhanced.jpg"
        image_path = f"output_assets/enhanced_output/{file}"
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        resize = tf.image.resize(img, (224,224))
        yhat = model_position.predict(np.expand_dims(resize/255,0))
        
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

        # Read the existing Excel file into a DataFrame
        df = pd.read_excel("results.xlsx")

        # Append a new row to the DataFrame with the new value in the "Position" column
        df.at[1, "Position"] = position_label

        # Save the updated DataFrame back to the Excel file
        df.to_excel("results.xlsx", index=False)
        
        classify_relation()
        classify_risk()
        
        return {"postPosition": position_label}

    except Exception as e:
        print(f"Error in position: {e}")
        raise HTTPException(status_code=500, detail="Error processing position")
    
# END FOR POSITION ------------------

# START FOR DISTANCE ------------------

# params: image file path
# returns float value minimum distance
# Others: saves image with distance overlay in /output_assets/distance_output/file_name.jpg

# Calculate distance between M3 and MC using Euclidean Distance formula
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

'''
- point1 and point2 are tuples representing the (x, y) coordinates of two points.
- The function uses the Euclidean distance formula: distance = sqrt((x2 - x1)^2 + (y2 - y1)^2).
'''

def filter_color(image, lower, upper):
    mask = cv2.inRange(image, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result
    
def detect_objects(image_path):
    # Load the image

    #dimensions = (355, 355)

    image = cv2.imread(image_path)

    #image = cv2.resize(image, dimensions)

    # Convert the image to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the purple color
    lower_purple = np.array([130, 50, 50])
    upper_purple = np.array([170, 255, 255])

    # Define the lower and upper bounds for the green color
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    # Filter out purple and green regions
    purple_regions = filter_color(hsv, lower_purple, upper_purple)
    green_regions = filter_color(hsv, lower_green, upper_green)

    # --- CHANGE THE GREEN AND PURPLE REGIONS TO ONLY ONE SHADE ---
    
    # Filter out green regions for pixel replacement
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    green_points = cv2.findNonZero(green_mask)

    # Assign the desired green color to all green points
    green_color = (16, 119, 26)
    for point in green_points:
        x, y = point[0]
        image[y, x] = green_color

    # Filter out purple regions for pixel replacement
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)
    purple_points = cv2.findNonZero(purple_mask)

    # Assign the desired purple color to all purple points
    purple_color = (101,49,142)
    for point in purple_points:
        x, y = point[0]
        image[y, x] = purple_color

    # --- CHANGE THE GREEN AND PURPLE REGIONS TO ONLY ONE SHADE ---

    # Combine the purple and green regions
    combined_regions = cv2.bitwise_or(purple_regions, green_regions)

    # Convert the combined image to grayscale
    gray = cv2.cvtColor(combined_regions, cv2.COLOR_BGR2GRAY)

    # Use a suitable method to detect objects, e.g., using contours
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on the topmost point of each contour
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])

    # Store points of each object in separate arrays
    object_points = []
    for contour in contours:
        points = np.array(contour[:, 0, :])
        object_points.append(points)

    # print("Len:", len(object_points))

    if len(object_points) > 2: # if there are more than 2 detected contours i.e. other teeth, choose the two at the bottom
        # Get the last two contours
        last_two_contours = object_points[-2:]
    
        # Remove the last two contours from object_points
        object_points = object_points[:-2]
    
        # Append the last two contours to the beginning of object_points
        object_points.insert(0, last_two_contours[0])
        object_points.insert(1, last_two_contours[1])

    # Calculate the least Euclidean distance between points of the two objects (this should be the true distance)
    min_distance = float('inf')
    point_a_min = None
    point_b_min = None
    # for point_a in object_points[0]:
    #     for point_b in object_points[1]:
    #         distance = calculate_distance(tuple(point_a), tuple(point_b))
    #         if distance < min_distance:
    #             min_distance = distance
    #             point_a_min = point_a
    #             point_b_min = point_b
    # Handle Exception
    try:
        for point_a in object_points[0]:
            for point_b in object_points[1]:
                distance = calculate_distance(tuple(point_a), tuple(point_b))
                if distance < min_distance:
                    min_distance = distance
                    point_a_min = point_a
                    point_b_min = point_b
    except IndexError as e:
        # print(f"An IndexError occurred: {e}")
        min_distance = 0

    # Display information about the detected objects
    min_distance *= 0.15510299643
    # print(min_distance - 2.7) # 8.462783060825256
    # min_distance -= 8.462783060825256
    min_distance = min_distance if min_distance > 0.5 else 0
    min_distance = round(min_distance, 2)

    # Draw a blue line connecting the closest points of the two objects
    if point_a_min is not None and point_b_min is not None:
        cv2.line(image, tuple(point_a_min), tuple(point_b_min), (255, 0, 0), 2)

        # Display the value of min_distance on top of the line
        text_position = ((point_a_min[0] + point_b_min[0]) // 2, (point_a_min[1] + point_b_min[1]) // 2)
    else:
        # If no points found, set distance to 0.0 and display it
        min_distance = 0.0
        # Position the text at the center of the image
        text_position = (image.shape[1] // 2, image.shape[0] // 2)

    # Display the distance on the image
    cv2.putText(image, f"{min_distance:.2f} mm", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # get current date and time
    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    str_current_datetime = str(current_datetime)

    # create a file object along with extension
    file_name = "output_assets/distance_output/distance_result-"+str_current_datetime+".jpg"
    cv2.imwrite(file_name, image)

    return min_distance

# END FOR DISTANCE ------------------

# START FOR RELATION ------------------

# params: distance, position, interruption values
# returns relation classification
# Others:
def classify_relation(distance, position, interruption):
    
    # # load excel file in dataframe
    # df = pd.read_excel("results.xlsx")
    
    # # get the values from the dataframe
    # distance = df.at[1, "Distance"].astype(float)
    # position = df.at[1, "Position"]
    # interruption = df.at[1, "Interruption"]
    
    # If else statements for classification
    
    if distance == 0.0 and interruption == "Negative" and position == "None":
        relation = "Class 0"
        
    elif position == "Buccal" or position == "Apical" and interruption == "Negative":
        if distance > 2.0:
            relation = "Class 1A"
        else:
            relation = "Class 1B"
    
    elif position == "Lingual" and interruption == "Negative":
        if distance > 2.0:
            relation = "Class 2A"
        else:
            relation = "Class 2B"
    
    elif position == "Buccal" or position == "Apical" and interruption == "Positive":
       if distance > 1.0:
           relation = "Class 3A"
       else:
           relation = "Class 3B"
           
    elif position == "Lingual" and interruption == "Positive":
        if distance > 1.0:
            relation = "Class 4A"
        else:
            relation = "Class 4B"
    
    else:
        relation = "Unclassified Relation"
    
    print(f"Relation: {relation}")
    
    # # Append the new value to the "Relation" column
    # df.at[1, "Relation"] = relation
    
    # # Save the updated DataFrame back to the Excel file
    # df.to_excel("results.xlsx", index=False)
    
    return relation

# END FOR RELATION ------------------

# START FOR RISK ------------------

# params: relation
# returns risk classification
# Others:
def classify_risk(relation):
    
    # # load excel file in dataframe
    # df = pd.read_excel("results.xlsx")
    
    # # get the values from the dataframe
    # relation = df.at[1, "Relation"]
    
    # If else statements for classification
    if relation == "Class 0":
        risk = "N.0 (Non-determinant)"
    
    elif relation == "Class 1A" or relation == "Class 1B" or relation == "Class 2A" or relation == "Class 2B" or relation == "Class 4A":
        risk = "N.1 (Low)"
    
    elif relation == "Class 3A" or relation == "Class 3B":
        risk = "N.2 (Medium)"
    
    elif relation == "Class 4B":
        risk = "N.3 (High)"
    
    else:
        risk = "Unclassified Risk"
    
    print(f"Risk: {risk}")
    
    # # Append the new value to the "Risk" column
    # df.at[1, "Risk"] = risk
    
    # # Save the updated DataFrame back to the Excel file
    # df.to_excel("results.xlsx", index=False)
    
    return risk

# END FOR RISK ------------------

#------------------ Start of Main Driver ------------------#

# params: image file paths for MC and M3
# returns MolarCase class from the SQLite database
# Others: saves image with distance overlay in /output_assets/distance_output/file_name.jpg

@app.post("/mainMolarSupportDriver")
async def root(
    mc_img: Annotated[UploadFile, File()], # mandibular canal
    m3_img: Annotated[UploadFile, File()] # mandibular third molar
    ):
# 1. Check if valid M3 and MC
    text_result_11 = check_valid_cbct(mc_img)
    if text_result_11 == False:
        raise HTTPException(status_code=500, detail="Image is not an MC image.")
    # text_result_11 = "path_to_mc_file"
    text_result_12 = check_valid_cbct(m3_img)
    if text_result_12 == False:
        raise HTTPException(status_code=500, detail="Image is not an M3 image.")
    # text_result_12 = "path_to_m3_file"
# 2. Process image
    # img_result_21 = process_image(m3_img)
# 3. Corticalization
    # text_result_31 = corticalization_type(img_result_21)
    text_result_31 = False
# 4. Position prediction
    # text_result_41 = position(img_result_21)
    text_result_41 = "Buccal"
# 5. Distance measurement
    # text_result_51 = detect_objects(img_result_21)
    text_result_51 = 3.14
# 6. Classify relation
    text_result_61 = classify_relation(text_result_51, text_result_41, text_result_31)
# 7. Classify risk
    text_result_71 = classify_risk(text_result_61)
# 8. Add to SQLite database
    create_table() # call the function to create molarcases table

    # Create a MolarCaseCreate instance
    new_case = MolarCaseCreate(
        mc_filename="path_to_mc_file",
        m3_filename="path_to_m3_file",
        generated_m3_mask_filename="path_to_original_m3_mask",
        final_img_filename="path_to_m3_mask_prediction",
        corticalization=False,
        position="Buccal",
        distance=3.14,
        final_image_distance="path_to_image_with_distance",
        relation=text_result_61,
        risk=text_result_71
    )

    create_case(new_case)

    # return a MolarCase:
    # [mc_file_upload, m3_file_upload, original_m3_mask, m3_mask_prediction, layered_image, corticalization, position, distance, image_with_distance, relation, risk]
    # print("Done.")
    print(new_case)
    return f"Done. \n {new_case}"

# REFER HERE FOR THE IMPORTANT DETAILS TO RETURN AND DISPLAY
# Class for the Molar Case
# class MolarCaseCreate(BaseModel):
#     mc_file_upload: str
#     m3_file_upload: str
#     original_m3_mask: str
#     m3_mask_prediction: str
#     layered_image: str
#     corticalization: str
#     position: str
#     distance: float
#     image_with_distance: str
#     relation: str
#     risk: str\

from typing import List

# GET ENDPOINT FOR SHOWING ALL ENTRIES IN SQLITE DATABASE
@app.get("/molarcases/", response_model=List[MolarCase])
async def get_molar_cases():
    cases = get_all_cases()
    return [{"id": row[0], "mc_filename": row[1], "m3_filename": row[2], "generated_m3_mask_filename": row[3],
             "final_img_filename": row[4], "corticalization": row[5], "position": row[6],
             "distance": row[7], "final_image_distance": row[8], "relation": row[9], "risk": row[10]} for row in cases]

# DELETE endpoint to delete the molarcases table
@app.delete("/molarcases/table")
async def delete_molarcases_table():
    delete_table()
    return {"message": "molarcases table deleted successfully"}

#------------------ End of Main Driver ------------------#

#------------------ ROD CHANGES ------------------#
    
if __name__ == '__main__':
    app.run()