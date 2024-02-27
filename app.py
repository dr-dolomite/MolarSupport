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

# --- Create a Database with SQLite ---
import sqlite3

# Class for the Molar Case
class MolarCaseCreate(BaseModel):
    mc_file_upload: str
    m3_file_upload: str
    original_m3_mask: str
    m3_mask_prediction: str
    layered_image: str
    distance: float
    image_with_distance: str
    corticalization: str
    position: str
    risk: str

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
                        mc_file_upload TEXT NOT NULL,
                        m3_file_upload TEXT NOT NULL,
                        distance FLOAT NOT NULL,
                        corticalization TEXT NOT NULL,
                        position TEXT NOT NULL,
                        risk TEXT NOT NULL
                );
                    """)
    connection.commit()
    connection.close()

create_table() # call the function to create molarcases table

def create_case(case: MolarCaseCreate): # (CRUD) Create molar case
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("INSERT INTO molarcases (mc_file_upload, m3_file_upload, distance, corticalization, position, risk) VALUES (?, ?, ?, ?, ?, ?)", 
                   (case.mc_file_upload, case.m3_file_upload, case.distance, case.corticalization, case.position, case.risk))
    connection.commit()
    connection.close()


# --------------- SQLITE AREA ---------------

# # Load the model for input classification
# # model_input_check = load_model(os.path.join('model_checkpoint', 'inputClassification.h5'))
# # model_corticilization_type = load_model(os.path.join('model_checkpoint', 'cortiClassification.h5'))
# # model_position = load_model(os.path.join('model_checkpoint', 'vgg16_checkpoint.h5'))

# # ------------------ Start of Segmentation Process Functions ------------------#
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


# #------------------ Start of Classification Functions ------------------#

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

# --------------- SQLITE AREA ---------------

@app.post("/molarcases")
def molarCases_endpoint(MolarCase: MolarCaseCreate):
    molar_id = create_case(MolarCase)
    return {"id": molar_id, **MolarCase.dict()}

# --------------- SQLITE AREA ---------------


#------------------ Start of Check for Valid CBCT input ------------------#
from typing import Annotated    

# Create directory if it doesn't exist
if not os.path.exists("uploaded_images"):
    os.makedirs("uploaded_images")

@app.post("/check_valid_cbct") # https://fastapi.tiangolo.com/tutorial/request-forms-and-files/
async def check_valid_cbct(file: Annotated[bytes, File()], fileb: Annotated[UploadFile, File()] ):
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
async def check_valid_cbct(file: Annotated[bytes, File()], fileb: Annotated[UploadFile, File()] ):
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
async def process_image_endpoint(file: UploadFile = File(...)):
    # Save the uploaded image to a temporary file
    input_image_path = "temp_input_image.jpg"
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
@app.post("/corticilization_type")
async def corticilization_type():
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
async def position():
    try:
        # Read the image from the specified path
        image_path = "output_assets/enhanced_output/enhanced.jpg"
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
async def read_excel_value():

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
    
if __name__ == '__main__':
    app.run()