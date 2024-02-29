import cv2
import os
import numpy as np

def remove_small_spots(segmented_image, min_size):
    # Find connected components
    _, labels, stats, _ = cv2.connectedComponentsWithStats(segmented_image, connectivity=4)

    # Filter out small spots
    for i, stat in enumerate(stats):
        if stat[4] < min_size:  # Check the area (stat[4]) of the connected component
            segmented_image[labels == i] = 0

    return segmented_image

def contour_filter(segmented_image, min_size_threshold):
    # Convert to grayscale
    gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)

    # Apply morphological operations
    kernel = np.ones((5, 5), np.uint8)
    morph_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)

    # Find contours on the morphological image
    contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_size_threshold:
            cv2.drawContours(segmented_image, [contour], 0, (0, 0, 0), -1)  # Fill contour with black (0, 0, 0)

    return segmented_image


def smoothen_edges(cleaned_image):
    # Applying the filter 
    bilateral = cv2.bilateralFilter(cleaned_image, 9, 75, 75)
    
    return bilateral

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
    result_image.save('output_assets/enhanced_output/enhanced.jpg')

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
    
    return enhanced_img
