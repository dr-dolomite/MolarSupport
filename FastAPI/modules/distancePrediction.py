import cv2
import numpy as np
import os

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

def detect_objects(session_id):
    # Load the image
    image_path = "output_images/enhanced_output/enhanced_final.jpg"
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
        cv2.putText(image, f"{min_distance:.2f} mm", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 3)
    
    # Create the folder if it does not exist
    if not os.path.exists("output_images/distance_output/"):
        os.makedirs("output_images/distance_output/")
        
    cv2.imwrite('output_images/distance_output/ouput_with_distance.jpg', image)
    
    # Write a temp image for public directory
    # Check first if temp.jpg already exists
    temp_img_path = '../public/temp-result/temp-' + session_id + '.jpg'
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)
    cv2.imwrite(temp_img_path, image)
    
    return min_distance