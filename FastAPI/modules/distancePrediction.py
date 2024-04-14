import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Function to convert OpenCV BGR image to RGB
def convert_to_rgb_and_display(image, title):
    image_contour_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Display the images using Matplotlib
    plt.figure(figsize=(10, 5))
    
    # Display the contour result
    plt.imshow(image_contour_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function to cluster points based on distance
def cluster_points(points, eps=1.5, min_samples=5):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(points)
    labels = dbscan.labels_
    unique_labels = set(labels)
    clustered_points = [points[labels == label] for label in unique_labels]
    return clustered_points

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
    contours = sorted(contours, key=lambda x: (cv2.boundingRect(x)[1] + cv2.boundingRect(x)[3]))

    # Store points of each object in separate arrays
    object_points = []
    object_colors = []

    for contour in contours:
        points = np.array(contour[:, 0, :])
        object_points.append(points)
        if np.any(purple_mask[points[:, 1], points[:, 0]] > 0):
            object_colors.append('purple')
        else:
            object_colors.append('green')

    # Combine contours with the same color that are at most 1.5 mm apart
    combined_object_points = []
    combined_object_colors = []

    for color in set(object_colors):
        color_indices = [i for i, c in enumerate(object_colors) if c == color]
        color_points = [object_points[i] for i in color_indices]
        flat_color_points = np.concatenate(color_points)
        clustered_points = cluster_points(flat_color_points, eps=1.5)
        combined_object_points.extend(clustered_points)
        combined_object_colors.extend([color] * len(clustered_points))

    # Update object_points and object_colors with the combined contours
    object_points = combined_object_points
    object_colors = combined_object_colors

    if len(object_points) > 2: # if there are more than 2 detected contours i.e. other teeth, choose the two at the bottom
        # Get the last two contours
        last_two_contours = object_points[-2:]
    
        # Remove the last two contours from object_points
        object_points = object_points[:-2]
    
        # Append the last two contours to the beginning of object_points
        object_points.insert(0, last_two_contours[0])
        object_points.insert(1, last_two_contours[1])

    image_contour = image.copy()
    cv2.drawContours(image_contour,[object_points[0]],0,(255,255,255), thickness=2, lineType=cv2.LINE_AA) # M3
    cv2.drawContours(image_contour,[object_points[1]],0,(255,255,255), thickness=2, lineType=cv2.LINE_AA) # MC
    # Convert images to RGB format
    convert_to_rgb_and_display(image_contour, "Image Contour of M3 and MC")

    # Reorganize object_points based on color detection criteria
    reorganized_object_points = []

    # Start from the last contour and move backward
    for i in range(len(object_points) - 1, -1, -1):
        # Check if the contour consists of purple color
        is_purple = False
        for point in object_points[i]:
            x, y = point
            if purple_mask[y, x] > 0:
                is_purple = True
                break
        
        # Check if the current contour consists of green color
        if not is_purple:
            # If the current contour consists of green color,
            # move it to the beginning of reorganized_object_points
            reorganized_object_points.insert(0, object_points[i])
        else:
            # If the current contour consists of purple color,
            # check the previous contour
            if i > 0:
                # Check if the previous contour consists of purple color
                is_prev_purple = False
                for point in object_points[i - 1]:
                    x, y = point
                    if purple_mask[y, x] > 0:
                        is_prev_purple = True
                        break
                
                # If the previous contour consists of purple color,
                # move to the next contour
                if is_prev_purple:
                    continue
                else:
                    # Otherwise, move both the current and previous contours
                    # to the beginning of reorganized_object_points
                    reorganized_object_points.insert(0, object_points[i])
                    reorganized_object_points.insert(0, object_points[i - 1])
                    break

    # Add the remaining contours to reorganized_object_points
    for i in range(len(object_points)):
        obj_points_list = [tuple(point) for point in object_points[i]]
        if obj_points_list not in [tuple(arr) for arr in reorganized_object_points]:
            reorganized_object_points.append(object_points[i])

    # Update object_points with the reorganized contours
    object_points = reorganized_object_points

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