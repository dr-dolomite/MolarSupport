import cv2
import os
 
from .cleaningUtils import remove_small_spots, smoothen_edges, contour_filter

def process_image(original_segmented_image, min_size_threshold=5000):
    # Load your segmented image in color mode
    segmented_image = cv2.imread(original_segmented_image)

    # Split the image into individual color channels
    b, g, r = cv2.split(segmented_image)

    # Remove small spots for each channel
    cleaned_b = remove_small_spots(b, min_size_threshold)
    cleaned_g = remove_small_spots(g, min_size_threshold)
    cleaned_r = remove_small_spots(r, min_size_threshold)

    # Smoothen image edges for each channel
    smoothen_b = smoothen_edges(cleaned_b)
    smoothen_g = smoothen_edges(cleaned_g)
    smoothen_r = smoothen_edges(cleaned_r)

    # Merge the channels back into a single image
    smoothen_image = cv2.merge([smoothen_b, smoothen_g, smoothen_r])
    
    # Apply contour filtering
    filtered_image = contour_filter(smoothen_image, min_size_threshold)

    # Add gaussian blur
    #blur_image = cv2.GaussianBlur(filtered_image, (15, 15), 0)
    
    # Save the processed image
    
    # Use the old image name, remove "_predicted" and append "_cleaned" to it
    old_image_name = os.path.splitext(os.path.basename(original_segmented_image))[0]
    
    # Remove the "_predicted" from the old image name and append "_cleaned" to it
    output_image_name = old_image_name.replace("_predicted", "_cleaned")
    
    # Define the output path
    output_path = os.path.join("output_images/cleaned_segmentation/", output_image_name + ".jpg")
    
    # Create the folder if it does not exist
    if not os.path.exists("output_images/cleaned_segmentation/"):
        os.makedirs("output_images/cleaned_segmentation/")
    
    # Save the processed image
    cv2.imwrite(output_path, filtered_image)
    
    return output_path