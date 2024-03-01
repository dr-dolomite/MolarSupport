import cv2
import os

def overlay_images(blur_image, original_image, alpha=0.5):
    # Read the original image
    original = cv2.imread(original_image)

    # Check if the original image is loaded successfully
    if original is None:
        print(f"Error: Unable to load original image.")
        return

    # Try to read the segmented image
    try:
        segmented = cv2.imread(blur_image)

        # Check if the segmented image is loaded successfully
        if segmented is None:
            print(f"Error: Unable to load segmented image.")
            return

        # Resize the segmented image to match the original image dimensions
        segmented = cv2.resize(segmented, (original.shape[1], original.shape[0]))

        # Blend the images using alpha blending
        blended = cv2.addWeighted(original, 1 - alpha, segmented, alpha, 0)

    except cv2.error as e:
        print(f"OpenCV Error: {e}")
        
    # Save the result
    overlay_output_path = os.path.join("output_images/initial_overlay/m3_temp_initial.jpg")
    
    # Create the folder if it does not exist
    if not os.path.exists("output_images/initial_overlay/"):
        os.makedirs("output_images/initial_overlay/")
    
    cv2.imwrite(overlay_output_path, blended)
    
    return overlay_output_path


def overlay_result_mc(overlay_output, mc_image_path, alpha=0.4):
    
    with_mc = cv2.imread(mc_image_path)
    overlayed = cv2.imread(overlay_output)
    
    with_mc = cv2.resize(with_mc, (overlayed.shape[1], overlayed.shape[0]))
    
    # blend the images
    blended_final = cv2.addWeighted(overlayed, 1 - alpha, with_mc, alpha, 0)
    
    # Save the result
    final_overlayed_path = os.path.join("output_images/overlayed_output/m3_temp_overlayed.jpg")
    
    # Create the folder if it does not exist
    if not os.path.exists("output_images/overlayed_output/"):
        os.makedirs("output_images/overlayed_output/")
    
    cv2.imwrite(final_overlayed_path, blended_final)
    
    return final_overlayed_path