import os
import shutil

def clean_directories():
    # Clean the contents of each of the folder of output_images folder
    folders = ["cleaned_segmentation", "initial_overlay", "overlayed_output", "preprocess_input", "distance_output", "enhanced_output"]
    for folder in folders:
        if os.path.exists("output_images/" + folder):
            shutil.rmtree("output_images/" + folder)
            os.makedirs("output_images/" + folder)
        else:
            os.makedirs("output_images/" + folder)
    
    # Clean the contents of the input_images folder
    folders = ["m3_cbct", "mc_cbct"]
    for folder in folders:
        if os.path.exists("input_images/" + folder):
            shutil.rmtree("input_images/" + folder)
            os.makedirs("input_images/" + folder)
        else:
            os.makedirs("input_images/" + folder)