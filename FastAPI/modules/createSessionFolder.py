import os
import datetime
import shutil

datetime_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

def createSessionFolder():
    # Create a new folder for the final output images based on the current date and time
    folder_with_date = "session_" + datetime_now
    new_output_folder = "session_output_images/" + folder_with_date
    
    if not os.path.exists(new_output_folder):
            os.makedirs(new_output_folder)
        
    # Copy the "output_images" folder and "input_iamges" folder to the new folder
    shutil.copytree("output_images", new_output_folder + "/output_images")
    shutil.copytree("input_images", new_output_folder + "/input_images")

    return folder_with_date, new_output_folder
    
    
    