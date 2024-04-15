import os
from .model import UNET
from .utils import load_checkpoint
from .prepocess import preprocess_input
import torch
import torchvision.transforms.functional as TF
import numpy as np
from PIL import Image

def load_model_and_predict(input_image_path):
    # Load the pre-trained model checkpoint
    checkpoint_path = "modules/model_checkpoint/my_checkpoint.pth.tar"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = UNET(in_channels=3, out_channels=1)
    load_checkpoint(torch.load(checkpoint_path, map_location ='cpu'), model)
    model = model.to(device).eval()

    # Load and preprocess the input image
    image_path = preprocess_input(input_image_path)
    
    image = Image.open(image_path).convert("RGB")
    # image = Image.open(image_path)
    image = TF.resize(image, (400, 400), antialias=True)
    image_tensor = TF.to_tensor(image).unsqueeze(0).to(device)

    # Perform segmentation on the input image
    with torch.no_grad():
        prediction = model(image_tensor)
        mask = torch.sigmoid(prediction) > 0.5
        mask = mask.squeeze(0).cpu().numpy().astype(np.uint8)

    # Create the colored segmentation mask
    color_mask = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)  # Modified line
    color_mask[mask[0] == 1] = (57, 150, 81)  # Green color (RGB values)  # Modified line

    # Convert the colored mask to PIL Image and save it
    color_mask_image = Image.fromarray(color_mask)
    
    # Save the segmented image. If directory is not present, create it
    if not os.path.exists("output_images/predicted_segmentation/"):
        os.makedirs("output_images/predicted_segmentation/")
    
    # Use the original image name and append "_predicted" to it
    segmentation_output_path = os.path.join("output_images/predicted_segmentation/", os.path.splitext(os.path.basename(input_image_path))[0] + "_predicted.jpg")
    
    color_mask_image.save(segmentation_output_path)
    
    return segmentation_output_path