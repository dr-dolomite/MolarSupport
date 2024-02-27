"""
FastAPI application for the Molar Support project.
Brief description of what the program does.
Copyright (C) 2024 Russel Yasol

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

For contact information, reach out to russel.yasol@gmail.com
"""

import cv2
import os
import datetime

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
from typing import Annotated
import sqlite3
import cuid

# Avoid Out Of Memory (OOM) errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices("GPU")
print(gpus)

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Import the models
position_model = load_model(
    os.path.join("modules/model_checkpoint", "vgg16_checkpoint.h5")
)
inputValidityModel = load_model(
    os.path.join("modules/model_checkpoint", "inputClassification.h5")
)
cortical_model = load_model(
    os.path.join("modules/model_checkpoint", "cortiClassification.h5")
)


# Define the database model
class MolarCase(BaseModel):
    session_id: str
    session_folder: str
    final_img_filename: str
    corticalization: str
    position: str
    distance: str
    relation: str
    risk: str


# --------------- SQLITE AREA ---------------#
def create_connection():  # establish connection
    connection = sqlite3.connect("db/molarcases.db")
    return connection


def create_table():  # create table for molar cases
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute(
        """
                CREATE TABLE IF NOT EXISTS molarcases (
                        session_id TEXT PRIMARY KEY NOT NULL,
                        session_folder TEXT NOT NULL,
                        final_img_filename TEXT NOT NULL,
                        corticalization TEXT NOT NULL,
                        position TEXT NOT NULL,
                        distance FLOAT NOT NULL,
                        relation TEXT NOT NULL,
                        risk TEXT NOT NULL
                );
                    """
    )
    connection.commit()
    connection.close()


create_table()


def create_case(case: MolarCase):  # (CRUD) Create molar case
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO molarcases (session_id, session_folder, final_img_filename, corticalization, position, distance, relation, risk) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            case.session_id,
            case.session_folder,
            case.final_img_filename,
            case.corticalization,
            case.position,
            case.distance,
            case.relation,
            case.risk,
        ),
    )
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


# --------------- SQLITE AREA ---------------#


# ----------------- FastAPI -----------------#
app = FastAPI(title="Molar Support with FastAPI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------- FastAPI POST API Endpoints -----------------#
@app.post("/check_m3_input")
async def check_m3_input(fileb: Annotated[UploadFile, File()]):
    try:
        from modules import inputClassify

        # Specify image file path
        image_path = "input_images/m3_cbct/m3_temp.jpg"

        with open(image_path, "wb") as buffer:
            buffer.write(await fileb.read())

        # Perform input classification on the input image and get the return message
        return_message = inputClassify.predict_input_validty(
            image_path, inputValidityModel
        )

        return JSONResponse(content=return_message)

    except:
        return JSONResponse(
            content={"error": "Error in checking the validity of the image."}
        )


@app.post("/check_mc_input")
async def check_mc_input(fileb: Annotated[UploadFile, File()]):
    try:
        from modules import inputClassify

        # Specify image file path
        image_path = "input_images/mc_cbct/mc_temp.jpg"

        with open(image_path, "wb") as buffer:
            buffer.write(await fileb.read())

        # Perform input classification on the input image and get the return message
        return_message = inputClassify.predict_input_validty(
            image_path, inputValidityModel
        )

        return JSONResponse(content=return_message)

    except:
        return JSONResponse(
            content={"error": "Error in checking the validity of the image."}
        )


@app.post("/start_process")
async def start_process():

    # try:
    # Specify image file path
    m3_image_path = "input_images/m3_cbct/m3_temp.jpg"
    mc_image_path = "input_images/mc_cbct/mc_temp.jpg"

    # Perform segmentation on M3 images
    from modules import m3predictSegment as m3Segment

    segmented_m3_image_path = m3Segment.load_model_and_predict(m3_image_path)

    # Preprocess the M3 segmentation image
    from modules import deepCleaning

    preprocessed_m3_image_path = deepCleaning.process_image(segmented_m3_image_path)

    # Overlay the M3 segmentation image on the original M3 image
    from modules import overlayUtils

    initial_overlayed_image_path = overlayUtils.overlay_images(
        preprocessed_m3_image_path, m3_image_path
    )
    # Overlay the MC image on the initial overlayed image
    overlayed_image_path = overlayUtils.overlay_result_mc(
        initial_overlayed_image_path, mc_image_path
    )
    # Enhance the final overlayed image
    from modules import enhance

    enhanced_image_path = enhance.enhance_colors(overlayed_image_path)
    # Flatten violet to color
    flattened_image_path = enhance.flat_violet_to_color(
        enhanced_image_path, (128, 47, 128)
    )

    # Call the corticilization prediction model
    from modules import corticalPrediction as cortiPredict

    corticalization = cortiPredict.predict_cortical(cortical_model)

    # Call distance prediction model
    from modules import distancePrediction as distancePredict

    distance = distancePredict.detect_objects()

    # Call the position prediction model
    from modules import positionPredict as predictPos

    position = predictPos.predict_position(position_model)

    # Call the relation model
    from modules import classifyUtils as classify

    relation = classify.classify_relation(distance, position, corticalization)

    risk = classify.classify_risk(relation)

    distance = str(distance)
    distance = distance + " mm"

    # Store the images to the a session folder
    from modules import createSessionFolder as createSession
    session_folder, session_dir = createSession.createSessionFolder()


    # Saving the values to sqlite db
    image_with_distance = "output_images/distance_ouput/output_with_distance.jpg"
    
    # Convert all of the needed values to string
    session_id = str(cuid.cuid())
    session_folder = str(session_folder)
    image_with_distance = str(image_with_distance)
    corticalization = str(corticalization)
    position = str(position)
    distance = str(distance)
    relation = str(relation)
    risk = str(risk)
    
    new_case = MolarCase(
        session_id=session_id,
        session_folder=session_folder,
        final_img_filename=image_with_distance,
        corticalization=corticalization,
        position=position,
        distance=distance,
        relation=relation,
        risk=risk,
    )

    create_case(new_case)

    # Clean the contens of the "output_images" folder
    from modules import cleanDirectories as cleanDirs
    cleanDirs.clean_directories()

    # return all of the class results
    return JSONResponse(
        content={
            "corticalization": corticalization,
            "position": position,
            "relation": relation,
            "risk": risk,
            "distance": distance,
        }
    )

    # except:
    # return JSONResponse(content={"error": "Error in performing segmentation on the input images."})


# ----------------- FastAPI POST API Endpoints -----------------#

# ----------------- FastAPI GET API Endpoint -----------------#
from typing import List


# Get endpoint for a single molar case based on the session_id
@app.get("/molarcases/{session_id}", response_model=MolarCase)
async def get_molar_case(session_id: str):
    cases = get_all_cases()
    for case in cases:
        if case[0] == session_id:
            return {
                "session_id": case[0],
                "session_folder": case[1],
                "final_img_filename": case[2],
                "corticalization": case[3],
                "position": case[4],
                "distance": case[5],
                "relation": case[6],
                "risk": case[7],
            }
    raise HTTPException(status_code=404, detail="Case not found")

# GET ENDPOINT FOR SHOWING ALL ENTRIES IN SQLITE DATABASE
@app.get("/molarcases", response_model=List[MolarCase])
async def get_molar_cases():
    cases = get_all_cases()
    # Return all of the molar case values in JSON format
    return [
        {
            "session_id": case[0],
            "session_folder": case[1],
            "final_img_filename": case[2],
            "corticalization": case[3],
            "position": case[4],
            "distance": case[5],
            "relation": case[6],
            "risk": case[7],
        }
        for case in cases
    ]


# ----------------- FastAPI GET API Endpoint -----------------#


# ----------------- FastAPI DELETE API Endpoint -----------------#


# DELETE endpoint to delete the molarcases table
@app.delete("/molarcases/table")
async def delete_molarcases_table():
    delete_table()
    return {"message": "molarcases table deleted successfully"}


# ----------------- FastAPI DELETE API Endpoint -----------------#

# ----------------- FastAPI -----------------#
