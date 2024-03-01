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

import os
import shutil

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

import datetime

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
    corticalization: str
    position: str
    distance: str
    relation: str
    risk: str
    date: str

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
                        corticalization TEXT NOT NULL,
                        position TEXT NOT NULL,
                        distance FLOAT NOT NULL,
                        relation TEXT NOT NULL,
                        risk TEXT NOT NULL,
                        date TEXT NOT NULL
                );
                    """
    )
    connection.commit()
    connection.close()


def create_case(case: MolarCase):  # (CRUD) Create molar case
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute(
        "INSERT INTO molarcases (session_id, session_folder, corticalization, position, distance, relation, risk, date) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            case.session_id,
            case.session_folder,
            case.corticalization,
            case.position,
            case.distance,
            case.relation,
            case.risk,
            case.date,
        ),
    )
    connection.commit()
    connection.close()


# Function to check if the first row is empty
def check_first_row():
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM molarcases LIMIT 1")
    row = cursor.fetchone()
    if not row:
        message = 0
        return message
    else:
        message = 1
        return message


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


# Function for deleting all of molarcases values
def delete_all_cases():
    
    # delete the temp-result folder
    temp_result_folder = "../public/temp-result"
    if os.path.exists(temp_result_folder):
        shutil.rmtree(temp_result_folder)
        
        # create the folder again
        os.mkdir(temp_result_folder)
    
    
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute("DELETE FROM molarcases")
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
@app.post("/api/check_m3_input")
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


@app.post("/api/check_mc_input")
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


@app.post("/api/start_process")
async def start_process():

    create_table()

    # try:
    # Specify image file path
    m3_image_path = "input_images/m3_cbct/m3_temp.jpg"
    mc_image_path = "input_images/mc_cbct/mc_temp.jpg"

    # Generate a session ID
    # Convert all of the needed values to string
    session_id = str(cuid.cuid())

    # if not os.path.exists(m3_image_path) or not os.path.exists(mc_image_path):
    #     from modules import cleanDirectories as cleanDirs

    #     cleanDirs.clean_directories()
    #     return JSONResponse(content={"error": "Error in finding the input images."})

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

    distance = distancePredict.detect_objects(session_id)

    # Call the position prediction model
    from modules import positionPredict as predictPos

    position = predictPos.predict_position(position_model)

    # Call the relation model
    from modules import classifyUtils as classify

    relation = classify.classify_relation(distance, position, corticalization)

    risk = classify.classify_risk(relation)

    distance = str(distance)
    distance = distance + " mm"

    # Saving the values to sqlite db
    image_with_distance = "output_images/distance_ouput/output_with_distance.jpg"

    # Store the images to the a session folder
    from modules import createSessionFolder as createSession

    session_folder = createSession.createSessionFolder(session_id)

    session_folder = str(session_folder)
    image_with_distance = str(image_with_distance)
    corticalization = str(corticalization)
    position = str(position)
    distance = str(distance)
    relation = str(relation)
    risk = str(risk)

    date = datetime.datetime.now().strftime("%Y-%m-%d")

    new_case = MolarCase(
        session_id=session_id,
        session_folder=session_folder,
        corticalization=corticalization,
        position=position,
        distance=distance,
        relation=relation,
        risk=risk,
        date=date,
    )

    create_case(new_case)

    # Clean the contens of the "output_images" folder
    from modules import cleanDirectories as cleanDirs

    cleanDirs.clean_directories()

    # return all of the class results
    return JSONResponse(
        content={
            "session_id": session_id,
            "corticalization": corticalization,
            "position": position,
            "relation": relation,
            "risk": risk,
            "distance": distance,
            "date": date,
        }
    )

    # except:
    # return JSONResponse(content={"error": "Error in performing segmentation on the input images."})


# ----------------- FastAPI POST API Endpoints -----------------#

# ----------------- FastAPI GET API Endpoint -----------------#
from typing import List


# Get endpoint for a single molar case based on the session_id
@app.get("/api/molarcase/{session_id}", response_model=MolarCase)
async def get_molar_case(session_id: str):
    cases = get_all_cases()
    for case in cases:
        if case[0] == session_id:
            return {
                "session_id": case[0],
                "session_folder": case[1],
                "corticalization": case[2],
                "position": case[3],
                "distance": case[4],
                "relation": case[5],
                "risk": case[6],
                "date": case[7],
            }
    raise HTTPException(status_code=404, detail="Case not found")


# GET ENDPOINT FOR SHOWING ALL ENTRIES IN SQLITE DATABASE
@app.get("/api/molarcases", response_model=List[MolarCase])
async def get_molar_cases():
    try:
        # Check if the table exists
        if not os.path.exists("db/molarcases.db"):
            raise HTTPException(
                status_code=404, detail="The molarcases table does not exist."
            )

        # Check the first row if empty
        if check_first_row() == 0:
            raise HTTPException(
                status_code=404, detail="The molarcases table is empty."
            )

        # If there are entries in the molarcases table
        cases = get_all_cases()
        if cases:
            return [
                {
                    "session_id": case[0],
                    "session_folder": case[1],
                    "corticalization": case[2],
                    "position": case[3],
                    "distance": case[4],
                    "relation": case[5],
                    "risk": case[6],
                    "date": case[7],
                }
                for case in cases
            ]
        else:
            raise HTTPException(status_code=404, detail="No data available")
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred: " + str(e))


@app.get("/api/check_if_both_images_exist")
async def check_if_both_images_exist():
    m3_image_path = "input_images/m3_cbct/m3_temp.jpg"
    mc_image_path = "input_images/mc_cbct/mc_temp.jpg"

    if not os.path.exists(m3_image_path) or not os.path.exists(mc_image_path):

        return {"error": "Error in finding the input images."}

    return {"success": "Both images exist."}


# ----------------- FastAPI GET API Endpoint -----------------#


# ----------------- FastAPI DELETE API Endpoint -----------------#


@app.delete("/api/delete_temp_images")
async def delete_temp_images():
    m3_image_path = "input_images/m3_cbct/m3_temp.jpg"
    mc_image_path = "input_images/mc_cbct/mc_temp.jpg"

    if os.path.exists(m3_image_path) and os.path.exists(mc_image_path):
        os.remove(m3_image_path)
        os.remove(mc_image_path)
        return {"message": "Temp images deleted successfully"}

    return {"message": "Temp images not found"}


# DELETE endpoint to delete the molarcases table
@app.delete("/api/molarcases/delete")
async def delete_molarcases_table():
    # delete_table()
    delete_all_cases()
    return {"message": "molarcases table deleted successfully"}


# ----------------- FastAPI DELETE API Endpoint -----------------#

# ----------------- FastAPI Misc Routes -----------------#

# Route for sample cases
@app.get("/api/sample_cases/{id}")
async def sample_cases(id: str):
    if id == "1":
        return (
            {
                "session_id": "1",
                "corticalization": "Negative",
                "position": "Lingual",
                "distance": "0 mm",
                "relation": "Class 2B",
                "risk": "N.1 (Low)",
            }
        )
    elif id == "2":
        return (
            {
                "session_id": "2",
                "corticalization": "Negative",
                "position": "Apical",
                "distance": "5.5 mm",
                "relation": "Class 1A",
                "risk": "N.1 (Low)",
            }
        )
    elif id == "3":
        return (
            {
                "session_id": "3",
                "corticalization": "Negative",
                "position": "Lingual",
                "distance": "0.64 mm",
                "relation": "Class 2B",
                "risk": "N.1 (Low)",
            }
        )
    else :
        return (
            {
                "session_id": "4",
                "corticalization": "Positive",
                "position": "Lingual",
                "distance": "0 mm",
                "relation": "Class 4B",
                "risk": "N.3 (High)",
            }
        )

# ----------------- FastAPI -----------------#
