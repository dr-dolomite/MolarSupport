# Molar Support Frontend and Backend

This is a simple web application that uses machine learning models to do segmentations, classifications, and predictions on CBCT slice images of M3 and MC to assess the risk based on Maliogne's classification. The frontend is built using NextJS and the backend is built using FastAPI. The machine learning models are built using Tensorflow and Pytorch.

<br/>

## Installation

<br/>

### For Frontend

Make sure to have Node.js installed

```bash
npm install
```

<br/>

### For Backend

Make sure to have a WSL2 Ubuntu installed 

<br/>

Create a new virtual environment using Anaconda

```bash
conda create --name FASTAPI python=3.9 -y
```


Activate the environment

```bash
conda activate FASTAPI
```


Install the required packages

```bash
pip install -r requirements.txt
```


Create the model folder
    
```bash
mkdir modules/model_checkpoint
```

Paste the model checkpoints

<br/>

## Usage

Run the API

```bash
uvicorn app:app --reload
```

<br/>

## Changes

- Ported to WSL2 Ubuntu
- Used newer Tensorflow version available for Linux
- Used Pytorch based on 12.1 CUDA
- Merged Rod's changes for database handling
- Removed id and changed it to generate session_id through cuid()
- Added a new endpoint for getting a speficic case based on session_id
- Removed unused values in the database

<br/>

## Contributing

- Rod Lester Moreno
- Russel Yasol
- Pio Lawrence Burgos
- Melvin Saracin

<br/>

## License
[GNU AGPLv3](https://choosealicense.com/licenses/agpl-3.0/)
