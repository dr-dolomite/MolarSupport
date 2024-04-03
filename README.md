# Molar Support Frontend and Backend

This is a simple web application that uses machine learning models to do segmentations, classifications, and predictions on CBCT slice images of M3 and MC to assess the risk based on Maliogne's classification. The frontend is built using NextJS and the backend is built using FastAPI. The machine learning models are built using Tensorflow and Pytorch.

Contact Russel Yasol for the models. 
russel.yasol@wvsu.edu.ph

<br/>

## Installation

<br/>

### For Frontend

Make sure to have Node.js installed. Go to MolarSupport root directory and run

```bash
npm install
```

<br/>

### For Backend

Make sure to have a a Docker Desktop Installed (and WSL2 if needed)

<br/>

Create the model folder at FastAPI root directory
    
```bash
mkdir modules/model_checkpoint
```

Paste the model checkpoints

Build the Docker Image. (For Linux, rename WSL2_requirements.txt to requirements.txt)

```bash
docker build -t molar-app . 
```

<br/>

## Usage

Run the Frontend

```bash
npm run dev
```

Run the Docker Container

```bash
docker run -p 8000:8000 molar-app
```

To exit

```bash
Press Ctrcl + C
wsl --shutdown
```

<br/>

## Changes


(For WSL2_requirements.txt)
- Added requirements.txt to support Windows by default
- Added WSL2_requirements.txt to support Linux
- Ported to WSL2 Ubuntu
- Used newer Tensorflow version available for Linux
- Used Pytorch based on 12.1 CUDA
- Merged Rod's changes for database handling
- Removed id and changed it to generate session_id through cuid()
- Added a new endpoint for getting a speficic case based on session_id
- Removed unused values in the database

<br/>

## Contributing

- Dianne Ritz Lapasaran
- Rod Lester Moreno
- Russel Yasol
- Pio Lawrence Burgos
- Melvin Saracin
- Neil Clarence Diaz

<br/>

## License
[GNU AGPLv3](https://choosealicense.com/licenses/agpl-3.0/)
