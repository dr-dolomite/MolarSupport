# FastAPI router for the API with SQLite database

This is a simple API that uses FastAPI and SQLite to store and retrieve data.



## Installation


### For Frontend

Make sure to have Node.js installed

```bash
npm install
```


### For Backend

Make sure to have a WSL2 Ubuntu installed 

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



## Usage

Run the API

```bash
uvicorn app:app --reload
```



## Changes

- Ported to WSL2 Ubuntu
- Used newer Tensorflow version available for Linux
- Used Pytorch based on 12.1 CUDA
- Merged Rod's changes for database handling
- Removed id and changed it to generate session_id through cuid()
- Added a new endpoint for getting a speficic case based on session_id
- Removed unused values in the database



## Contributing

Rod Lester Moreno
Russel Yasol

## License
[GNU AGPLv3](https://choosealicense.com/licenses/agpl-3.0/)
