# Description: Dockerfile for the FastAPI app


# Select the base image python:3.10-slim-bookworm
FROM python:3.10-slim-bookworm

# Initialize the working directory
WORKDIR /fastapiapp

# Copy the requirements file
COPY ./requirements.txt /fastapiapp/requirements.txt

# Update the pip
RUN pip install --upgrade pip

# Install the requirements
RUN pip install --no-cache-dir -r /fastapiapp/requirements.txt

# Install pytorch
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy the rest of the files
COPY ./ /fastapiapp/

#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]