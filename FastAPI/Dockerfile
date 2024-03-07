#
FROM python:3.9

#
WORKDIR /FASTAPI


# COPY requirements.txt ./

#
RUN pip install --upgrade pip
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install pydantic
RUN pip install python-multipart
RUN pip install tensorflow[and-cuda]
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install opencv-python
RUN pip install pandas

#
COPY . .

#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]