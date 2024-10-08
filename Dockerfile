FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

WORKDIR /model

COPY . .

RUN pip install -r requirements.txt

CMD ["python", "main.py"]