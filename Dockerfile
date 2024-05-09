FROM python:3.10

RUN apt update && apt install -y libgl1-mesa-dev

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]
