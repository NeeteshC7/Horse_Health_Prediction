FROM python:3.8

WORKDIR /app

COPY . /app

RUN apt update -y && apt install awscli -y

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]