FROM python:3.8.2
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python","main.py"]
