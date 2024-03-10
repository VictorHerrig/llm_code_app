FROM python:3.10

WORKDIR /code

COPY ./config /code/config

COPY ./src /code/src

RUN pip install --no-cache-dir ./src

CMD ["uvicorn", "codegen_backend.api:app", "--host", "0.0.0.0", "--port", "80"]
