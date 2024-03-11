FROM python:3.12

WORKDIR /app

COPY ./codegen_frontend .

RUN pip install -r requirements.txt

RUN reflex init --template blank

# Download all npm dependencies and compile frontend
RUN reflex export --no-zip

# Needed until Reflex properly passes SIGTERM on backend.
STOPSIGNAL SIGKILL

# Always apply migrations before starting the backend.
CMD [ -d alembic ] && reflex db migrate; reflex run --env prod
