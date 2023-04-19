# FROM python:3.8-alpine
FROM python:latest
COPY ./requirements.txt /app/requirements.txt


# Install native libraries, required for numpy for 3.8-alpine to work
# RUN apk --no-cache add musl-dev linux-headers g++

# Upgrade pip
RUN pip install --upgrade pip

WORKDIR /app


RUN pip install -r requirements.txt
COPY . /app
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]