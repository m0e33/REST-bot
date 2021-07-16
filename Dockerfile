ARG PREBUILD_IMAGE
FROM $PREBUILD_IMAGE:latest

# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin

RUN pip install -U pip && mkdir -p /app
RUN pip install --upgrade pip
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
COPY /kubeflow_utils /app
COPY . /app

RUN gcloud auth configure-docker --quiet
RUN gcloud auth activate-service-account --key-file="./gcp-bakdata-kubeflow-cluster.json" --quiet
