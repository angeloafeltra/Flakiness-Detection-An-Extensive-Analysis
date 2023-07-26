FROM python:3.10.12-bookworm

WORKDIR /RepoMiner
RUN mkdir -p ../spazioCondiviso
RUN mkdir -p ../spazioCondiviso/Dataset
RUN mkdir -p ../spazioCondiviso/MetricsDetector
RUN mkdir -p ../spazioCondiviso/Repository
COPY Cloner.py DatasetGenerator.py main.py FlakeFlagger.csv testConnessione.py ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas tqdm requests pydriller


