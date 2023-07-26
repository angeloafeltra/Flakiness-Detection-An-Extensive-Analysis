FROM python:3.10.12-bookworm

WORKDIR /RepoMiner
COPY spazioCondiviso ./spazioCondiviso
COPY Cloner.py DatasetGenerator.py main.py ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas tqdm requests pydriller


