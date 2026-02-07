FROM apache/airflow:2.10.2
USER root
RUN apt-get update && apt-get install -y gcc
USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
