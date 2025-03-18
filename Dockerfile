FROM continuumio/anaconda3:2023.03-1
#LABEL authors="David Kadish"

ENV PROFILE=single
ENV BASE_PATH=/code
ENV PROFILE_PATH=$BASE_PATH/profiles/local/$PROFILE
WORKDIR /code

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# The code to run when container is started:
COPY scripts/process_files.py /code/scripts/process_files.py
COPY src/soundade/ /code/src/soundade/
COPY profiles/ /code/profiles/
COPY run-pipeline.sh /code/run-pipeline.sh

ENTRYPOINT bash run-pipeline.sh
