FROM continuumio/anaconda3:2023.03-1
#LABEL authors="David Kadish"

ENV PROFILE=single
ENV PROFILE_PATH=./profiles/local/$PROFILE
WORKDIR /code

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# The code to run when container is started:
COPY scripts/process_files.py /code/process_files.py
COPY src/soundade/ /code/soundade/
COPY profiles/ /code/profiles/

ENTRYPOINT conda run -n soundade xargs -a $PROFILE_PATH python ./process_files.py
