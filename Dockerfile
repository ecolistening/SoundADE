FROM continuumio/anaconda3:2023.03-1
#LABEL authors="David Kadish"

ENV PROFILE=small
ENV PROFILE_PATH=./profiles/$PROFILE
WORKDIR /code

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml
RUN conda develop -n soundade /workspaces/SoundADE/src/

# Replace the RUN command macro to use the new environment:
SHELL ["conda", "run", "-n", "soundade", "-v", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure pandas is installed:"
RUN python -c "import pandas"

# The code to run when container is started:
COPY scripts/process_files.py /code/process_files.py
COPY src/soundade/ /code/soundade/
COPY profiles/ /code/profiles/

ENTRYPOINT conda run -n soundade xargs -a $PROFILE_PATH python ./process_files.py
