FROM continuumio/anaconda3
#LABEL authors="David Kadish"

WORKDIR /code

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "soundade", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure pandas is installed:"
RUN python -c "import pandas"

# The code to run when container is started:
COPY data/ /code/data/
COPY scripts/ /code/scripts/
COPY src/soundade /code/src/soundade

SHELL ["conda", "develop", "-n", "soundade", "/code/src"]