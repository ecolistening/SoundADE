FROM continuumio/anaconda3:2023.03-1

ENV CODE_PATH=/code
ENV DATA_PATH=/data
ENV PROFILE_PATH=$DATA_PATH/run-environment/profile
ENV GIT_COMMIT="Docker doesn't know"
WORKDIR /code

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml
COPY src/soundade/ /code/src/soundade/
COPY pyproject.toml /code/pyproject.toml
RUN conda run -n soundade python -m pip install /code

# The code to run when container is started:
COPY scripts/process_files.py /code/scripts/process_files.py
COPY pipeline-steps.sh /code/pipeline-steps.sh

ENTRYPOINT exec /code/pipeline-steps.sh
