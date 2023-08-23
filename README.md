# SoundADE
Acoustic Descriptor Extraction tool for processing sound on High Performance Computing clusters


## Installation

`conda env create -f environment.yaml`

### Troubleshooting

#### Solving Environment | Killed
Sometimes the environement solving step can take up too much memory (especially for a login node on a cluster). This can be fixed by removing channels from the YAML file. For me, removing `anaconda` and `defaults` did the trick.

## Docker
