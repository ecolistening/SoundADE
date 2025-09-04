import os
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cli import index_sites
from cli import index_audio
from cli import index_solar
from cli import index_weather
from cli import acoustic_features
from cli import birdnet_species
from cli import pipeline

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="SoundADE Acoustics Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command")

    pipeline_steps = [
        index_sites,
        index_audio,
        index_solar,
        index_weather,
        acoustic_features,
        birdnet_species,
        pipeline,
    ]
    for module in pipeline_steps:
        module.register_subparser(subparsers)

    args = parser.parse_args()
    log.info(args)

    if hasattr(args, "func"):
        args.func(**vars(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
