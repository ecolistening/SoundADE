import argparse
import os
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def mp3_to_wav(mp3_file: Path, src: Path, dest: Path):
    rel_path = mp3_file.relative_to(src)
    wav_file = dest / rel_path.with_suffix(".wav")
    wav_file.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-y", "-i", str(mp3_file), "-acodec", "pcm_s16le", str(wav_file)]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return mp3_file, wav_file

def main(
    src: str | Path,
    dest: str | Path,
    cores: int,
) -> None:
    out_dir.mkdir(exist_ok=True)
    mp3_files = [(file_path, src, dest) for file_path in list(in_dir.glob("**/*.[mM][pP]3"))]

    log.info(f"Converting {len(mp3_files)} MP3 files using {num_threads} threads...")

    with ThreadPoolExecutor(max_workers=cores) as executor:
        futures = [executor.submit(mp3_to_wav, mp3) for mp3 in mp3_files]
        for future in as_completed(futures):
            mp3_name, wav_name = future.result()
            log.info(f"{mp3_name} -> {wav_name}")

    log.info("All conversions complete!")

def get_base_parser():
    parser = argparse.ArgumentParser(
        description="Recursively convert .mp3 files to .wav and store in target directory",
        add_help=False,
    )
    parser.add_argument(
        "--src",
        type=lambda p: Path(p),
        help="/path/to/source/**/*.mp3",
    )
    parser.add_argument(
        "--dest",
        type=lambda p: Path(p),
        help="/path/to/dest/**/*.wav",
    )
    parser.add_argument(
        "--cores",
        type=int,
        help="Number of workers",
    )
    parser.set_defaults(func=main, **{
        cores: os.cpu_count()
    })

if __name__ == '__main__':
    parser = get_base_parser()
    args = parser.parse_args()
    log.info(args)
    args.func(**vars(args))
