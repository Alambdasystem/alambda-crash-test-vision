# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Alambda Systems
"""Verify or download the exact footage and YOLOX model used by this demo."""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/object_detection_yolox/object_detection_yolox_2022nov.onnx"
)
MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Source metadata must contain a lowercase SHA-256 digest.")
    return value


def download_verified(url, destination, expected):
    """Replace a destination only after the complete download passes its hash."""
    expected = checked_digest(expected)
    if destination.is_file() and sha256(destination) == expected:
        print(f"Verified {destination.name}", flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "Alambda-Crash-Test-Vision/1.0"}
    )
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", suffix=".part", delete=False
    ) as output:
        temporary = Path(output.name)
    try:
        print(f"Downloading {destination.name}", flush=True)
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
        actual = sha256(temporary)
        if actual != expected:
            raise RuntimeError(
                f"Checksum mismatch for {destination.name}: expected {expected}, got {actual}. "
                "The existing destination was left unchanged."
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified {destination.name}", flush=True)


def verify_clip_metadata(path, expected):
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ]
    probe = json.loads(subprocess.check_output(command, text=True))
    videos = [s for s in probe["streams"] if s.get("codec_type") == "video"]
    if len(videos) != 1 or any(s.get("codec_type") == "audio" for s in probe["streams"]):
        raise RuntimeError("The extracted sample must have one video stream and no audio.")
    stream = videos[0]
    values = {
        "width": stream["width"], "height": stream["height"],
        "frame_count": int(stream["nb_read_frames"]), "codec": stream["codec_name"],
        "pixel_format": stream["pix_fmt"],
    }
    for name, actual in values.items():
        if actual != expected[name]:
            raise RuntimeError(f"Extracted {name}: expected {expected[name]}, got {actual}.")
    if Fraction(stream["avg_frame_rate"]) != Fraction(expected["fps_rational"]):
        raise RuntimeError("The extracted frame rate differs from assets/source.json.")
    if not math.isclose(float(probe["format"]["duration"]), expected["duration_seconds"], abs_tol=.001):
        raise RuntimeError("The extracted duration differs from assets/source.json.")


def ensure_source(metadata):
    clip = metadata["clip"]
    expected = checked_digest(clip["sha256"])
    relative_path = Path(clip["path"])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("The clip path must be relative to the project directory.")
    destination = ROOT / relative_path
    if destination.is_file() and sha256(destination) == expected:
        print(f"Verified {clip['path']} (included excerpt)", flush=True)
        return
    for program in ("ffmpeg", "ffprobe"):
        if not shutil.which(program):
            raise RuntimeError(f"Install {program} to recreate the sample excerpt.")
    original = metadata["original"]
    name = original["file_name"]
    if Path(name).name != name:
        raise ValueError("The original filename must be a basename.")
    original_path = ASSETS / name
    original_digest = checked_digest(original["sha256"])
    if original_digest != metadata["original_sha256"]:
        raise ValueError("Conflicting original checksums in assets/source.json.")
    download_verified(metadata["source_url"], original_path, original_digest)

    arguments = metadata["extraction"]["ffmpeg_arguments"]
    if not isinstance(arguments, list) or not all(isinstance(s, str) for s in arguments):
        raise ValueError("The extraction arguments must be a list of strings.")
    if arguments.count("ORIGINAL.ogv") != 1 or arguments[-1] != clip["path"]:
        raise ValueError("Unexpected extraction input/output placeholders in assets/source.json.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix="source-extraction-", suffix=".mp4", delete=False
    ) as output:
        temporary = Path(output.name)
    try:
        command = ["ffmpeg", "-nostdin", "-y"] + [
            str(original_path) if arg == "ORIGINAL.ogv" else arg for arg in arguments[:-1]
        ] + [str(temporary)]
        print("Extracting the source segment recorded in assets/source.json", flush=True)
        subprocess.run(command, check=True)
        verify_clip_metadata(temporary, clip)
        actual = sha256(temporary)
        if actual != expected:
            raise RuntimeError(
                f"Re-encoded excerpt checksum differs: expected {expected}, got {actual}. "
                "Metadata matches, but FFmpeg/encoder versions can change compressed bytes. "
                "Use the included checksum-verified source.mp4 or the documented FFmpeg build. "
                "The existing destination was left unchanged."
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified {clip['path']} (recreated excerpt)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-only", action="store_true", help="Verify or restore the footage without downloading the model."
    )
    args = parser.parse_args()
    manifest = ASSETS / "source.json"
    if not manifest.is_file():
        raise FileNotFoundError("assets/source.json is required; restore the project's source metadata.")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    ASSETS.mkdir(parents=True, exist_ok=True)
    ensure_source(metadata)
    if not args.source_only:
        download_verified(MODEL_URL, ASSETS / "yolox_s.onnx", MODEL_SHA256)


if __name__ == "__main__":
    main()
