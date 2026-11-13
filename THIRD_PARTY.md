# Third-party materials

## NHTSA crash-test footage

The sample is derived from [NHTSA Crash Test of the 2006 Honda Ridgeline](https://commons.wikimedia.org/wiki/File:NHTSA_Crash_Test_of_the_2006_Honda_Ridgeline.ogv), with the [original OGV file](https://upload.wikimedia.org/wikipedia/commons/5/54/NHTSA_Crash_Test_of_the_2006_Honda_Ridgeline.ogv) hosted by Wikimedia Commons.

The Commons file page identifies **NHTSA** as the author and labels the recording **public domain in the United States** as work created by an employee of the U.S. Department of Transportation during official duties (`PD-USGov-DOT`). This attribution describes the source's stated status; the project does not claim ownership of the underlying footage. The Apache 2.0 license applies to this project's code and does not imply an endorsement by NHTSA, Honda, or Wikimedia.

`assets/source.json` records the exact download, checksums, selected segment, and manually reviewed contact annotation. Source timestamps represent playback time in the hosted recording. They are not a measurement of real elapsed crash time; the recording already contains slow-motion imagery.

## YOLOX model and decoding example

`process_crash.py` adapts the grid/stride decoding approach from the [OpenCV Zoo YOLOX example](https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox). The pretrained model is `object_detection_yolox_2022nov.onnx`, retrieved from the official OpenCV Zoo repository. The downloader verifies its SHA-256:

```text
c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063
```

The model directory is licensed under Apache 2.0. A copy of its license is included in [assets/YOLOX_LICENSE.txt](assets/YOLOX_LICENSE.txt). The large model weights are downloaded separately.

## Runtime dependencies

OpenCV, NumPy, Pillow, and FFmpeg remain subject to their respective upstream licenses. Python requirements are declared in `requirements.txt`; FFmpeg and its H.264 encoder are separate system dependencies. Their packages and binaries are not vendored here.
