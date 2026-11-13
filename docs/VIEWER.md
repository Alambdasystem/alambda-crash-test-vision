# Interactive Crash Lab

Explore the crash-test example yourself: select an exact source frame, slow the playback, inspect the contact crop, and show or hide the recorded vehicle box and box-center trail.

## Open the portable viewer

1. Open [viewer.html](viewer.html).
2. Use GitHub's **Download raw file** control to save it as an `.html` file.
3. Open the saved file in a modern browser.

The viewer embeds all 180 recorded frames and their tracking observations. Its playback and overlay controls work without a server or an internet connection. Repository, community, attribution, and download links still need internet access.

[Download the viewer source package](viewer-source.zip?raw=true) to edit the HTML, CSS, JavaScript, or the Python exporter. The package includes the source excerpt, recorded tracks, frame sheets, and the existing Python-demo download.

## Controls

- Play or pause; choose 1×, 0.5×, or 0.25× **source playback**.
- Move one frame at a time or scrub the source-frame timeline.
- Jump to the manually reviewed contact marker at frame 13.
- Compare the full frame with a synchronized crop of the contact area.
- Toggle the detection box and observed box-center trail.
- Review the full clip or loop frames 0–73, the interval used by the original replay.

## What you are viewing

The original source excerpt is 180 frames at 30000/1001 fps. Indices run from **0 to 179**, and the clip lasts **6.006 seconds**. Its last frame begins at approximately **5.973 seconds**. The source is already slow motion, so the browser's timing is a playback clock, not physical crash time.

The contact marker is **manually reviewed frame 13 ±3 frames**, corresponding to frames **10–16 inclusive**. It is approximate and distinct from the camera's burned-in time-zero near frame 16. It does not come from an automatic collision detector.

Boxes and detection scores are recorded outputs from the existing OpenCV/YOLOX pipeline. The trail uses box centers in original image coordinates. It is not a vehicle center-of-mass trajectory; crushing changes the box geometry. No physical speed, acceleration, impact force, injury, fault, or causation is calculated.

The portable viewer uses spatially resized, compressed copies of the original decoded frames. No temporal frames are generated or interpolated.

## Modify and rebuild

Unzip `viewer-source.zip`, then follow its `README.md`. It contains a buildless `dist/` application and two Python tools:

- `tools/export_viewer_assets.py` rebuilds the frame sheets and case data from this repository's source excerpt and tracking records.
- `tools/bundle_portable_viewer.py` embeds the application and frames into a single HTML file.

The included contact annotation and crop belong to this reviewed NHTSA example. When adapting the pipeline to different footage, review and replace those values before describing them as the contact window for another clip.

## Build with us

[Join the free ArchTek AI community](https://www.skool.com/archtek-ai-6649/about) to share your experiment or suggest the next subject to track.

Footage: NHTSA via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:NHTSA_Crash_Test_of_the_2006_Honda_Ridgeline.ogv), public domain per the source page. See the project's [credits and licenses](../THIRD_PARTY.md).
